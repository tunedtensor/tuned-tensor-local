from __future__ import annotations

import json
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


TRAINING_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
BASE_MODEL_DIR = Path(os.environ.get("SM_CHANNEL_BASE_MODEL", "/opt/ml/input/data/base_model"))
HYPERPARAMETERS_PATH = Path(
    os.environ.get("TT_HYPERPARAMETERS_PATH", "/opt/ml/input/config/hyperparameters.json")
)
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output"))


def load_hyperparameters() -> dict[str, str]:
    if not HYPERPARAMETERS_PATH.is_file():
        return {}
    raw = json.loads(HYPERPARAMETERS_PATH.read_text())
    return {str(key): str(value) for key, value in raw.items()}


HP = load_hyperparameters()


def hp(name: str, default: str | None = None) -> str | None:
    value = os.getenv(f"SM_HP_{name.upper()}", HP.get(name, default))
    if value is None:
        return None
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def hp_int(name: str, default: int) -> int:
    return int(hp(name, str(default)) or default)


def hp_float(name: str, default: float) -> float:
    return float(hp(name, str(default)) or default)


def hp_bool(name: str, default: bool) -> bool:
    return (hp(name, str(default)) or "").lower() in {"1", "true", "yes", "y"}


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(TRAINING_DIR.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path}:{line_number} missing messages")
            rows.append(row)
    if not rows:
        raise ValueError(f"No chat JSONL rows found under {TRAINING_DIR}")
    return rows


def resolve_model_source() -> str:
    archive = BASE_MODEL_DIR / "model.tar.gz"
    if archive.is_file():
        extracted = Path("/tmp/base-model")
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extracted)
        candidates = [path for path in extracted.iterdir() if path.is_dir()]
        return str(candidates[0] if len(candidates) == 1 else extracted)
    return hp("base_model") or "Qwen/Qwen3.5-2B"


def row_to_text(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row["messages"]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


class TextDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int):
        self.examples = [
            tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            for text in texts
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.examples[index]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


def create_model_and_tokenizer(model_source: str):
    trust_remote_code = hp_bool("trust_remote_code", True)
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        token=token,
        torch_dtype=dtype if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    return model, tokenizer


def maybe_apply_lora(model: Any):
    rank = hp_int("lora_rank", 16)
    alpha = hp_int("lora_alpha", 32)
    dropout = hp_float("lora_dropout", 0.05)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return get_peft_model(model, config)


def create_model_archive() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname="model")
    return archive_path


def main() -> None:
    started = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    model_source = resolve_model_source()
    model, tokenizer = create_model_and_tokenizer(model_source)
    model = maybe_apply_lora(model)

    texts = [row_to_text(tokenizer, row) for row in rows]
    dataset = TextDataset(texts, tokenizer, max_length=hp_int("max_seq_length", 2048))
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    with TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            num_train_epochs=hp_int("n_epochs", 3),
            learning_rate=hp_float("learning_rate", 0.00001),
            per_device_train_batch_size=hp_int("per_device_train_batch_size", 1),
            gradient_accumulation_steps=hp_int("gradient_accumulation_steps", 8),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=collator,
        )
        result = trainer.train()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    archive_path = create_model_archive()
    metrics = {
        "training_rows": len(rows),
        "model_source": model_source,
        "train_runtime": round(time.time() - started, 3),
        **{key: float(value) for key, value in result.metrics.items() if isinstance(value, (int, float))},
        "model_archive": str(archive_path),
    }
    (MODEL_DIR / "training-metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUTPUT_DIR / "training-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
