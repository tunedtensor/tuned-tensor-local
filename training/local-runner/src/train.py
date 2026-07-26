from __future__ import annotations

import json
import os
import tarfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from model_contract import (
    CERTIFIED_BASE_MODEL,
    assert_certified_model_config,
)
from sft_data import IGNORE_INDEX, build_assistant_only_example


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
    raw = json.loads(HYPERPARAMETERS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Hyperparameters must be a JSON object")
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


def model_revision_kwargs(model_source: str) -> dict[str, str]:
    revision = hp("base_model_revision")
    return {"revision": revision} if revision and not Path(model_source).exists() else {}


def assert_certified_request() -> None:
    requested_model = hp("base_model", CERTIFIED_BASE_MODEL)
    if requested_model != CERTIFIED_BASE_MODEL:
        raise ValueError(
            f"The bundled trainer currently certifies only {CERTIFIED_BASE_MODEL}; got {requested_model!r}"
        )
    loader = hp("model_loader", "causal_lm")
    if loader != "causal_lm":
        raise ValueError(f"The bundled trainer is text-only and requires model_loader=causal_lm; got {loader!r}")
def require_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TT Local bundled training requires an NVIDIA CUDA GPU. "
            "Run this workflow on DGX Spark and verify it first with tt-local doctor."
        )
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "device_index": device_index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(TRAINING_DIR.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} contains invalid JSON") from exc
            if not isinstance(row, dict) or "messages" not in row:
                raise ValueError(f"{path}:{line_number} must contain a messages array")
            rows.append(row)
    if not rows:
        raise ValueError(f"No chat JSONL rows found under {TRAINING_DIR}")
    return rows


def resolve_model_source() -> str:
    if BASE_MODEL_DIR.is_dir() and any(BASE_MODEL_DIR.iterdir()):
        return str(BASE_MODEL_DIR)
    return CERTIFIED_BASE_MODEL


class AssistantOnlyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ):
        self.examples: list[dict[str, torch.Tensor]] = []
        for row_index, row in enumerate(rows, start=1):
            try:
                tokenized = build_assistant_only_example(
                    tokenizer,
                    row["messages"],
                    max_length=max_length,
                )
            except Exception as exc:
                raise ValueError(f"Training row {row_index} is not valid Qwen SFT data: {exc}") from exc
            self.examples.append({
                key: torch.tensor(value, dtype=torch.long)
                for key, value in tokenized.items()
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence(
                [feature["input_ids"] for feature in features],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [feature["attention_mask"] for feature in features],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [feature["labels"] for feature in features],
                batch_first=True,
                padding_value=IGNORE_INDEX,
            ),
        }


def create_model_and_tokenizer(model_source: str) -> tuple[Any, Any, torch.dtype]:
    token = os.getenv("HF_TOKEN")
    source_kwargs: dict[str, Any] = {
        **model_revision_kwargs(model_source),
        "trust_remote_code": False,
        "token": token,
    }
    if not Path(model_source).exists():
        source_kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_source, **source_kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{CERTIFIED_BASE_MODEL} tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **source_kwargs,
        dtype=dtype,
        device_map={"": torch.cuda.current_device()},
    )
    assert_certified_model_config(model.config)
    model.config.use_cache = False
    return model, tokenizer, dtype


def apply_lora(model: Any) -> Any:
    config = LoraConfig(
        r=hp_int("lora_rank", 16),
        lora_alpha=hp_int("lora_alpha", 32),
        lora_dropout=hp_float("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return get_peft_model(model, config)


def create_model_archive() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(MODEL_DIR, arcname="model")
    return archive_path


def run_training(rows: list[dict[str, Any]], model_source: str) -> tuple[dict[str, Any], torch.dtype]:
    model, tokenizer, dtype = create_model_and_tokenizer(model_source)
    model = apply_lora(model)
    dataset = AssistantOnlyDataset(
        rows,
        tokenizer,
        max_length=hp_int("max_seq_length", 2048),
    )
    collator = AssistantOnlyCollator(tokenizer.pad_token_id)

    with TemporaryDirectory(prefix="tt-local-sft-") as output_dir:
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=hp_int("n_epochs", 3),
            learning_rate=hp_float("learning_rate", 0.00001),
            per_device_train_batch_size=hp_int("per_device_train_batch_size", 1),
            gradient_accumulation_steps=hp_int("gradient_accumulation_steps", 8),
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            optim="adamw_torch",
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
    return result.metrics, dtype


def main() -> None:
    started = time.time()
    assert_certified_request()
    cuda = require_cuda()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    model_source = resolve_model_source()
    metrics, dtype = run_training(rows, model_source)
    archive_path = create_model_archive()
    output_metrics = {
        "training_rows": len(rows),
        "base_model": CERTIFIED_BASE_MODEL,
        "base_model_revision": hp("base_model_revision"),
        "model_source": model_source,
        "model_loader": "causal_lm",
        "loss_mask": "final_assistant_only",
        "device": "cuda",
        "cuda_device": cuda["device_name"],
        "cuda_compute_capability": cuda["compute_capability"],
        "dtype": str(dtype).removeprefix("torch."),
        "train_runtime": round(time.time() - started, 3),
        **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "model_archive": str(archive_path),
    }
    metrics_json = json.dumps(output_metrics, indent=2)
    (MODEL_DIR / "training-metrics.json").write_text(metrics_json, encoding="utf-8")
    (OUTPUT_DIR / "training-metrics.json").write_text(metrics_json, encoding="utf-8")
    print(metrics_json)


if __name__ == "__main__":
    main()
