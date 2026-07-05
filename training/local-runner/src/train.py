from __future__ import annotations

import base64
import json
import os
import tarfile
import time
import urllib.request
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


TRAINING_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
BASE_MODEL_DIR = Path(os.environ.get("SM_CHANNEL_BASE_MODEL", "/opt/ml/input/data/base_model"))
HYPERPARAMETERS_PATH = Path(
    os.environ.get("TT_HYPERPARAMETERS_PATH", "/opt/ml/input/config/hyperparameters.json")
)
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output"))
ROW_SOURCE_DIR_KEY = "_tt_source_dir"


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
            row[ROW_SOURCE_DIR_KEY] = str(path.parent)
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


def strip_file_uri(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("file://"):
        return value[7:]
    return value


def resolve_adapter_path(value: str | None, tmp: Path) -> str | None:
    path_value = strip_file_uri(value)
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_file() and path.name.endswith(".tar.gz"):
        extracted = tmp / "parent-adapter"
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:gz") as tar:
            extracted_root = extracted.resolve()
            for member in tar.getmembers():
                destination = (extracted / member.name).resolve()
                try:
                    destination.relative_to(extracted_root)
                except ValueError:
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(extracted)
        candidates = [candidate for candidate in extracted.rglob("adapter_config.json")]
        if candidates:
            return str(candidates[0].parent)
        directories = [candidate for candidate in extracted.iterdir() if candidate.is_dir()]
        return str(directories[0] if len(directories) == 1 else extracted)
    return str(path)


def chat_template_kwargs() -> dict[str, Any]:
    """Optional kwargs forwarded to apply_chat_template (for example
    {"enable_thinking": false} for thinking-mode models), passed by the
    runner as a JSON hyperparameter so training text matches inference."""
    raw = hp("chat_template_kwargs")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def row_to_text(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = row["messages"]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs(),
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


def image_from_data_uri(value: str) -> Image.Image:
    _, encoded = value.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def load_image(value: str, base_dir: Path) -> Image.Image:
    if value.startswith("data:"):
        return image_from_data_uri(value)
    if value.startswith("http://") or value.startswith("https://"):
        with urllib.request.urlopen(value, timeout=30) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    path_value = value[7:] if value.startswith("file://") else value
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return Image.open(path).convert("RGB")


def image_value_from_part(part: dict[str, Any], fallback: str | None) -> str | None:
    for key in ("image", "path", "uri", "data_uri"):
        value = part.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def multimodal_content(
    content: Any,
    top_level_images: list[str],
    base_dir: Path,
) -> tuple[str | list[dict[str, Any]], list[Image.Image]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    images: list[Image.Image] = []
    image_index = 0
    converted: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            converted.append({"type": "text", "text": str(part)})
            continue
        if part.get("type") != "image":
            converted.append(part)
            continue
        fallback = top_level_images[image_index] if image_index < len(top_level_images) else None
        image_index += 1
        image_value = image_value_from_part(part, fallback)
        if image_value:
            images.append(load_image(image_value, base_dir))
        converted.append({"type": "image"})
    return converted, images


def row_to_multimodal_example(row: dict[str, Any]) -> dict[str, Any]:
    source_dir = Path(str(row.get(ROW_SOURCE_DIR_KEY) or TRAINING_DIR))
    top_level_images = [str(value) for value in row.get("images", []) if isinstance(value, str)]
    all_images: list[Image.Image] = []
    messages: list[dict[str, Any]] = []
    for message in row["messages"]:
        content, images = multimodal_content(message.get("content", ""), top_level_images, source_dir)
        all_images.extend(images)
        messages.append({
            "role": message.get("role", "user"),
            "content": content,
        })
    if not all_images and top_level_images:
        all_images = [load_image(value, source_dir) for value in top_level_images]
    return {
        "messages": messages,
        "images": all_images,
    }


def create_text_model_and_tokenizer(model_source: str):
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


def create_multimodal_model_and_processor(model_source: str):
    trust_remote_code = hp_bool("trust_remote_code", True)
    token = os.getenv("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        trust_remote_code=trust_remote_code,
        token=token,
        torch_dtype=dtype if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    return model, processor


def lora_target_modules(default: str) -> list[str] | str:
    raw = hp("lora_target_modules", default) or default
    if raw == "all-linear":
        return raw
    return [item.strip() for item in raw.split(",") if item.strip()]


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
        target_modules=lora_target_modules("all-linear"),
    )
    return get_peft_model(model, config)


def multimodal_lora_config() -> LoraConfig:
    return LoraConfig(
        r=hp_int("lora_rank", 16),
        lora_alpha=hp_int("lora_alpha", 32),
        lora_dropout=hp_float("lora_dropout", 0.05),
        bias="none",
        target_modules=lora_target_modules("q_proj,v_proj"),
    )


def create_model_archive() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname="model")
    return archive_path


def run_text_training(rows: list[dict[str, Any]], model_source: str) -> dict[str, Any]:
    with TemporaryDirectory() as tmp:
        parent_adapter = resolve_adapter_path(hp("parent_model_artifact"), Path(tmp))
        model, tokenizer = create_text_model_and_tokenizer(model_source)
        model = (
            PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
            if parent_adapter
            else maybe_apply_lora(model)
        )

        texts = [row_to_text(tokenizer, row) for row in rows]
        dataset = TextDataset(texts, tokenizer, max_length=hp_int("max_seq_length", 2048))
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
    return result.metrics


def run_multimodal_training(rows: list[dict[str, Any]], model_source: str) -> dict[str, Any]:
    with TemporaryDirectory() as tmp:
        parent_adapter = resolve_adapter_path(hp("parent_model_artifact"), Path(tmp))
        model, processor = create_multimodal_model_and_processor(model_source)
        if parent_adapter:
            model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
        dataset = HFDataset.from_list([row_to_multimodal_example(row) for row in rows])

        args = SFTConfig(
            output_dir=tmp,
            num_train_epochs=hp_int("n_epochs", 3),
            learning_rate=hp_float("learning_rate", 0.00001),
            per_device_train_batch_size=hp_int("per_device_train_batch_size", 1),
            gradient_accumulation_steps=hp_int("gradient_accumulation_steps", 8),
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=None,
        )
        trainer_values: dict[str, Any] = {
            "model": model,
            "args": args,
            "train_dataset": dataset,
            "processing_class": processor,
        }
        if not parent_adapter:
            trainer_values["peft_config"] = multimodal_lora_config()
        trainer = SFTTrainer(**trainer_values)
        result = trainer.train()
        trainer.save_model(MODEL_DIR)
    processor.save_pretrained(MODEL_DIR)
    return result.metrics


def main() -> None:
    started = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    model_source = resolve_model_source()
    model_loader = hp("model_loader", "causal_lm")
    if model_loader == "image_text_to_text":
        metrics = run_multimodal_training(rows, model_source)
    else:
        metrics = run_text_training(rows, model_source)

    archive_path = create_model_archive()
    output_metrics = {
        "training_rows": len(rows),
        "model_source": model_source,
        "model_loader": model_loader,
        "parent_model_artifact": hp("parent_model_artifact"),
        "train_runtime": round(time.time() - started, 3),
        **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "model_archive": str(archive_path),
    }
    (MODEL_DIR / "training-metrics.json").write_text(json.dumps(output_metrics, indent=2))
    (OUTPUT_DIR / "training-metrics.json").write_text(json.dumps(output_metrics, indent=2))
    print(json.dumps(output_metrics, indent=2))


if __name__ == "__main__":
    main()
