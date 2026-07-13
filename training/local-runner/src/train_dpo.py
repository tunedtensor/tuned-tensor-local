from __future__ import annotations

import inspect
import json
import os
import tarfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


TRAINING_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
BASE_MODEL_DIR = Path(os.environ.get("SM_CHANNEL_BASE_MODEL", "/opt/ml/input/data/base_model"))
HYPERPARAMETERS_PATH = Path(
    os.environ.get("TT_HYPERPARAMETERS_PATH", "/opt/ml/input/config/hyperparameters.json")
)
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output"))
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_EXPANDED_BYTES = 20 * 1024 * 1024 * 1024


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


def model_revision_kwargs(model_source: str) -> dict[str, str]:
    revision = hp("base_model_revision")
    return {"revision": revision} if revision and not Path(model_source).exists() else {}


def supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    accepted = set(inspect.signature(callable_obj).parameters)
    return {key: value for key, value in values.items() if key in accepted}


def load_preference_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(TRAINING_DIR.rglob("*.jsonl")):
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            cleaned: dict[str, str] = {}
            for key in ("prompt", "chosen", "rejected"):
                value = row.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{path}:{line_number} missing non-empty string field {key}")
                cleaned[key] = value
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"No preference JSONL rows found under {TRAINING_DIR}")
    return rows


def safe_extract_archive(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"Model archive exceeds {MAX_ARCHIVE_MEMBERS} members")
        expanded_bytes = sum(max(0, member.size) for member in members if member.isfile())
        if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
            raise ValueError("Model archive exceeds the 20 GiB expanded-size limit")
        for member in members:
            destination_path = (destination / member.name).resolve()
            try:
                destination_path.relative_to(destination_root)
            except ValueError:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe archive member type: {member.name}")
        tar.extractall(destination)


def resolve_model_source(tmp: Path) -> str:
    archive = BASE_MODEL_DIR / "model.tar.gz"
    if archive.is_file():
        extracted = tmp / "base-model"
        safe_extract_archive(archive, extracted)
        candidates = [path for path in extracted.iterdir() if path.is_dir()]
        return str(candidates[0] if len(candidates) == 1 else extracted)
    if BASE_MODEL_DIR.is_dir() and any(BASE_MODEL_DIR.iterdir()):
        return str(BASE_MODEL_DIR)
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
        safe_extract_archive(path, extracted)
        candidates = [candidate for candidate in extracted.rglob("adapter_config.json")]
        if candidates:
            return str(candidates[0].parent)
        directories = [candidate for candidate in extracted.iterdir() if candidate.is_dir()]
        return str(directories[0] if len(directories) == 1 else extracted)
    return str(path)


def create_model_and_tokenizer(model_source: str):
    trust_remote_code = hp_bool("trust_remote_code", True)
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        **model_revision_kwargs(model_source),
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **model_revision_kwargs(model_source),
        trust_remote_code=trust_remote_code,
        token=token,
        torch_dtype=dtype if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    return model, tokenizer


def lora_target_modules(default: str) -> list[str] | str:
    raw = hp("lora_target_modules", default) or default
    if raw == "all-linear":
        return raw
    return [item.strip() for item in raw.split(",") if item.strip()]


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=hp_int("lora_rank", 16),
        lora_alpha=hp_int("lora_alpha", 32),
        lora_dropout=hp_float("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_target_modules("all-linear"),
    )


def create_model_archive() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname="model")
    return archive_path


def dpo_config(output_dir: str) -> DPOConfig:
    max_prompt_length = hp("max_prompt_length")
    max_completion_length = hp("max_completion_length")
    config_values: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": hp_int("n_epochs", 3),
        "learning_rate": hp_float("learning_rate", 0.00001),
        "per_device_train_batch_size": hp_int("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": hp_int("gradient_accumulation_steps", 8),
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "beta": hp_float("dpo_beta", 0.1),
        "loss_type": hp("dpo_loss_type", "sigmoid"),
        "label_smoothing": hp_float("dpo_label_smoothing", 0.0),
        "reference_free": hp_bool("dpo_reference_free", False),
        "max_length": hp_int("max_seq_length", 2048),
        "max_prompt_length": int(max_prompt_length) if max_prompt_length else None,
        "max_completion_length": int(max_completion_length) if max_completion_length else None,
    }
    return DPOConfig(**supported_kwargs(
        DPOConfig,
        {key: value for key, value in config_values.items() if value is not None},
    ))


def run_training(rows: list[dict[str, str]], model_source: str) -> dict[str, Any]:
    with TemporaryDirectory() as tmp:
        parent_adapter = resolve_adapter_path(hp("parent_model_artifact"), Path(tmp))
        model, tokenizer = create_model_and_tokenizer(model_source)
        if parent_adapter:
            model = PeftModel.from_pretrained(model, parent_adapter, is_trainable=True)
        dataset = HFDataset.from_list(rows)

        args = dpo_config(tmp)
        trainer_values: dict[str, Any] = {
            "model": model,
            "ref_model": None,
            "args": args,
            "train_dataset": dataset,
            "processing_class": tokenizer,
            "tokenizer": tokenizer,
        }
        if not parent_adapter:
            trainer_values["peft_config"] = lora_config()
        trainer = DPOTrainer(**supported_kwargs(DPOTrainer, trainer_values))
        result = trainer.train()
        trainer.save_model(MODEL_DIR)

    tokenizer.save_pretrained(MODEL_DIR)
    return result.metrics


def main() -> None:
    started = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_preference_rows()
    with TemporaryDirectory(prefix="tt-local-base-model-") as base_tmp:
        model_source = resolve_model_source(Path(base_tmp))
        metrics = run_training(rows, model_source)

    archive_path = create_model_archive()
    output_metrics = {
        "training_method": "dpo",
        "preference_rows": len(rows),
        "model_source": str(BASE_MODEL_DIR) if BASE_MODEL_DIR.is_dir() and any(BASE_MODEL_DIR.iterdir()) else hp("base_model"),
        "base_model_revision": hp("base_model_revision"),
        "parent_model_artifact": hp("parent_model_artifact"),
        "dpo_beta": hp_float("dpo_beta", 0.1),
        "dpo_loss_type": hp("dpo_loss_type", "sigmoid"),
        "train_runtime": round(time.time() - started, 3),
        **{key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))},
        "model_archive": str(archive_path),
    }
    (MODEL_DIR / "training-metrics.json").write_text(json.dumps(output_metrics, indent=2))
    (OUTPUT_DIR / "training-metrics.json").write_text(json.dumps(output_metrics, indent=2))
    print(json.dumps(output_metrics, indent=2))


if __name__ == "__main__":
    main()
