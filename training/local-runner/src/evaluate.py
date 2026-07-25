from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from model_contract import (
    CERTIFIED_BASE_MODEL,
    assert_certified_model_config,
)

MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_EXPANDED_BYTES = 20 * 1024 * 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def configure_hugging_face_cache(cache_home: str | None) -> None:
    """Treat model_cache as HF_HOME and keep legacy overrides consistent."""
    if not cache_home:
        return
    home = Path(cache_home).expanduser().absolute()
    hub = home / "hub"
    os.environ.update({
        "HF_HOME": str(home),
        "HF_HUB_CACHE": str(hub),
        "HUGGINGFACE_HUB_CACHE": str(hub),
    })
    for deprecated in (
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
    ):
        os.environ.pop(deprecated, None)


def import_runtime_dependencies() -> None:
    """Import libraries only after Hugging Face cache settings are final."""
    global torch, PeftModel, AutoModelForCausalLM, AutoTokenizer

    import torch as torch_module
    from peft import PeftModel as peft_model
    from transformers import (
        AutoModelForCausalLM as auto_model_for_causal_lm,
        AutoTokenizer as auto_tokenizer,
    )

    torch = torch_module
    PeftModel = peft_model
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoTokenizer = auto_tokenizer


def strip_file_uri(value: str | None) -> str | None:
    if not value:
        return value
    if value.startswith("file://"):
        return value[7:]
    return value


def _extract_adapter_archive(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(f"Model archive exceeds {MAX_ARCHIVE_MEMBERS} members")
        expanded_bytes = sum(max(0, member.size) for member in members if member.isfile())
        if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
            raise ValueError("Model archive exceeds the 20 GiB expanded-size limit")
        for member in members:
            member_path = (destination / member.name).resolve()
            try:
                member_path.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe archive member: {member.name}") from exc
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe archive member type: {member.name}")
        archive.extractall(destination)


def resolve_adapter_path(value: str | None, tmp: Path) -> str | None:
    path_value = strip_file_uri(value)
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_file() and path.name.endswith(".tar.gz"):
        extracted = tmp / "adapter"
        _extract_adapter_archive(path, extracted)
        candidates = sorted(extracted.rglob("adapter_config.json"))
        if candidates:
            return str(candidates[0].parent)
        raise ValueError(f"Adapter archive contains no adapter_config.json: {path}")
    if not path.is_dir():
        raise ValueError(f"Adapter path is not a directory or .tar.gz archive: {path}")
    return str(path)


def format_prompt(
    tokenizer: Any,
    system: str,
    prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise ValueError(f"Qwen chat-template rendering failed: {exc}") from exc


def resolve_device(device: str) -> str:
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"device must be cuda or cpu; got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested, but torch.cuda.is_available() is false")
    return device


def _assert_certified_model_source(base_model: str) -> None:
    if not Path(base_model).exists() and base_model != CERTIFIED_BASE_MODEL:
        raise ValueError(
            f"The bundled evaluator currently certifies only {CERTIFIED_BASE_MODEL}; got {base_model!r}"
        )


def load_text_model(payload: dict[str, Any], adapter_path: str | None):
    base_model = str(payload["base_model"])
    _assert_certified_model_source(base_model)
    device = resolve_device(str(payload.get("device", "cuda")))
    token = os.getenv("HF_TOKEN")
    revision = payload.get("base_model_revision")
    source_kwargs: dict[str, Any] = {
        "trust_remote_code": False,
        "token": token,
    }
    if not Path(base_model).exists():
        source_kwargs["local_files_only"] = True
    if revision and not Path(base_model).exists():
        source_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(base_model, **source_kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{CERTIFIED_BASE_MODEL} tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model_kwargs: dict[str, Any] = dict(source_kwargs)
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if device == "cuda":
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    assert_certified_model_config(model.config)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def sampling_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    temperature = float(generation.get("temperature", 0))
    if temperature <= 0:
        return {"do_sample": False}
    return {
        "do_sample": True,
        "temperature": temperature,
        "top_p": float(generation.get("top_p", 1)),
    }


def generate_text_one(
    model: Any,
    tokenizer: Any,
    system: str,
    example: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    if example.get("input_assets"):
        raise ValueError("The bundled Qwen SFT evaluator is text-only and does not accept input_assets")
    prompt = str(example["input"])
    formatted = format_prompt(tokenizer, system, prompt)
    inputs = tokenizer(formatted, return_tensors="pt")
    target_device = next(model.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            pad_token_id=tokenizer.eos_token_id,
            **sampling_kwargs(generation),
        )
    latency_ms = max(0, round((time.perf_counter() - started) * 1000))
    input_length = int(inputs["input_ids"].shape[-1])
    output_tokens = generated[0][input_length:]
    actual = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    return {
        "id": str(example["id"]),
        "actual": actual,
        "latency_ms": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = load_json(Path(args.input))
    if payload.get("protocol_version") != 2:
        raise ValueError("Unsupported inference protocol; expected protocol_version 2")
    if payload.get("model_loader") not in (None, "causal_lm"):
        raise ValueError("The bundled evaluator is text-only and requires model_loader=causal_lm")
    configure_hugging_face_cache(payload.get("model_cache"))
    import_runtime_dependencies()

    with TemporaryDirectory(prefix="tt-local-evaluate-") as tmp_dir:
        adapter_path = resolve_adapter_path(payload.get("adapter_path"), Path(tmp_dir))
        generation = payload.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("generation must be a JSON object")
        model, tokenizer, _device = load_text_model(payload, adapter_path)
        results = [
            generate_text_one(
                model,
                tokenizer,
                str(payload.get("system", "")),
                example,
                generation,
            )
            for example in payload.get("examples", [])
        ]

    output = {
        "provider": "transformers",
        "model_id": payload.get("model_id"),
        "base_model": payload.get("base_model"),
        "adapter_path": payload.get("adapter_path"),
        "generation_config": generation,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
