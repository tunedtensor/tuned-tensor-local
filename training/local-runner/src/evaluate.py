from __future__ import annotations

import argparse
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

MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_EXPANDED_BYTES = 20 * 1024 * 1024 * 1024


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def configure_hugging_face_cache(cache_home: str | None) -> None:
    """Treat model_cache as HF_HOME and keep legacy overrides consistent."""
    if not cache_home:
        return
    # Preserve the lexical absolute path chosen by the Node runner rather than
    # canonicalizing platform symlinks such as macOS /var -> /private/var.
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
    """Import libraries that read Hugging Face cache settings at import time."""
    global torch, PeftModel, Image
    global AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    import torch as torch_module
    from peft import PeftModel as peft_model
    from PIL import Image as pil_image
    from transformers import (
        AutoModelForCausalLM as auto_model_for_causal_lm,
        AutoModelForImageTextToText as auto_model_for_image_text_to_text,
        AutoProcessor as auto_processor,
        AutoTokenizer as auto_tokenizer,
    )

    torch = torch_module
    PeftModel = peft_model
    Image = pil_image
    Image.MAX_IMAGE_PIXELS = 20_000_000
    AutoModelForCausalLM = auto_model_for_causal_lm
    AutoModelForImageTextToText = auto_model_for_image_text_to_text
    AutoProcessor = auto_processor
    AutoTokenizer = auto_tokenizer


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
        extracted = tmp / "adapter"
        extracted.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "r:gz") as tar:
            members = tar.getmembers()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError(f"Model archive exceeds {MAX_ARCHIVE_MEMBERS} members")
            expanded_bytes = sum(max(0, member.size) for member in members if member.isfile())
            if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
                raise ValueError("Model archive exceeds the 20 GiB expanded-size limit")
            for member in members:
                destination = (extracted / member.name).resolve()
                try:
                    destination.relative_to(extracted.resolve())
                except ValueError:
                    raise ValueError(f"Unsafe archive member: {member.name}")
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"Unsafe archive member type: {member.name}")
            tar.extractall(extracted)
        candidates = [candidate for candidate in extracted.rglob("adapter_config.json")]
        if candidates:
            return str(candidates[0].parent)
        directories = [candidate for candidate in extracted.iterdir() if candidate.is_dir()]
        return str(directories[0] if len(directories) == 1 else extracted)
    return str(path)


def fallback_prompt(system: str, prompt: str) -> str:
    return f"system: {system}\nuser: {prompt}\nassistant:"


def format_prompt(
    tokenizer: Any,
    system: str,
    prompt: str,
    chat_template_kwargs: dict[str, Any] | None = None,
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
            **(chat_template_kwargs or {}),
        )
    except Exception:
        return fallback_prompt(system, prompt)


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


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


def example_assets(example: dict[str, Any]) -> list[str]:
    assets = []
    for asset in example.get("input_assets") or []:
        if not isinstance(asset, dict):
            continue
        for key in ("image", "path", "uri", "data_uri"):
            value = asset.get(key)
            if isinstance(value, str) and value:
                assets.append(value)
                break
    return assets


def should_use_multimodal(payload: dict[str, Any]) -> bool:
    if payload.get("model_loader") == "image_text_to_text":
        return True
    return any(example_assets(example) for example in payload.get("examples", []))


def load_text_model(payload: dict[str, Any], adapter_path: str | None):
    base_model = payload["base_model"]
    trust_remote_code = bool(payload.get("trust_remote_code", True))
    device = resolve_device(str(payload.get("device", "auto")))
    token = os.getenv("HF_TOKEN")
    revision = payload.get("base_model_revision")
    source_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": token,
    }
    if revision and not Path(base_model).exists():
        source_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        **source_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = None
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    kwargs: dict[str, Any] = dict(source_kwargs)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def load_multimodal_model(payload: dict[str, Any], adapter_path: str | None):
    base_model = payload["base_model"]
    trust_remote_code = bool(payload.get("trust_remote_code", True))
    device = resolve_device(str(payload.get("device", "auto")))
    token = os.getenv("HF_TOKEN")
    revision = payload.get("base_model_revision")
    source_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "token": token,
    }
    if revision and not Path(base_model).exists():
        source_kwargs["revision"] = revision

    processor = AutoProcessor.from_pretrained(
        base_model,
        **source_kwargs,
    )
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    dtype = None
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = dict(source_kwargs)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device == "cuda":
        kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, processor, device


def sampling_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    """Greedy decoding must not pass sampling flags: transformers warns that
    temperature/top_p are ignored when do_sample=False."""
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
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = str(example["input"])
    expected = str(example["output"])
    formatted = format_prompt(tokenizer, system, prompt, chat_template_kwargs)
    inputs = tokenizer(formatted, return_tensors="pt")
    target_device = next(model.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.no_grad():
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
        "prompt": prompt,
        "expected": expected,
        "actual": actual,
        "latency_ms": latency_ms,
    }


def generate_multimodal_one(
    model: Any,
    processor: Any,
    system: str,
    example: dict[str, Any],
    generation: dict[str, Any],
    base_dir: Path,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = str(example["input"])
    expected = str(example["output"])
    image_values = example_assets(example)
    images = [load_image(value, base_dir) for value in image_values]
    user_content: list[dict[str, str]] = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    inputs = processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
    )
    target_device = next(model.parameters()).device
    inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(generation.get("max_new_tokens", 256)),
            pad_token_id=processor.tokenizer.eos_token_id,
            **sampling_kwargs(generation),
        )
    latency_ms = max(0, round((time.perf_counter() - started) * 1000))
    input_length = int(inputs["input_ids"].shape[-1])
    actual = processor.batch_decode(
        generated[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return {
        "prompt": prompt,
        "expected": expected,
        "actual": actual,
        "latency_ms": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = load_json(Path(args.input))
    configure_hugging_face_cache(payload.get("model_cache"))
    import_runtime_dependencies()

    with TemporaryDirectory() as tmp_dir:
        adapter_path = resolve_adapter_path(payload.get("adapter_path"), Path(tmp_dir))
        generation = payload.get("generation", {})
        chat_template_kwargs = payload.get("chat_template_kwargs") or None
        base_dir = Path(args.input).parent
        if should_use_multimodal(payload):
            model, processor, _device = load_multimodal_model(payload, adapter_path)
            results = [
                generate_multimodal_one(
                    model,
                    processor,
                    str(payload.get("system", "")),
                    example,
                    generation,
                    base_dir,
                    chat_template_kwargs,
                )
                for example in payload.get("examples", [])
            ]
        else:
            model, tokenizer, _device = load_text_model(payload, adapter_path)
            results = [
                generate_text_one(
                    model,
                    tokenizer,
                    str(payload.get("system", "")),
                    example,
                    generation,
                    chat_template_kwargs,
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
