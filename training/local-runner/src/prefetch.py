import argparse
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.safetensors.index.json",
    "*.tiktoken",
    "*.txt",
    "chat_template.jinja",
    "merges.txt",
    "pytorch_model*.bin",
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.*",
    "vocab.*",
]

IGNORE_PATTERNS = [
    "*.ckpt",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.ot",
    "optimizer.pt",
    "rng_state*.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
]


def write_json(path: str, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch a TT Local Hugging Face base model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    base_model = str(payload["base_model"])
    cache_dir = payload.get("model_cache")
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)

    print(f"Prefetching {base_model} into {cache_dir or 'default Hugging Face cache'}...", flush=True)
    snapshot_path = snapshot_download(
        repo_id=base_model,
        cache_dir=cache_dir,
        token=os.getenv("HF_TOKEN"),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    write_json(args.output, {
        "ok": True,
        "base_model": base_model,
        "loader": payload.get("loader"),
        "model_cache": cache_dir,
        "snapshot_path": snapshot_path,
    })


if __name__ == "__main__":
    main()
