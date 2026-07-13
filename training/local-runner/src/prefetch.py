import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

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


def configure_hugging_face_cache(cache_home: str | None) -> None:
    """Treat model_cache as HF_HOME and keep legacy overrides consistent."""
    if not cache_home:
        return
    # Keep the lexical absolute path supplied by the Node runner. Path.resolve
    # canonicalizes macOS /var to /private/var, making metadata disagree with
    # training even though both locations identify the same directory.
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


def hash_file(path: Path, algorithm: str, prefix: bytes = b"") -> str:
    digest = hashlib.new(algorithm)
    digest.update(prefix)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path) -> tuple[list[Path], int]:
    config = snapshot / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        raise ValueError(f"Cached snapshot is missing a non-empty config.json: {snapshot}")
    try:
        json.loads(config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cached snapshot has an invalid config.json: {snapshot}") from exc

    weight_files = sorted(snapshot.glob("*.safetensors")) + sorted(snapshot.glob("pytorch_model*.bin"))
    index_files = sorted(snapshot.glob("*.safetensors.index.json")) + sorted(snapshot.glob("pytorch_model*.bin.index.json"))
    for index in index_files:
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cached snapshot has an invalid weight index: {index}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Cached snapshot has an empty weight index: {index}")
        for name in set(weight_map.values()):
            if not isinstance(name, str) or not (snapshot / name).is_file():
                raise ValueError(f"Cached snapshot is missing an indexed weight shard: {name}")
    if not weight_files:
        raise ValueError(f"Cached snapshot contains no Transformers model weights: {snapshot}")
    for weight in weight_files:
        if weight.stat().st_size == 0:
            raise ValueError(f"Cached snapshot contains an empty model weight: {weight}")

    tokenizer_config = snapshot / "tokenizer_config.json"
    vocabulary = [
        snapshot / "tokenizer.json",
        snapshot / "tokenizer.model",
        snapshot / "sentencepiece.bpe.model",
        snapshot / "spiece.model",
        snapshot / "vocab.json",
        snapshot / "tokenizer.tiktoken",
    ]
    if not tokenizer_config.is_file() or not any(path.is_file() and path.stat().st_size > 0 for path in vocabulary):
        raise ValueError(f"Cached snapshot is missing tokenizer metadata or vocabulary files: {snapshot}")

    files = [path for path in snapshot.rglob("*") if path.is_file()]
    verified_blobs = 0
    for path in files:
        blob = path.resolve()
        expected = blob.name.lower()
        if len(expected) == 64 and all(character in "0123456789abcdef" for character in expected):
            actual = hash_file(blob, "sha256")
        elif len(expected) == 40 and all(character in "0123456789abcdef" for character in expected):
            actual = hash_file(blob, "sha1", f"blob {blob.stat().st_size}\0".encode())
        else:
            continue
        if actual != expected:
            raise ValueError(f"Cached Hugging Face blob checksum mismatch: {path}")
        verified_blobs += 1
    return files, verified_blobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch a TT Local Hugging Face base model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    base_model = str(payload["base_model"])
    cache_dir = payload.get("model_cache")
    configure_hugging_face_cache(cache_dir)

    # Import only after the cache environment is final. huggingface_hub reads
    # these variables while importing its constants module.
    from huggingface_hub import constants, snapshot_download

    action = "Verifying cached" if payload.get("local_files_only") else "Prefetching"
    print(f"{action} {base_model} in {constants.HF_HUB_CACHE}...", flush=True)
    snapshot_path = snapshot_download(
        repo_id=base_model,
        revision=payload.get("revision"),
        token=os.getenv("HF_TOKEN"),
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        local_files_only=bool(payload.get("local_files_only", False)),
    )
    snapshot = Path(snapshot_path)
    print(f"Verifying snapshot files under {snapshot}...", flush=True)
    snapshot_files, verified_blob_count = verify_snapshot(snapshot)

    write_json(args.output, {
        "ok": True,
        "base_model": base_model,
        "loader": payload.get("loader"),
        "model_cache": str(constants.HF_HOME),
        "hf_home": str(constants.HF_HOME),
        "hub_cache": str(constants.HF_HUB_CACHE),
        "snapshot_path": snapshot_path,
        "snapshot_revision": Path(snapshot_path).name,
        "file_count": len(snapshot_files),
        "size_bytes": sum(path.stat().st_size for path in snapshot_files),
        "verified_blob_count": verified_blob_count,
    })


if __name__ == "__main__":
    main()
