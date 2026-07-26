from __future__ import annotations

from typing import Any


CERTIFIED_BASE_MODEL = "Qwen/Qwen3.5-2B"
CERTIFIED_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 6144,
    "vocab_size": 248320,
}


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def assert_certified_model_config(value: Any, label: str = "base-model config") -> None:
    model_type = _field(value, "model_type")
    architectures = _field(value, "architectures")
    if model_type == "qwen3_5":
        text_config = _field(value, "text_config")
        if (
            not isinstance(architectures, (list, tuple))
            or "Qwen3_5ForConditionalGeneration" not in architectures
            or text_config is None
        ):
            raise ValueError(f"{label} is not the certified {CERTIFIED_BASE_MODEL} architecture")
    elif model_type == "qwen3_5_text":
        # AutoModelForCausalLM exposes the selected text sub-config after it
        # dispatches the verified repository-level Qwen3.5 config.
        text_config = value
    else:
        raise ValueError(f"{label} is not the certified {CERTIFIED_BASE_MODEL} architecture")
    for name, expected in CERTIFIED_TEXT_CONFIG.items():
        actual = _field(text_config, name)
        if actual != expected:
            raise ValueError(
                f"{label} is not the certified {CERTIFIED_BASE_MODEL} architecture: "
                f"text_config.{name} must be {expected!r}, got {actual!r}"
            )
