from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


IGNORE_INDEX = -100


def _token_ids(value: Any, description: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{description} did not return a token-id sequence")
    token_ids = list(value)
    if not token_ids or any(not isinstance(token_id, int) for token_id in token_ids):
        raise ValueError(f"{description} returned empty or invalid token ids")
    return token_ids


def build_assistant_only_example(
    tokenizer: Any,
    messages: Any,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize one chat row while computing loss only on the final assistant turn.

    The entire assistant completion is reserved before the prompt is truncated.
    Prompt truncation therefore removes the oldest prompt tokens and can never
    silently discard the expected answer.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must contain prompt context and a final assistant turn")
    if any(not isinstance(message, dict) for message in messages):
        raise ValueError("every message must be an object")
    allowed_roles = {"system", "user", "assistant", "tool"}
    for message in messages:
        role = message.get("role")
        if role not in allowed_roles:
            raise ValueError(f"unsupported message role: {role!r}")
        if not isinstance(message.get("content"), str):
            raise ValueError("the bundled trainer accepts text-only message content")

    assistant = messages[-1]
    if assistant.get("role") != "assistant":
        raise ValueError("the final message must have role=assistant")
    content = assistant.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("the final assistant message must contain non-empty text")

    try:
        prompt_ids = _token_ids(
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            ),
            "prompt chat template",
        )
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
            ),
            "full chat template",
        )
    except Exception as exc:
        raise ValueError(f"Qwen chat-template rendering failed: {exc}") from exc

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Qwen chat-template output is not prefix-aligned; refusing to guess the assistant loss boundary"
        )
    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        raise ValueError("Qwen chat template produced no assistant completion tokens")
    if len(completion_ids) >= max_length:
        raise ValueError(
            f"assistant completion requires {len(completion_ids)} tokens, but max_length={max_length}; "
            "increase max_seq_length so the answer is not truncated"
        )

    prompt_budget = max_length - len(completion_ids)
    retained_prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = [*retained_prompt_ids, *completion_ids]
    labels = [IGNORE_INDEX] * len(retained_prompt_ids) + completion_ids.copy()
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
