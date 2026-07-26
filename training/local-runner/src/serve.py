from __future__ import annotations

import hmac
import json
import math
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from evaluate import (
    configure_hugging_face_cache,
    import_runtime_dependencies,
    load_text_model,
    resolve_adapter_path,
    sampling_kwargs,
)


MODEL_ARTIFACT = os.environ.get("TT_MODEL_ARTIFACT")
BASE_MODEL = os.environ["TT_BASE_MODEL"]
BASE_MODEL_REVISION = os.environ.get("TT_BASE_MODEL_REVISION")
MODEL_NAME = os.environ.get("TT_MODEL_NAME", "tuned-tensor-local")
MODEL_LOADER = os.environ.get("TT_MODEL_LOADER", "causal_lm")
SYSTEM_PROMPT = os.environ.get("TT_SYSTEM_PROMPT", "").strip()
HOST = os.environ.get("TT_HOST", "127.0.0.1")
PORT = int(os.environ.get("TT_PORT", "8000"))
DEVICE_REQUEST = os.environ.get("TT_DEVICE", "cuda")
DEFAULT_MAX_TOKENS = int(os.environ.get("TT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("TT_TEMPERATURE", "0"))
DEFAULT_TOP_P = float(os.environ.get("TT_TOP_P", "1"))
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_PROMPT_CHARS = 100_000
MAX_PROMPT_TOKENS = 16_384
MAX_MESSAGES = 128
API_KEY = os.environ.get("TT_API_KEY", "")
MAX_CONCURRENT_REQUESTS = int(os.environ.get("TT_MAX_CONCURRENT_REQUESTS", "1"))


if MODEL_LOADER != "causal_lm":
    raise ValueError("The bundled model server is text-only and requires TT_MODEL_LOADER=causal_lm")
configure_hugging_face_cache(os.environ.get("HF_HOME"))
import_runtime_dependencies()
TEMP_DIR = TemporaryDirectory(prefix="tt-local-serve-")
ADAPTER_PATH = (
    resolve_adapter_path(MODEL_ARTIFACT, Path(TEMP_DIR.name))
    if MODEL_ARTIFACT
    else None
)
MODEL_PAYLOAD = {
    "base_model": BASE_MODEL,
    "base_model_revision": BASE_MODEL_REVISION,
    "device": DEVICE_REQUEST,
    "model_loader": MODEL_LOADER,
}
MODEL, TOKENIZER, DEVICE = load_text_model(MODEL_PAYLOAD, ADAPTER_PATH)
GENERATION_LOCK = threading.Lock()
REQUEST_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        raise ValueError("Message content must be text or an array of text parts.")
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            raise ValueError("The bundled Qwen model server accepts text content only.")
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Every text content part must contain a string text field.")
        parts.append(text)
    return "\n".join(parts)


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Request must include a non-empty messages array.")
    if len(raw_messages) > MAX_MESSAGES:
        raise ValueError(f"Request exceeds the {MAX_MESSAGES}-message limit.")

    system_parts = [SYSTEM_PROMPT] if SYSTEM_PROMPT else []
    conversation: list[dict[str, str]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict) or raw.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Each message must have a supported role and text content.")
        role = str(raw["role"])
        content = text_content(raw.get("content", ""))
        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
        else:
            conversation.append({"role": role, "content": content})

    if not conversation:
        raise ValueError("Request must contain at least one non-system message.")
    messages: list[dict[str, str]] = []
    if system_parts:
        # Qwen permits one leading system message. Merge the invariant owner
        # prompt and any client context instead of triggering an implicit
        # template fallback with duplicate system turns.
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.extend(conversation)
    if sum(len(message["content"]) for message in messages) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_CHARS}-character limit.")
    return messages


def generate_text(messages: list[dict[str, str]], generation: dict[str, Any]) -> tuple[str, int, int]:
    try:
        prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        raise ValueError(f"Qwen chat-template rendering failed: {exc}") from exc
    inputs = TOKENIZER(prompt, return_tensors="pt")
    if int(inputs["input_ids"].shape[-1]) > MAX_PROMPT_TOKENS:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_TOKENS}-token limit.")
    target_device = next(MODEL.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    with GENERATION_LOCK:
        import torch

        with torch.inference_mode():
            generated = MODEL.generate(
                **inputs,
                max_new_tokens=int(generation["max_new_tokens"]),
                pad_token_id=TOKENIZER.eos_token_id,
                **sampling_kwargs(generation),
            )
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    completion_tokens = generated[0][prompt_tokens:]
    content = TOKENIZER.decode(completion_tokens, skip_special_tokens=True).strip()
    return content, prompt_tokens, int(completion_tokens.shape[-1])


def bounded_number(value: Any, default: float, minimum: float, maximum: float) -> float:
    number = float(default if value is None else value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"Generation value must be between {minimum} and {maximum}.")
    return number


def bounded_integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    number = bounded_number(value, default, minimum, maximum)
    if not number.is_integer():
        raise ValueError(f"Generation value must be an integer between {minimum} and {maximum}.")
    return int(number)


class Handler(BaseHTTPRequestHandler):
    server_version = "TunedTensorLocalServer/0.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def authorized(self) -> bool:
        if not API_KEY:
            return True
        supplied = self.headers.get("authorization", "")
        return hmac.compare_digest(supplied, "Bearer " + API_KEY)

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if not self.authorized():
            self.send_json(401, {"error": {"message": "Unauthorized"}})
            return
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "model": MODEL_NAME, "device": DEVICE})
            return
        if self.path == "/v1/models":
            self.send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "tuned-tensor-local"}],
            })
            return
        self.send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.authorized():
            self.send_json(401, {"error": {"message": "Unauthorized"}})
            return
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self.send_json(404, {"error": {"message": "Not found"}})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body is empty or too large.")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object.")
            if body.get("stream"):
                self.send_json(400, {"error": {"message": "Streaming is not supported yet."}})
                return
            messages = normalize_messages(body.get("messages"))
            generation = {
                "max_new_tokens": bounded_integer(body.get("max_tokens"), DEFAULT_MAX_TOKENS, 1, 8192),
                "temperature": bounded_number(body.get("temperature"), DEFAULT_TEMPERATURE, 0, 5),
                "top_p": bounded_number(body.get("top_p"), DEFAULT_TOP_P, 0, 1),
            }
            if not REQUEST_SLOTS.acquire(blocking=False):
                self.send_json(429, {"error": {"message": "The local model is busy; retry shortly."}})
                return
            started = time.perf_counter()
            try:
                content, prompt_tokens, completion_tokens = generate_text(messages, generation)
            finally:
                REQUEST_SLOTS.release()
            latency_ms = round((time.perf_counter() - started) * 1000)
            self.send_json(200, {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "tt_local": {"latency_ms": latency_ms, "device": DEVICE},
            })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": {"message": str(exc)}})
        except Exception as exc:
            print(f"Model server error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self.send_json(500, {"error": {"message": "Internal model server error."}})

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def main() -> None:
    print(f"Serving {MODEL_NAME} on http://{HOST}:{PORT}", flush=True)
    print(f"OpenAI-compatible endpoint: http://{HOST}:{PORT}/v1/chat/completions", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    server.request_queue_size = max(2, MAX_CONCURRENT_REQUESTS * 2)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Model server stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
