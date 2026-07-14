from __future__ import annotations

import json
import base64
import hmac
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

# HF_HOME is supplied by the Node launcher before this module starts. Importing
# the model stack only after reading process configuration keeps every local
# workflow on the same Hugging Face cache contract.
from evaluate import (  # noqa: E402
    configure_hugging_face_cache,
    generate_multimodal_one,
    import_runtime_dependencies,
    load_multimodal_model,
    load_text_model,
    resolve_adapter_path,
    sampling_kwargs,
)


MODEL_ARTIFACT = os.environ["TT_MODEL_ARTIFACT"]
BASE_MODEL = os.environ["TT_BASE_MODEL"]
BASE_MODEL_REVISION = os.environ.get("TT_BASE_MODEL_REVISION")
MODEL_NAME = os.environ.get("TT_MODEL_NAME", "tuned-tensor-local")
MODEL_LOADER = os.environ.get("TT_MODEL_LOADER", "causal_lm")
SYSTEM_PROMPT = os.environ.get("TT_SYSTEM_PROMPT", "").strip()
HOST = os.environ.get("TT_HOST", "127.0.0.1")
PORT = int(os.environ.get("TT_PORT", "8000"))
DEVICE_REQUEST = os.environ.get("TT_DEVICE", "auto")
TRUST_REMOTE_CODE = os.environ.get("TT_TRUST_REMOTE_CODE", "true").lower() in {"1", "true", "yes"}
DEFAULT_MAX_TOKENS = int(os.environ.get("TT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("TT_TEMPERATURE", "0"))
DEFAULT_TOP_P = float(os.environ.get("TT_TOP_P", "1"))
CHAT_TEMPLATE_KWARGS = json.loads(os.environ.get("TT_CHAT_TEMPLATE_KWARGS", "{}"))
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_PROMPT_CHARS = 100_000
MAX_PROMPT_TOKENS = 16_384
MAX_MESSAGES = 128
MAX_IMAGE_BYTES = 5 * 1024 * 1024
API_KEY = os.environ.get("TT_API_KEY", "")
MAX_CONCURRENT_REQUESTS = int(os.environ.get("TT_MAX_CONCURRENT_REQUESTS", "1"))


configure_hugging_face_cache(os.environ.get("HF_HOME"))
import_runtime_dependencies()
TEMP_DIR = TemporaryDirectory(prefix="tt-local-serve-")
ADAPTER_PATH = resolve_adapter_path(MODEL_ARTIFACT, Path(TEMP_DIR.name))
MODEL_PAYLOAD = {
    "base_model": BASE_MODEL,
    "base_model_revision": BASE_MODEL_REVISION,
    "device": DEVICE_REQUEST,
    "trust_remote_code": TRUST_REMOTE_CODE,
    "model_loader": MODEL_LOADER,
}

if MODEL_LOADER == "image_text_to_text":
    MODEL, PROCESSOR, DEVICE = load_multimodal_model(MODEL_PAYLOAD, ADAPTER_PATH)
    TOKENIZER = getattr(PROCESSOR, "tokenizer", PROCESSOR)
else:
    MODEL, TOKENIZER, DEVICE = load_text_model(MODEL_PAYLOAD, ADAPTER_PATH)
    PROCESSOR = None

GENERATION_LOCK = threading.Lock()
REQUEST_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_REQUESTS)


def normalize_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("Request must include a non-empty messages array.")
    if len(raw_messages) > MAX_MESSAGES:
        raise ValueError(f"Request exceeds the {MAX_MESSAGES}-message limit.")
    messages: list[dict[str, Any]] = []
    for raw in raw_messages:
        if not isinstance(raw, dict) or raw.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Each message must have a supported role and content.")
        messages.append({"role": raw["role"], "content": raw.get("content", "")})
    # A spec passed by the server owner is invariant behavior. Client-supplied
    # system messages may add context but cannot replace it.
    if SYSTEM_PROMPT:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    if sum(len(text_content(message.get("content"))) for message in messages) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_CHARS}-character limit.")
    return messages


def text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
            continue
        else:
            parts.append(str(part))
    return "\n".join(part for part in parts if part)


def image_assets(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    assets: list[dict[str, str]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"image", "image_url"}:
                continue
            image = part.get("image_url")
            if isinstance(image, dict):
                image = image.get("url")
            image = image or part.get("image") or part.get("data_uri") or part.get("uri") or part.get("path")
            if isinstance(image, str) and image:
                if not image.startswith("data:image/") or "," not in image:
                    raise ValueError("Serving accepts image content only as data:image/... URIs.")
                encoded = image.split(",", 1)[1]
                try:
                    decoded_size = len(base64.b64decode(encoded, validate=True))
                except Exception as exc:
                    raise ValueError("Image data URI contains invalid base64.") from exc
                if decoded_size > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte decoded limit.")
                assets.append({"type": "image", "image": image})
    return assets


def generate_text(messages: list[dict[str, Any]], generation: dict[str, Any]) -> tuple[str, int, int]:
    normalized = [{"role": message["role"], "content": text_content(message.get("content"))} for message in messages]
    try:
        prompt = TOKENIZER.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
    except Exception:
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in normalized) + "\nassistant:"
    inputs = TOKENIZER(prompt, return_tensors="pt")
    if int(inputs["input_ids"].shape[-1]) > MAX_PROMPT_TOKENS:
        raise ValueError(f"Prompt exceeds the {MAX_PROMPT_TOKENS}-token limit.")
    target_device = next(MODEL.parameters()).device
    inputs = {key: value.to(target_device) for key, value in inputs.items()}
    with GENERATION_LOCK:
        import torch

        with torch.no_grad():
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


def generate_multimodal(messages: list[dict[str, Any]], generation: dict[str, Any]) -> tuple[str, int, int]:
    system_parts = [text_content(message.get("content")) for message in messages if message["role"] == "system"]
    conversation = "\n".join(
        f"{message['role']}: {text_content(message.get('content'))}"
        for message in messages
        if message["role"] != "system"
    )
    example = {
        "input": conversation,
        "output": "",
        "input_assets": image_assets(messages),
    }
    with GENERATION_LOCK:
        result = generate_multimodal_one(
            MODEL,
            PROCESSOR,
            "\n\n".join(part for part in system_parts if part),
            example,
            generation,
            Path.cwd(),
            CHAT_TEMPLATE_KWARGS,
        )
    # The evaluator does not currently expose token counts for multimodal
    # requests. Returning zero is preferable to inventing usage data.
    return str(result["actual"]), 0, 0


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
        expected = "Bearer " + API_KEY
        return hmac.compare_digest(supplied, expected)

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
                if MODEL_LOADER == "image_text_to_text":
                    content, prompt_tokens, completion_tokens = generate_multimodal(messages, generation)
                else:
                    if image_assets(messages):
                        raise ValueError("This text model does not accept image content parts.")
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
