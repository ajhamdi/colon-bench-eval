from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import random
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2


OPENROUTER_MODEL_ALIASES: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {"id": "openai/gpt-4o", "video": False},
    "gpt-5.2": {"id": "openai/gpt-5.2", "video": False},
    "gpt-5.4": {"id": "openai/gpt-5.4", "video": False},
    "claude-opus-4.6": {"id": "anthropic/claude-opus-4.6", "video": False},
    "molmo-2-8b": {
        "id": "allenai/molmo-2-8b",
        "video": True,
        "max_video_bytes": 12 * 1024 * 1024,
    },
    "seed-1.6": {"id": "bytedance-seed/seed-1.6", "video": True},
    "seed-1.6-flash": {"id": "bytedance-seed/seed-1.6-flash", "video": True},
    "glm-4.6v": {"id": "z-ai/glm-4.6v", "video": True},
    "nemotron-nano-12b-v2-vl-free": {
        "id": "nvidia/nemotron-nano-12b-v2-vl:free",
        "video": True,
    },
    "qwen3-vl-8b": {"id": "qwen/qwen3-vl-8b-instruct", "video": True},
    "qwen3-vl-32b": {"id": "qwen/qwen3-vl-32b-instruct", "video": True},
    "qwen3-vl-235b": {"id": "qwen/qwen3-vl-235b-a22b-instruct", "video": True},
    "qwen3-vl-plus": {"id": "qwen/qwen-vl-plus", "video": True},
    "qwen-vl-max": {"id": "qwen/qwen-vl-max", "video": True},
    "qwen3.5-plus": {"id": "qwen/qwen3.5-plus-02-15", "video": True},
    "qwen3.5-397b-a17b": {"id": "qwen/qwen3.5-397b-a17b", "video": True},
    "minimax-m3": {"id": "minimax/minimax-m3", "video": True},
    "mimo-v2.5": {"id": "xiaomi/mimo-v2.5", "video": True},
    "gemini-2.5-flash": {"id": "google/gemini-2.5-flash", "video": True},
    "gemini-2.5-pro": {"id": "google/gemini-2.5-pro", "video": True},
    "gemini-2.5-flash-lite": {"id": "google/gemini-2.5-flash-lite", "video": True},
    "gemini-3-flash-preview": {"id": "google/gemini-3-flash-preview", "video": True},
    "gemini-3-pro-preview": {"id": "google/gemini-3-pro-preview", "video": True},
    "gemini-3.1-pro-preview": {"id": "google/gemini-3.1-pro-preview", "video": True},
    "gemini-3.1-flash-lite-preview": {
        "id": "google/gemini-3.1-flash-lite-preview",
        "video": True,
    },
    "nova-premier": {"id": "amazon/nova-premier-v1", "video": True},
    # Google Gemini models served through the native Gemini API (uses
    # GEMINI_API_KEY instead of OPENROUTER_API_KEY).
    "gemini-3.5-flash": {
        "id": "gemini-3.5-flash",
        "video": True,
        "provider": "gemini",
    },
}


DEFAULT_EXTRACTION_MODEL = "openai/gpt-4o-mini"
MAX_VIDEO_SIZE_BYTES = 20 * 1024 * 1024
FRAMES_PER_MINUTE = 8
MAX_FRAMES_CAP = 128


class _RateLimiter:
    """Simple thread-safe token bucket."""

    def __init__(self, rpm: int):
        self.interval = max(60.0 / max(rpm, 1), 0.0)
        self.lock = threading.Lock()
        self.last_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            wait_time = self.interval - (now - self.last_time)
            if wait_time > 0:
                time.sleep(wait_time)
            self.last_time = time.time()


_OR_RPM = int(os.getenv("OPENROUTER_REQUESTS_PER_MIN", "60"))
_OR_MAX_CONCURRENT = max(int(os.getenv("OPENROUTER_MAX_CONCURRENT", "8")), 1)
_OR_MAX_RETRIES = max(int(os.getenv("OPENROUTER_MAX_RETRIES", "5")), 1)
_OR_BACKOFF_BASE = float(os.getenv("OPENROUTER_BACKOFF_BASE", "1.0"))
_OR_BACKOFF_CAP = float(os.getenv("OPENROUTER_BACKOFF_CAP", "8.0"))

_RATE_LIMITER = _RateLimiter(_OR_RPM)
_CONCURRENCY_GUARD = threading.BoundedSemaphore(_OR_MAX_CONCURRENT)


def resolve_openrouter_model(model_name: str) -> Tuple[str, bool]:
    """Resolve a friendly model name into a liteLLM model slug + video capability.

    Most aliases route through OpenRouter (``openrouter/<id>``). Aliases that set
    ``"provider": "gemini"`` route through the native Gemini API
    (``gemini/<id>``), which liteLLM authenticates with ``GEMINI_API_KEY``.
    """
    lower = model_name.lower().strip()
    info = OPENROUTER_MODEL_ALIASES.get(lower)
    if info is not None:
        provider = info.get("provider", "openrouter")
        prefix = "gemini" if provider == "gemini" else "openrouter"
        return f"{prefix}/{info['id']}", bool(info.get("video", False))
    if lower.startswith(("openrouter/", "gemini/")):
        return model_name, False
    return f"openrouter/{model_name}", False


def list_openrouter_models() -> List[str]:
    return sorted(OPENROUTER_MODEL_ALIASES.keys())


def get_max_video_bytes(model_name: str) -> int:
    info = OPENROUTER_MODEL_ALIASES.get(model_name.lower().strip())
    if info and "max_video_bytes" in info:
        return int(info["max_video_bytes"])
    return MAX_VIDEO_SIZE_BYTES


def encode_video_base64(video_path: str) -> str:
    with open(video_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def extract_video_frames(
    video_path: str,
    num_frames: int = 16,
    jpeg_quality: int = 85,
    auto_scale: bool = True,
) -> List[str]:
    """Extract evenly spaced frames and return them as base64 JPEGs."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    effective_frames = num_frames
    if auto_scale:
        duration_sec = total_frames / fps
        duration_min = duration_sec / 60.0
        scaled = int(FRAMES_PER_MINUTE * duration_min)
        effective_frames = max(num_frames, min(scaled, MAX_FRAMES_CAP))

    if effective_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = [
            int(i * (total_frames - 1) / (effective_frames - 1))
            for i in range(effective_frames)
        ]

    frames_b64: List[str] = []
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = capture.read()
        if not ok:
            continue
        _, buffer = cv2.imencode(".jpg", frame, encode_params)
        frames_b64.append(base64.b64encode(buffer).decode("utf-8"))

    capture.release()
    return frames_b64


class VideoFrameCache:
    """Thread-safe cache for extracted video frames."""

    def __init__(self, num_frames: int = 16, jpeg_quality: int = 85):
        self.num_frames = num_frames
        self.jpeg_quality = jpeg_quality
        self._cache: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def get_or_extract(self, video_path: str) -> List[str]:
        key = hashlib.md5(video_path.encode("utf-8")).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        frames = extract_video_frames(
            video_path,
            num_frames=self.num_frames,
            jpeg_quality=self.jpeg_quality,
        )
        with self._lock:
            self._cache[key] = frames
        return frames

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def _build_video_content(video_path: str) -> Dict[str, Any]:
    mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
    return {
        "type": "video_url",
        "video_url": {"url": f"data:{mime_type};base64,{encode_video_base64(video_path)}"},
    }


def _build_video_url_content(video_url: str) -> Dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": video_url}}


def _build_frames_content(frames_b64: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"},
        }
        for frame_b64 in frames_b64
    ]


def _get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY to call OpenRouter models.")
    return key


def _get_gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY to call native Gemini API models.")
    return key


def _api_key_for_model(resolved_model: str) -> str:
    """Pick the right provider API key for an already-resolved liteLLM slug."""
    if resolved_model.startswith("gemini/"):
        return _get_gemini_api_key()
    return _get_api_key()


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _completion_text(response: Any) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "")
    return _extract_text_from_content(content).strip()


def run_openrouter_completion(
    model: str,
    prompt_text: str,
    *,
    video_path: Optional[str] = None,
    video_url: Optional[str] = None,
    frames_b64: Optional[List[str]] = None,
    use_video: bool = False,
    max_retries: int = _OR_MAX_RETRIES,
) -> Tuple[Any, str]:
    """
    Send a multimodal prompt through liteLLM/OpenRouter.

    Exactly one visual source should be provided:
    - `video_url` for a remotely accessible video
    - `video_path` for an inline base64 video
    - `frames_b64` for frame-based fallback
    """
    import litellm

    resolved_model, _ = resolve_openrouter_model(model)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    if use_video and video_url:
        content.append(_build_video_url_content(video_url))
    elif use_video and video_path:
        content.append(_build_video_content(video_path))
    elif frames_b64:
        content.extend(_build_frames_content(frames_b64))

    messages = [{"role": "user", "content": content}]

    last_exc: Optional[Exception] = None
    response = None
    for attempt in range(max_retries):
        _RATE_LIMITER.wait()
        try:
            with _CONCURRENCY_GUARD:
                response = litellm.completion(
                    model=resolved_model,
                    messages=messages,
                    api_key=_api_key_for_model(resolved_model),
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            backoff = min(_OR_BACKOFF_BASE * (2 ** attempt), _OR_BACKOFF_CAP)
            time.sleep(backoff + random.uniform(0.0, 0.2))
    else:
        raise last_exc  # type: ignore[misc]

    usage_data = getattr(response, "usage", None)
    usage = SimpleNamespace(
        prompt_tokens=int(getattr(usage_data, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_data, "completion_tokens", 0) or 0),
    )
    completion = SimpleNamespace(usage=usage, raw_response=response)
    return completion, _completion_text(response)


def run_openrouter_text_completion(
    model: str,
    prompt_text: str,
    max_retries: int = _OR_MAX_RETRIES,
) -> Tuple[Any, str]:
    """Send a text-only prompt through liteLLM/OpenRouter."""
    import litellm

    resolved_model, _ = resolve_openrouter_model(model)
    messages = [{"role": "user", "content": prompt_text}]

    last_exc: Optional[Exception] = None
    response = None
    for attempt in range(max_retries):
        _RATE_LIMITER.wait()
        try:
            with _CONCURRENCY_GUARD:
                response = litellm.completion(
                    model=resolved_model,
                    messages=messages,
                    api_key=_api_key_for_model(resolved_model),
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            backoff = min(_OR_BACKOFF_BASE * (2 ** attempt), _OR_BACKOFF_CAP)
            time.sleep(backoff + random.uniform(0.0, 0.2))
    else:
        raise last_exc  # type: ignore[misc]

    usage_data = getattr(response, "usage", None)
    usage = SimpleNamespace(
        prompt_tokens=int(getattr(usage_data, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage_data, "completion_tokens", 0) or 0),
    )
    completion = SimpleNamespace(usage=usage, raw_response=response)
    return completion, _completion_text(response)
