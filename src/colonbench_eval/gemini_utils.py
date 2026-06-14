"""Native Google Gemini API helpers for Colon-Bench evaluation.

Gemini models are routed through the official ``google-genai`` SDK (not
litellm) so that usage is billed to ``GEMINI_API_KEY`` rather than an
OpenRouter account. Videos are sent through the Gemini **File API** (upload +
poll for ``ACTIVE``) rather than inline base64, which the Gemini API does not
handle reliably for video. Image frames are sent inline (supported fine).

Only the native Gemini-API models in this repo (those whose resolved liteLLM
slug starts with ``gemini/``) use this module; all other models — including the
OpenRouter-hosted ``google/gemini-*`` aliases — continue to go through
``openrouter_utils``.
"""

from __future__ import annotations

import atexit
import base64
import hashlib
import os
import random
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None


GEMINI_MODELS = {
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
}


def is_gemini_model(model_name: str) -> bool:
    """Return True if *model_name* is a bare Gemini API model id."""
    if not model_name:
        return False
    return model_name in GEMINI_MODELS or model_name.startswith("gemini-")


def get_gemini_client() -> Any:
    """Create a Gemini API client, validating dependencies and credentials."""
    if genai is None:
        raise ImportError(
            "google-genai is not installed. Install it with "
            "`pip install google-genai` before selecting a Gemini model."
        )
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to call Gemini models.")
    return genai.Client(api_key=api_key)


def text_to_contents(text: str) -> List[Any]:
    """Wrap a plain prompt string into a Gemini contents payload."""
    return [text]


class _GeminiRateLimiter:
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


_GEMINI_RPM = int(os.getenv("GEMINI_REQUESTS_PER_MIN", "60"))
_GEMINI_MAX_CONCURRENT = max(int(os.getenv("GEMINI_MAX_CONCURRENT", "8")), 1)
_GEMINI_MAX_RETRIES = max(int(os.getenv("GEMINI_MAX_RETRIES", "5")), 1)
_GEMINI_BACKOFF_BASE = float(os.getenv("GEMINI_BACKOFF_BASE", "1.0"))
_GEMINI_BACKOFF_CAP = float(os.getenv("GEMINI_BACKOFF_CAP", "8.0"))
_GEMINI_RATE_LIMITER = _GeminiRateLimiter(_GEMINI_RPM)
_GEMINI_CONCURRENCY_GUARD = threading.BoundedSemaphore(_GEMINI_MAX_CONCURRENT)


def run_gemini_completion(
    client: Any,
    model: str,
    contents: List[Any],
) -> Tuple[Any, str]:
    """Execute a Gemini completion and normalize output/usage to an
    OpenAI-style ``SimpleNamespace`` used across the baselines."""
    last_exc: Optional[Exception] = None
    response = None
    for attempt in range(_GEMINI_MAX_RETRIES):
        _GEMINI_RATE_LIMITER.wait()
        try:
            with _GEMINI_CONCURRENCY_GUARD:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            backoff = min(_GEMINI_BACKOFF_BASE * (2 ** attempt), _GEMINI_BACKOFF_CAP)
            time.sleep(backoff + random.uniform(0.0, 0.2))
    else:
        raise last_exc  # type: ignore[misc]

    text: Optional[str] = None
    try:
        text = getattr(response, "text", None)
    except Exception:  # noqa: BLE001 - SDK raises if content blocked
        text = None

    if text is None:
        chunks: List[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)
        text = "\n".join(chunks)

    block_reason: Optional[str] = None
    if not text:
        reasons: List[str] = []
        prompt_fb = getattr(response, "prompt_feedback", None)
        if prompt_fb and getattr(prompt_fb, "block_reason", None):
            reasons.append(f"prompt_blocked={prompt_fb.block_reason}")
        for cand in getattr(response, "candidates", None) or []:
            fr = getattr(cand, "finish_reason", None)
            if fr:
                reasons.append(f"finish_reason={fr}")
        if reasons:
            block_reason = "; ".join(reasons)

    usage_meta = getattr(response, "usage_metadata", None)
    usage = SimpleNamespace(
        prompt_tokens=int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        if usage_meta
        else 0,
        completion_tokens=int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        if usage_meta
        else 0,
    )
    completion = SimpleNamespace(
        usage=usage, raw_response=response, block_reason=block_reason
    )
    return completion, (text or "").strip()


def upload_video_to_gemini(client: Any, video_path: str, verbose: bool = False) -> Any:
    """Upload a video via the File API and wait until it is ACTIVE."""
    uploaded = client.files.upload(file=video_path)
    max_wait = 300
    wait_interval = 5
    waited = 0
    while waited < max_wait:
        state = getattr(uploaded.state, "name", str(uploaded.state))
        if state == "ACTIVE":
            break
        if state == "FAILED":
            raise RuntimeError(f"Gemini video processing failed for {video_path}")
        time.sleep(wait_interval)
        waited += wait_interval
        uploaded = client.files.get(name=uploaded.name)
    state = getattr(uploaded.state, "name", str(uploaded.state))
    if state != "ACTIVE":
        raise RuntimeError(
            f"Gemini video processing timed out ({state}) for {video_path}"
        )
    if verbose:
        print(f"  [GEMINI] Uploaded {video_path} -> {uploaded.name} ({state})")
    return uploaded


def cleanup_all_gemini_files(client: Any, verbose: bool = False) -> int:
    """Delete every file currently stored under the Gemini File API."""
    try:
        files = list(client.files.list())
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [WARN] Could not list Gemini files: {exc}")
        return 0
    deleted = 0
    for f in files:
        try:
            client.files.delete(name=f.name)
            deleted += 1
        except Exception:  # noqa: BLE001
            pass
    if verbose and deleted:
        print(f"  [CLEANUP] Deleted {deleted} leftover Gemini file(s).")
    return deleted


class GeminiFileCache:
    """Thread-safe cache of uploaded Gemini videos keyed by local path.

    A given video is uploaded once and reused across questions. Files are
    deleted in bulk via :meth:`clear` (registered at interpreter exit).
    """

    def __init__(self, client: Any, verbose: bool = False):
        self.client = client
        self.verbose = verbose
        self._cache: Dict[str, Any] = {}
        self._upload_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def get_or_upload(self, video_path: str) -> Any:
        path_hash = hashlib.md5(video_path.encode()).hexdigest()
        with self._lock:
            if path_hash in self._cache:
                return self._cache[path_hash]
            upload_lock = self._upload_locks.setdefault(path_hash, threading.Lock())
        with upload_lock:
            with self._lock:
                if path_hash in self._cache:
                    return self._cache[path_hash]
            uploaded = upload_video_to_gemini(
                self.client, video_path, verbose=self.verbose
            )
            with self._lock:
                self._cache[path_hash] = uploaded
            return uploaded

    def clear(self) -> None:
        with self._lock:
            files = list(self._cache.values())
            self._cache.clear()
            self._upload_locks.clear()
        for file_ref in files:
            try:
                self.client.files.delete(name=file_ref.name)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# High-level entry points used by the evaluation scripts (via openrouter_utils)
# ---------------------------------------------------------------------------

_CLIENT: Any = None
_CLIENT_LOCK = threading.Lock()
_FILE_CACHE: Optional[GeminiFileCache] = None
_FILE_CACHE_LOCK = threading.Lock()


def _client() -> Any:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = get_gemini_client()
        return _CLIENT


def _file_cache() -> GeminiFileCache:
    global _FILE_CACHE
    with _FILE_CACHE_LOCK:
        if _FILE_CACHE is None:
            _FILE_CACHE = GeminiFileCache(_client())
            atexit.register(_FILE_CACHE.clear)
        return _FILE_CACHE


def _image_part(frame_b64: str) -> Any:
    data = base64.b64decode(frame_b64)
    try:
        return genai_types.Part.from_bytes(data=data, mime_type="image/jpeg")
    except Exception:  # noqa: BLE001
        return genai_types.Part(
            inline_data=genai_types.Blob(data=data, mime_type="image/jpeg")
        )


def run_gemini_media_completion(
    model: str,
    prompt_text: str,
    *,
    video_path: Optional[str] = None,
    video_url: Optional[str] = None,
    frames_b64: Optional[List[str]] = None,
    use_video: bool = False,
    max_retries: int = _GEMINI_MAX_RETRIES,
) -> Tuple[Any, str]:
    """Run a Gemini completion with video (File API) or frame inputs.

    Mirrors the signature of ``openrouter_utils.run_openrouter_completion`` so
    the evaluation scripts can route Gemini calls transparently.
    """
    _ = max_retries  # retries handled inside run_gemini_completion
    client = _client()

    if use_video and video_path:
        uploaded = _file_cache().get_or_upload(video_path)
        contents: List[Any] = [uploaded, prompt_text]
    elif frames_b64:
        contents = [prompt_text] + [_image_part(b) for b in frames_b64]
    else:
        contents = text_to_contents(prompt_text)

    return run_gemini_completion(client, model, contents)


def run_gemini_text_completion(
    model: str,
    prompt_text: str,
    max_retries: int = _GEMINI_MAX_RETRIES,
) -> Tuple[Any, str]:
    """Text-only Gemini completion (used for answer extraction fallback)."""
    _ = max_retries
    return run_gemini_completion(_client(), model, text_to_contents(prompt_text))


__all__ = [
    "GEMINI_MODELS",
    "is_gemini_model",
    "get_gemini_client",
    "run_gemini_completion",
    "run_gemini_media_completion",
    "run_gemini_text_completion",
    "upload_video_to_gemini",
    "cleanup_all_gemini_files",
    "GeminiFileCache",
]
