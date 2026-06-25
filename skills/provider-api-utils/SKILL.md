---
name: provider-api-utils
description: Build a thin, normalized wrapper module around an LLM provider API (Gemini-style and OpenRouter/litellm-style). Covers the normalized completion return shape, env-driven rate limiting + bounded concurrency + exponential backoff with jitter, empty/blocked-response diagnostics, and a friendly-name→provider-slug alias table.
---

# Provider API Utility Modules

A reusable pattern for wrapping any LLM provider behind one small module so the
rest of your codebase never imports the raw SDK. Two reference shapes:
a **native-SDK** wrapper (Gemini-style) and a **unified-gateway** wrapper
(OpenRouter via litellm). Both expose the *same* return contract so callers are
provider-agnostic — that contract is what makes [[interchangeable-model-backends]]
possible.

## The normalized return contract

Every completion function returns `(completion, text)` where `completion` is a
`SimpleNamespace` with a `.usage` that always has `prompt_tokens` and
`completion_tokens`, regardless of provider. Callers read tokens and text the
same way everywhere:

```python
from types import SimpleNamespace
usage = SimpleNamespace(prompt_tokens=..., completion_tokens=...)
completion = SimpleNamespace(usage=usage, raw_response=response)
return completion, text
```

Normalize at the boundary; never leak the raw SDK response shape upward (keep it
on `.raw_response` for debugging only).

## Rate limiting + bounded concurrency + retry — all env-driven

Provider modules should be tunable without code edits. Read knobs from env with
sane defaults, and combine three independent guards:

```python
_RPM        = int(os.getenv("PROVIDER_REQUESTS_PER_MIN", "60"))
_MAX_CONC   = max(int(os.getenv("PROVIDER_MAX_CONCURRENT", "8")), 1)
_MAX_RETRY  = max(int(os.getenv("PROVIDER_MAX_RETRIES", "5")), 1)
_BACKOFF    = float(os.getenv("PROVIDER_BACKOFF_BASE", "1.0"))
_BACKOFF_CAP= float(os.getenv("PROVIDER_BACKOFF_CAP", "8.0"))

class _RateLimiter:           # thread-safe min-interval gate
    def __init__(self, rpm): self.interval = 60.0 / max(rpm, 1); self.lock = threading.Lock(); self.last = 0.0
    def wait(self):
        with self.lock:
            gap = self.interval - (time.time() - self.last)
            if gap > 0: time.sleep(gap)
            self.last = time.time()

_LIMITER = _RateLimiter(_RPM)
_GUARD   = threading.BoundedSemaphore(_MAX_CONC)
```

The call loop: rate-limit *outside* the semaphore, run *inside* it, and retry
transient failures with **exponential backoff + jitter** (jitter avoids
thundering-herd resync across threads):

```python
for attempt in range(_MAX_RETRY):
    _LIMITER.wait()
    try:
        with _GUARD:
            response = call_provider(...)
        break
    except Exception as exc:
        last_exc = exc
        backoff = min(_BACKOFF * (2 ** attempt), _BACKOFF_CAP) + random.uniform(0, 0.2)
        time.sleep(backoff)
else:
    raise last_exc   # for/else: ran out of retries
```

The `for/else` idiom is the cleanest retry-exhaustion signal in Python: `else`
runs only if the loop never `break`s.

## Diagnose empty responses (don't silently return "")

Safety filters and token limits produce *empty* completions that look like
success. Extract text robustly and surface *why* it was empty so logs are
actionable:

```python
text = getattr(response, "text", None)   # may raise on blocked content
if not text:
    # fall back to walking candidates -> content -> parts[].text
    # then collect a block_reason from prompt_feedback.block_reason,
    # candidate.finish_reason, and any non-negligible safety_ratings
    completion.block_reason = "; ".join(reasons)
```

A run that records `block_reason="prompt_blocked=SAFETY"` is debuggable; one that
records `""` is not.

## Friendly name → provider slug alias table

Let users pass short names on the CLI; resolve to the real provider/model slug
and carry capability flags (e.g. native video, per-model size caps) in one
table. Unknown names pass through as raw slugs so new models work without code
changes:

```python
ALIASES = {
    "fast-vl":  {"id": "vendor/fast-vl-instruct", "video": True},
    "big-vl":   {"id": "vendor/big-vl-235b",      "video": True, "max_video_bytes": 12*1024*1024},
}
def resolve(name):
    info = ALIASES.get(name.lower())
    if info: return f"gateway/{info['id']}", info.get("video", False)
    if name.startswith("gateway/"): return name, False
    return f"gateway/{name}", False     # bare slug -> auto-prefix
```

## Module conventions worth copying

- `load_dotenv()` at import; read keys from env (`*_API_KEY`), never hardcode.
- A `is_<provider>_model(name)` predicate so the dispatcher can route by name.
- An explicit `__all__` listing the public surface.
- Lazy-import heavy/optional SDKs inside functions, with a clear install hint on
  `ImportError`, so the module imports even when that provider isn't installed.

See [[mllm-video-input]] for the video content-part builders these wrappers
accept, and [[fault-tolerant-api-runs]] for the run-loop that sits above them.
