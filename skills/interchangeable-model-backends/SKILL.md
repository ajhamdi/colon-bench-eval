---
name: interchangeable-model-backends
description: Architect one evaluation/inference pipeline that runs interchangeably against hosted API models (multiple providers) and local on-device models, selected by a single --model string. Covers name-based dispatch, the shared completion contract, capability-aware input selection, and graceful degradation between input modes.
---

# Interchangeable API + Local Model Backends

Goal: a single `--model NAME` flag selects *any* backend — Gemini, an OpenRouter
gateway model, or a locally-loaded weights model — and the rest of the pipeline
(prompting, parsing, metrics, checkpointing) is identical. This is what lets you
benchmark dozens of models with one code path.

## The dispatch layer

Route on the model name, not on scattered `if backend ==` checks. One predicate
per hosted provider plus a fallthrough to local:

```python
def call(model, prompt, *, frame_b64=None, video_ref=None, clients):
    if is_gemini_model(model):                 # name-based predicate
        return run_gemini_completion(clients.gemini, model, build_gemini_contents(...))
    if model in OPENROUTER_ALIASES or "/" in model:
        return run_openrouter_completion(model, prompt, frames_b64=[frame_b64], use_video=...)
    return clients.local_vlm.generate(prompt, ...)   # local weights fallthrough
```

The only requirement: **every branch returns the same `(completion, text)`
contract** (see [[provider-api-utils]]). Above this line, no code knows or cares
which backend ran.

## Capability-aware input selection with graceful degradation

Backends differ in what they accept (native video vs frames; remote URL vs
inline bytes vs local file). Resolve a model's capabilities, then pick the best
input it supports and **fall back down the ladder** when a preferred path fails:

```python
effective_use_video = use_video and model_supports_video
if effective_use_video and resolver:
    try:
        remote_url = resolver.resolve_video_url(name)   # zero-upload, preferred
    except Exception:
        remote_url = None                                # fall through
if effective_use_video and remote_url is None:
    if not local_file and resolver:
        local_file = resolver.download_video(name)       # ensure a local copy
    if too_large_to_inline(local_file):
        effective_use_video = False                      # degrade to frames
if not effective_use_video:
    frames = frame_cache.get_or_extract(local_file)      # universal fallback
```

Always have a **universal fallback** (frame extraction) so a new or quirky model
still produces *some* result instead of crashing the sweep. See
[[mllm-video-input]] for the strategy table.

## Local model wrapper: same surface, lazy + cached load

Wrap local weights behind the same call surface. Lazy-import torch/transformers
inside the wrapper (so the pipeline imports cleanly on API-only machines) and
load the model once:

```python
class LocalVideoVLM:
    def __init__(self, model_id, *, nframes=32, max_new_tokens=32):
        self.model = self.processor = None          # lazy
    def _ensure_loaded(self):
        if self.model is None:
            import torch; from transformers import AutoProcessor, ModelClass
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = ModelClass.from_pretrained(self.model_id, torch_dtype="auto", device_map="auto")
    def generate(self, prompt, video_path):
        self._ensure_loaded()
        ...
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0)), text
```

Local models often can't report token usage — return zeros rather than `None` so
downstream aggregation never special-cases them.

## Why this architecture pays off

- **One benchmark harness, N models.** Add a model by adding an alias-table row;
  no pipeline edits.
- **Provider-specific quirks stay isolated** in the provider utils
  ([[provider-api-utils]]) and in the parse layer (e.g. Gemini's 0–1000 box grid,
  see [[mllm-detection]]).
- **Cross-model comparison is trivial** because every run writes the same JSON
  schema ([[json-checkpoint-records]]), which the plotting layer reads uniformly
  ([[cross-model-comparison-plots]]).

## Orchestrating the sweep

Drive the matrix from a shell script with an editable `MODELS=(...)` array and a
`MAX_PARALLEL_MODELS` job-slot gate (`while jobs -rp | wc -l >= N; do sleep 1`).
Each model gets its own log file and its own resumable output JSON, so models run
concurrently and independently — one model failing never blocks the others.
Resilience details in [[fault-tolerant-api-runs]].
