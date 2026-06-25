---
name: json-checkpoint-records
description: Use JSON files as the intermediate record between every stage of a multi-stage pipeline, so each step is independently inspectable, verifiable, resumable, and re-runnable without re-calling expensive upstream work. Covers the stage-output schema, raw-response preservation, validation gates, and metadata headers.
---

# JSON as the Pipeline's Intermediate Record

Treat JSON files as the contract between stages. Every stage reads the prior
stage's JSON and writes its own. This makes a long pipeline observable and
debuggable: you can open any stage's output, see exactly what happened, fix one
step, and re-run only downstream — without repeating expensive API calls.

## Why JSON-between-stages beats in-memory pipelines

- **Verify each step in isolation.** A human or a script can confirm stage K's
  output before stage K+1 consumes it.
- **Resume / re-run cheaply.** Re-parsing or re-metricking a saved record costs
  nothing; only stages whose JSON is missing/invalid need recompute.
- **Cheap audit trail.** The files *are* the experiment log — diff them across
  runs to see what changed.
- **Decouples teams/tools.** The plotting code, the review UI, and the dataset
  uploader all read the same JSON without importing the pipeline.

## Preserve raw model output alongside parsed results

The single most valuable habit: store the **raw model text** next to the parsed
structured result. A parser bug then costs a re-parse, not a re-run of the API:

```python
frame_results.append({
    "frame_index": idx,
    "frame_width": w, "frame_height": h,
    "predicted_boxes": parsed,      # structured, parser's interpretation
    "ground_truth_boxes": gt,
    "raw_response": raw_text,        # <- verbatim model output, re-parseable
    "prompt_tokens": p, "completion_tokens": c,
})
```

## Standard stage-output schema

Wrap results in a `metadata` header + `metrics` block + `results` array. The
header lets any consumer judge the file without scanning every record:

```python
{
  "metadata": {
    "task": "detection", "model": model, "params": {...},
    "total_items": N, "items_evaluated": K, "items_failed": N-K,
    "complete": (N-K) == 0,
    "last_updated": "2026-...T...",
    "metrics_summary": {"f1": ..., "ap50": ...},
  },
  "metrics": { ...full recomputed metrics... },
  "results": [ {per-item record with raw_response}, ... ],
}
```

Recompute aggregate `metrics` from `results` on every save, so the file is always
internally consistent even after a resume merged old + new items.

## Validation gates between stages

Before a stage trusts its input, validate it — existence, non-empty, parseable,
and *semantically* complete (not just syntactically valid JSON):

```python
def validate_json_syntax(p):
    return p.exists() and p.stat().st_size > 0 and _loads_ok(p)

def validate_stage_output(json_path, log_path):
    if not validate_json_syntax(json_path): return False
    if log_path.exists() and log_has_errors(log_path): return False   # scan log too
    return True
```

Two domain-neutral validation patterns worth copying:

- **Coverage check**: does the stage's last record correspond to the last input
  item? (e.g. last processed chunk == last input file) — catches truncated runs.
- **Status check**: scan records for failure sentinels (`status in
  {"api_error","error","not_found"}`) and reject the file if any are present, so
  a downstream stage never builds on poisoned records.

## Naming, timestamps, and atomic writes

- Filenames carry identity + time: `{task}_{model_safe}_{timestamp}.json`; pick
  the latest by mtime to resume.
- Sanitize model names for filesystems (`/`,`:`,space → `_`).
- Write atomically (temp + `os.replace`) and save incrementally — see
  [[fault-tolerant-api-runs]] for the full resilience loop.

This schema is the substrate everything else reads:
[[interchangeable-model-backends]] writes it identically per model,
[[cross-model-comparison-plots]] aggregates it, and
[[interactive-review-app]] renders it for human verification.
