---
name: fault-tolerant-api-runs
description: Make long API-based batch experiments survive flaky failures and resume cleanly. Covers per-item try/except isolation, atomic incremental checkpointing, resume-by-skipping-completed-items, only-persist-successes (so failures auto-retry on re-run), periodic + finally saves, and completion accounting in metadata.
---

# Fault-Tolerant Batch API Runs

API experiments over hundreds/thousands of items *will* hit rate limits,
timeouts, safety blocks, and transient 5xx. The design goal: a run can die at any
point and a plain re-run continues from where it stopped, retrying only what
failed. No babysitting.

## Core invariant: only successes are persisted

The trick that makes re-runs self-healing: **write only successful items to the
results JSON.** Failed items simply aren't there, so the next run sees them as
"not done" and retries them. Never write a placeholder/partial record for a
failure — that would mark it permanently complete.

```python
res = process_item(item)        # returns None on any failure
if res is not None:
    results.append(res)
else:
    failed_this_run += 1        # counted, but NOT written
```

## Resume by skipping already-completed items

On startup, load the most recent prior results file, index completed items by a
stable key, and partition the work:

```python
existing = {r["item_id"]: r for r in load_json_safe(prev_path).get("results", [])} if prev_path else {}
to_process, results = [], []
for item in benchmark:
    (results if item["id"] in existing else to_process).append(
        existing[item["id"]] if item["id"] in existing else item)
print(f"Resuming: {len(existing)}/{len(benchmark)} already done")
```

Pick the latest prior file by mtime, matching a per-model filename prefix
(`{prefix}{timestamp}.json`), and offer a `--new` flag to force a fresh run.

## Two layers of retry

1. **Inside the provider call** — transient HTTP/transport errors get
   exponential backoff + jitter, bounded by `max_retries` (see
   [[provider-api-utils]]).
2. **Across runs** — anything that exhausts inner retries returns `None`, is
   left unwritten, and is retried on the next invocation. Re-running until
   `missing == 0` is a valid, intended workflow.

Per-item isolation is mandatory — one bad item must never abort the batch:

```python
def process_item(item):
    try:
        return evaluate(item)
    except Exception as exc:
        log(f"[ERROR] {item['id']}: {exc}")
        return None
```

## Atomic, incremental, and finally-guarded saves

Checkpoint **during** the run so a crash loses at most a few items, and write
atomically (temp file + `os.replace`) so a crash mid-write never corrupts the
JSON:

```python
def atomic_json_save(data, path):
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f: json.dump(data, f, indent=2)
        os.replace(tmp, path)            # atomic rename
    except Exception:
        os.path.exists(tmp) and os.unlink(tmp); raise
```

Save on a cadence (every N completions), once at the end, and again in a
`finally` block so a `KeyboardInterrupt` still flushes partial progress:

```python
try:
    for i, item in enumerate(to_process, 1):
        results.append(process_item(item)) ...
        if i % 5 == 0: save(results_path, results)   # periodic
    save(results_path, results)                        # end
finally:
    try: save(results_path, results)                   # crash/interrupt safety
    except Exception: pass
```

With `ThreadPoolExecutor`, checkpoint from the `as_completed` loop on the main
thread (workers never write the file) to keep saves serialized.

## Completion accounting in metadata

Record enough in a `metadata` header to know, without reprocessing, whether a run
finished and what to retry:

```python
"metadata": {
    "total_items": total, "items_evaluated": len(results),
    "items_failed": total - len(results),
    "complete": (total - len(results)) == 0,
    "last_updated": datetime.now().isoformat(),
    "metrics_summary": {...},
}
```

End the run with an honest status line: `COMPLETE — all N (fail=0)` or
`done: 240/272 (32 missing). Re-run to retry.` This pairs directly with
[[json-checkpoint-records]] (the schema) and [[cross-model-comparison-plots]]
(which reads `complete`/`items_failed` to flag partial models).
