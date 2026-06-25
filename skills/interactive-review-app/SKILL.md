---
name: interactive-review-app
description: Build a Streamlit app that runs a multi-stage processing pipeline on an uploaded file and lets a human review the results item-by-item with feedback (approve/reject/navigate). Covers session-state-driven navigation, per-stage subprocess execution with live progress, validation gating between stages, and packaging outputs for download.
---

# Interactive Review App with Human Feedback

A Streamlit pattern for "upload → run pipeline → human reviews each result"
workflows. Domain-neutral: works for any pipeline that produces a list of
reviewable items (clips, images, documents, predictions).

## Session-state-driven item navigation

Keep the reviewer's position and verdicts in `st.session_state` so reruns (every
Streamlit interaction triggers a full script rerun) don't lose state. Index by a
**stable per-item key**, not by list position, so feedback survives reordering:

```python
def render_review_ui(items):
    st.session_state.setdefault("review_feedback", {})   # key -> {"status": ...}
    st.session_state.setdefault("review_index", 0)

    idx = max(0, min(st.session_state.review_index, len(items) - 1))
    st.session_state.review_index = idx
    item = items[idx]
    key = stable_key(item)
    fb = st.session_state.review_feedback.get(key, {})

    st.success(f"Item {idx + 1}/{len(items)}")
    st.write(describe(item))
    st.video(media_path(item))          # or st.image / st.dataframe / st.json

    c1, c2 = st.columns(2)
    if c1.button("Previous", disabled=idx == 0):
        st.session_state.review_index = idx - 1; st.rerun()
    if c2.button("Next", disabled=idx >= len(items) - 1):
        st.session_state.review_index = idx + 1; st.rerun()
```

For approve/reject, write the verdict into `review_feedback[key]` then `st.rerun()`
so the new status renders immediately. Persist verdicts back to JSON
([[json-checkpoint-records]]) when the reviewer is done.

## Run pipeline stages as monitored subprocesses

Run each heavy stage as a subprocess (keeps the UI process light, isolates
crashes, and lets you stream a log). After each stage, **gate on validation**
before starting the next — abort early with a clear message instead of cascading
a bad stage forward:

```python
for stage in STAGES:
    with st.status(f"Running {stage.name}…", expanded=True) as status:
        run_subprocess(stage.cmd, log_path=stage.log)          # tees to a logfile
        if not validate_stage_output(stage.json, stage.log):   # syntax + error-scan
            status.update(label=f"{stage.name} failed", state="error")
            st.error(f"Stage {stage.name} produced no valid output. See log.")
            st.stop()
        status.update(label=f"{stage.name} ✓", state="complete")
```

Surface progress by **tailing the stage's logfile** and parsing a
`completed/total` pattern, with a simple ETA from elapsed time:

```python
def estimate_eta(start, done, total):
    if done == 0: return "…"
    rate = (time.time() - start) / done
    return human_time(rate * (total - done))
```

## Validation gating is what makes it trustworthy

Reuse the same validators the batch pipeline uses (existence, non-empty,
parseable, semantic-coverage, failure-sentinel scan — see
[[json-checkpoint-records]]). Also scan stage **logs** for provider error
strings (`RESOURCE_EXHAUSTED`, `quota`, `Traceback`, `Exception`) and treat a
log with errors as a failed stage even if the JSON looks fine.

## Package results for download

Let the reviewer take everything with them: zip the reviewed media + the
records + a generated report into an in-memory archive and offer it via
`st.download_button` — no server-side temp files to clean up:

```python
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in reviewed_media: z.write(p, arcname=p.name)
    z.writestr("records.json", json.dumps(records, indent=2))
    z.writestr("report.md", report_text)
st.download_button("Download results", buf.getvalue(), "results.zip")
```

## Conventions

- `st.set_page_config(layout="wide")`; hardcode non-user-facing pipeline params,
  expose only what the reviewer needs.
- Cache expensive derived state in `st.session_state` (e.g. `last_run_dir`) so
  navigating the review UI never re-triggers the pipeline.
- Keep the review loop read-only over the pipeline's JSON; write verdicts to a
  separate feedback structure so reruns are idempotent.
