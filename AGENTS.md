# AGENTS.md

## Cursor Cloud specific instructions

`colon-bench-eval` is a single Python research toolkit (managed by `uv`) for reproducing the
Colon-Bench colonoscopy video benchmark. There is no database, server, or background daemon —
just CLI eval/plot scripts plus an optional Streamlit viewer. The `edgetam/` directory is a
vendored third-party model (Meta EdgeTAM), not a separate product. Standard install/run commands
live in `README.md`; this section only captures non-obvious caveats.

### Environment notes
- Dependencies are installed by the startup update script (`uv sync --extra all`). Use `uv run …`
  to execute anything inside the project venv. The plotting scripts need `matplotlib`, which only
  comes from the `viewer`/`all` extra — `uv sync` (core) alone is not enough to plot.
- **Git LFS is essential and non-obvious.** These files are LFS pointers and are useless until
  fetched: `edgetam/checkpoints/edgetam.pt` (EdgeTAM checkpoint), `data/colon-bench/*_seg_*.json`,
  `data/colon-bench/benchmark_detection.json`, `data/colon-bench/benchmark_segmentation.json`.
  Without `git lfs pull`, `scripts/plot_segmentation_metrics.py` reports "No segmentation data
  found" (JSON decode errors) and EdgeTAM segmentation fails for a missing checkpoint. The update
  script runs `git lfs pull`; if data files look like `version https://git-lfs...` pointers, run it
  again. Note `git lfs install` may warn about the Cursor-managed pre-push hook — that warning is
  harmless; `git lfs pull` still works on its own.
- After `git lfs pull`, the materialized LFS files may show as "modified" in `git status` because
  the smudge filter was not active at clone time. Do NOT stage/commit these data/checkpoint files.
- No GPU in this environment: `torch` runs on CPU and EdgeTAM's CUDA extension is skipped
  (`SAM2_BUILD_ALLOW_ERRORS=1`), which is fine for plots and the viewer. The local VLM path
  (`--extra local-vlm`, Qwen3-VL via `--local`) requires a GPU and is not runnable here; use the
  OpenRouter API path instead.

### Secrets
- `HF_TOKEN` and `OPENROUTER_API_KEY` are read from the environment (or a `.env` file). `HF_TOKEN`
  is required for the gated `ajhamdi/colon-bench` dataset (viewer video downloads, eval scripts);
  `OPENROUTER_API_KEY` is required for API-based VQA/classification/detection evals. Both are
  typically already present as env vars in this environment — check before declaring blocked.

### Running things
- Offline sanity check (no secrets, no network): the `scripts/plot_*.py` scripts run against the
  shipped result JSONs. They always save as `.pdf` regardless of the file extension you pass to
  `--save` (the extension is overwritten to `.pdf`).
- Streamlit viewer (a real long-running service):
  `uv run --extra viewer streamlit run viewer/visualize_benchmark.py --server.headless true --server.port 8501 -- --benchmark data/colon-bench --mode vqa_prompted`
  It serves on port 8501 and downloads colonoscopy clips from the gated HF dataset on demand the
  first time a sample is shown (needs `HF_TOKEN`); the first sample can take a few seconds.
- Full evals (`scripts/llm_evaluate_*.py`) hit the OpenRouter API and stream large videos from HF —
  they cost API credits and bandwidth, so prefer the offline plot scripts for smoke testing.

### Lint / tests / build
- There is no first-party lint, test, or CI configuration in the main project, and no Makefile or
  docker-compose. "Build" is just `uv sync` (EdgeTAM builds automatically as an editable install).
