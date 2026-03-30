# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Colon-Bench Eval is a Python evaluation toolkit for the Colon-Bench colonoscopy video benchmark. It provides VQA evaluation, binary classification, detection + EdgeTAM segmentation pipelines, plotting scripts, and a Streamlit viewer. See `README.md` for full details.

### Package management

`uv` is the package manager. The lockfile is `uv.lock`. Core install: `uv sync`. Full install with viewer extras: `uv sync --extra viewer`. All extras: `uv sync --extra all`.

### Running scripts

All scripts should be run via `uv run` from the repo root, e.g.:
```
uv run python scripts/plot_vqa_accuracy.py --save plots/vqa_accuracy_prompted.pdf plots/vqa_accuracy_unprompted.pdf
```

### Streamlit viewer

Before launching Streamlit, ensure `~/.streamlit/credentials.toml` exists with `email = ""` under `[general]` and `~/.streamlit/config.toml` sets `headless = true` and `gatherUsageStats = false`. Otherwise Streamlit blocks on an interactive email prompt. Launch with:
```
uv run streamlit run viewer/visualize_benchmark.py -- --benchmark data/colon-bench --mode vqa_prompted
```

### EdgeTAM CUDA extension

On CPU-only environments, set `SAM2_BUILD_CUDA=0` before `uv sync` to skip building the CUDA extension. The extension is optional and its absence only affects post-processing in segmentation (does not break imports or other tasks).

### API keys

`HF_TOKEN` and `OPENROUTER_API_KEY` are required environment variables. `HF_TOKEN` is needed for all dataset access (videos are streamed from HF Hub). `OPENROUTER_API_KEY` is needed for API-based evaluations. Without them, the evaluation scripts and viewer will fail on video resolution.

### No automated test suite

This repo has no `pytest` or other automated test framework. Validation is done by running the plotting scripts (which exercise data loading and metrics computation) and the Streamlit viewer.

### Linting

No linting configuration (ruff, flake8, pylint, mypy, etc.) is present in the repository.
