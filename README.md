# ColonBench Eval

<p align="center">
  <a href="https://arxiv.org/abs/2603.25645"><img src="https://img.shields.io/badge/arXiv-2603.25645-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/ajhamdi/colon-bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow" alt="HF Dataset"></a>
  <a href="https://abdullahamdi.com/colon-bench"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://github.com/ajhamdi/colon-bench-eval"><img src="https://img.shields.io/github/stars/ajhamdi/colon-bench-eval?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/ajhamdi/colon-bench-eval/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-CC--BY--4.0-green.svg" alt="License"></a>
</p>

**ColonBench** is a comprehensive, human-verified, multi-task video benchmark for colonoscopy understanding. It spans **14 lesion categories** (including polyps, ulcers, and bleeding), over **300,000 bounding boxes**, **213,000 segmentation masks**, and **133,000 words** of clinical descriptions.

`colon-bench-eval` is a compact toolkit for reproducing the main ColonBench benchmark numbers. It ships benchmark JSONs, canonical result files, plotting scripts, an interactive Streamlit viewer, and runnable baselines for:

- **VQA** on `prompted` and `unprompted` splits
- **Binary lesion classification**
- **Detection** + **EdgeTAM-based segmentation**

> **Naming:** the public release uses `prompted` / `unprompted` (internal names were `easy` / `hard`).

## Dataset Summary

| Statistic | Value |
|---|---|
| Total videos | 1,597 |
| Total video size | ~81 GB |
| Bounding boxes | 300,000+ |
| Segmentation masks | 213,000+ |
| Lesion categories | 14 |
| VQA questions (`prompted`) | 1,485 |
| VQA questions (`unprompted`) | 2,740 |
| Classification samples | 790 |
| Segmentation samples | 264 |

## What's Included

- **`notebooks/explore_colon_bench.ipynb`**: interactive dataset explorer — play videos, view questions/labels/masks inline
- `scripts/llm_evaluate_vqa.py`: VQA baseline with `--local` Qwen3-VL support
- `scripts/llm_evaluate_cls.py`: classification baseline
- `scripts/llm_evaluate_det_seg.py`: detection + EdgeTAM segmentation
- `scripts/plot_*.py`: public plotting scripts for the shipped result JSONs
- `viewer/visualize_benchmark.py`: Streamlit benchmark browser
- `skills/colon-skill.md`: optional VQA skill/context file (works with both prompted and unprompted)
- `data/colon-bench/`: benchmark JSONs and canonical result JSONs
- `plots/`: pre-generated public plot assets
- `edgetam/`: vendored EdgeTAM source tree with bundled checkpoint

## Install

```bash
git clone https://github.com/ajhamdi/colon-bench-eval.git
cd colon-bench-eval

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The EdgeTAM checkpoint (`edgetam/checkpoints/edgetam.pt`, ~54 MB) is bundled in the repo — no extra download step is needed for segmentation.

## API Keys & Authentication

You need two keys:

| Key | Purpose | Get one at |
|---|---|---|
| `HF_TOKEN` | Access the gated ColonBench dataset | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENROUTER_API_KEY` | API-based evaluation (VQA, classification, detection) | [openrouter.ai/keys](https://openrouter.ai/keys) |

ColonBench is public but gated — request access on the [dataset page](https://huggingface.co/datasets/ajhamdi/colon-bench), then provide your token below.

**Option A — export directly in your shell** (simplest, great for cloud sandboxes):

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export HF_TOKEN=hf_...
```

**Option B — use a `.env` file** (convenient for repeated local use):

```bash
cp .env.example .env
# edit .env and fill in your keys
```

The `.env` file is loaded automatically if `python-dotenv` is installed.
Both approaches work — shell exports always take priority.

> `python-dotenv` is optional. If you prefer not to install it, just export
> the keys in your terminal before running any script.

**Option C — Hugging Face CLI login** (alternative for `HF_TOKEN`):

```bash
hf auth login
```

For API-based evaluations (VQA, classification, detection), video access works through Hugging Face Hub:

1. A Hub `resolve` URL is built with `hf_hub_url(..., repo_type="dataset")`
2. It is resolved to a temporary CDN link via `get_hf_file_metadata(...).location`
3. The resulting time-limited URL is passed to the model provider

Gated access stays on the Hugging Face side; the model provider receives a short-lived downloadable URL.

To verify access manually:

```bash
wget --header="Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/datasets/ajhamdi/colon-bench/resolve/main/videos/<video_id>.mp4"
```

## Quick Dataset Explore

```python
from datasets import load_dataset

vqa_prompted = load_dataset("ajhamdi/colon-bench", "vqa-prompted", split="test")
vqa_unprompted = load_dataset("ajhamdi/colon-bench", "vqa-unprompted", split="test")
cls = load_dataset("ajhamdi/colon-bench", "classification", split="test")
seg = load_dataset("ajhamdi/colon-bench", "segmentation", split="test")

print(vqa_prompted[0])
print(vqa_unprompted[0])
print(cls[0])
print(seg[0])
```

> **Interactive notebook:**
> [`notebooks/explore_colon_bench.ipynb`](notebooks/explore_colon_bench.ipynb)
> plays videos inline, prints questions with answer choices, displays
> classification labels, and renders segmentation mask grids — all
> loaded directly from the Hugging Face dataset. Open it to browse any
> sample by changing `CONFIG` and `IDX`.

## Run The Baselines

> Make sure `OPENROUTER_API_KEY` and `HF_TOKEN` are set before running
> (see [API Keys & Authentication](#api-keys--authentication) above).

### VQA via OpenRouter

Prompted split:

```bash
python scripts/llm_evaluate_vqa.py \
  --benchmark data/colon-bench \
  --mode prompted \
  --model qwen3-vl-8b \
  --parallel \
  --max-workers 4
```

Unprompted split:

```bash
python scripts/llm_evaluate_vqa.py \
  --benchmark data/colon-bench \
  --mode unprompted \
  --model gpt-4o
```

With the bundled skill prompt:

```bash
python scripts/llm_evaluate_vqa.py \
  --benchmark data/colon-bench \
  --mode prompted \
  --model qwen3-vl-8b \
  --skill-file skills/colon-skill.md
```

### VQA Locally With Qwen3-VL-8B

```bash
python scripts/llm_evaluate_vqa.py \
  --benchmark data/colon-bench \
  --mode prompted \
  --local \
  --local-model-id Qwen/Qwen3-VL-8B-Instruct
```

The local path downloads videos from the dataset on demand when a local `videos/` directory is not present.
Only `HF_TOKEN` is needed (no OpenRouter key required).

### Classification

```bash
python scripts/llm_evaluate_cls.py \
  --benchmark data/colon-bench \
  --model qwen3-vl-8b \
  --parallel \
  --max-workers 4
```

### Detection

```bash
python scripts/llm_evaluate_det_seg.py \
  --benchmark data/colon-bench \
  --model qwen3-vl-8b \
  --frames-count 3 \
  --parallel \
  --max-workers 4
```

### Segmentation

Run detection first, then segmentation:

```bash
python scripts/llm_evaluate_det_seg.py \
  --benchmark data/colon-bench \
  --model qwen3-vl-8b \
  --seg \
  --checkpoint edgetam/checkpoints/edgetam.pt \
  --model-cfg edgetam.yaml
```

The segmentation evaluator downloads GT masks from the Hugging Face dataset when `data/colon-bench/masks/` is not present locally.

## Canonical Results And Plots

This repo ships canonical public result JSONs in:

- `data/colon-bench/eval_results/`
- `data/colon-bench/cls_results/`
- `data/colon-bench/det_results/`

The pre-generated public figures are in `plots/`. To replot from the shipped JSONs:

```bash
python scripts/plot_vqa_accuracy.py \
  --save plots/vqa_accuracy_prompted.pdf plots/vqa_accuracy_unprompted.pdf

python scripts/plot_classification_metrics.py \
  --save plots/cls_accuracy.pdf plots/cls_prf1.pdf

python scripts/plot_detection_metrics.py \
  --save plots/detection_metrics.pdf

python scripts/plot_segmentation_metrics.py \
  --save plots/segmentation_metrics.pdf
```

## Streamlit Viewer

Run the bundled viewer from the repo root:

```bash
streamlit run viewer/visualize_benchmark.py -- --benchmark data/colon-bench --mode vqa_prompted
```

Notes:

- VQA, classification, and detection browsing work from the shipped benchmark JSONs plus a local `videos/` directory if you want playable clips.
- Segmentation browsing still expects local mask/frame assets; this viewer path has only been cleaned up for public release, not fully reworked to stream masks directly from Hugging Face.

## Citation

If you use ColonBench, please cite:

```bibtex
@misc{hamdi2026colonbench,
  title={Colon-Bench: An Agentic Workflow for Scalable Dense Lesion Annotation in Full-Procedure Colonoscopy Videos},
  author={Abdullah Hamdi and Changchun Yang and Xin Gao},
  year={2026},
  eprint={2603.25645},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2603.25645}
}
```

## Links

| | |
|---|---|
| Paper | [arXiv:2603.25645](https://arxiv.org/abs/2603.25645) |
| Project page | [abdullahamdi.com/colon-bench](https://abdullahamdi.com/colon-bench) |
| Dataset | [huggingface.co/datasets/ajhamdi/colon-bench](https://huggingface.co/datasets/ajhamdi/colon-bench) |
| Code | [github.com/ajhamdi/colon-bench-eval](https://github.com/ajhamdi/colon-bench-eval) |
| License | [CC-BY-4.0](LICENSE) |
