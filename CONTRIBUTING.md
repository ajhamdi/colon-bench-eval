# Contributing: Submit Your Model to the Leaderboard

We welcome community contributions! If you have evaluated a model on Colon-Bench and want it included in the [live leaderboard](https://abdullahamdi.com/colon-bench/#leaderboard), open a Pull Request to this repository with the following.

## What to Include in Your PR

1. **Result JSON files** — place them in the appropriate subdirectories using the existing naming convention:

   | Task | Directory | Filename pattern |
   |---|---|---|
   | VQA (prompted) | `data/colon-bench/eval_results/` | `llm_eval_<model-slug>_prompted_<YYYYMMDD_HHMMSS>.json` |
   | VQA (unprompted) | `data/colon-bench/eval_results/` | `llm_eval_<model-slug>_unprompted_<YYYYMMDD_HHMMSS>.json` |
   | Classification | `data/colon-bench/cls_results/` | `llm_eval_<model-slug>_cls_<YYYYMMDD_HHMMSS>.json` |
   | Detection / Segmentation | `data/colon-bench/det_results/` | `llm_eval_<model-slug>_det_<YYYYMMDD_HHMMSS>.json` / `…_seg_….json` |

   `<model-slug>` should be a lower-case, hyphen-separated identifier (e.g. `my-model-7b`). Look at existing files for reference.

2. **Updated [`leaderboard.csv`](leaderboard.csv)** — append a new row for your model with all available metrics. Use `—` for tasks you did not evaluate. The columns are:

   ```
   Model,VQA Prompted,VQA Unprompted,Cls. Accuracy,Cls. Precision,Cls. Recall,Cls. F1,Seg. IoU,Seg. Dice
   ```

3. **New model entry in `README.md`** — depending on how inference was run, add your model to one of the two tables in the Supported Models section. **Hyperlink the model name** so reviewers and future users can verify the model:

   - **OpenRouter models** — add a row to the *OpenRouter / API models* table. Link to the model's [OpenRouter](https://openrouter.ai/) page if available, otherwise to its paper, arXiv, HF card, or website:

     ```markdown
     | `my-model-7b` | [My Lab — My Model 7B](https://openrouter.ai/mylab/my-model-7b) | yes |
     ```

   - **Local models** — add a row to the *Local Models (GPU)* table. Link to the model's HF card, paper, arXiv, or website:

     ```markdown
     | `MyLab/My-Model-7B` | [7B](https://huggingface.co/MyLab/My-Model-7B) | Notes |
     ```

4. **Updated leaderboard table in `README.md`** — add a `<tr>` row for your model inside the `<tbody>` of the Leaderboard section (keep rows sorted by VQA Prompted ascending, or place segmentation-only / classification-only models in the existing ordering).

## PR Checklist

- [ ] Result JSONs follow the naming convention and are placed in the correct `data/colon-bench/` subdirectory.
- [ ] `leaderboard.csv` has been updated with a new row matching your results.
- [ ] Model added to the appropriate Supported Models table in the README with the model name hyperlinked to a paper, arXiv page, HF card, or website.
- [ ] The `<table>` in the Leaderboard section of the README has a corresponding new `<tr>`.
- [ ] Metrics were computed using the evaluation scripts in this repo (`scripts/llm_evaluate_*.py`) or an equivalent methodology on the official benchmark splits.
- [ ] PR description includes the model name, a link to the model (paper/HF/GitHub), and a brief note on how inference was run.

## Evaluation Tips

- Run the evaluation scripts shipped in `scripts/` against the official benchmark JSONs in `data/colon-bench/` to ensure your numbers are comparable.
- For segmentation, use the same [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) tracker with 3 box detections per model, as described in the paper.
- If you only evaluate on a subset of tasks, that is fine — fill in `—` for the remaining columns.

> **Note:** Once your PR is merged into `main`, the [live leaderboard](https://abdullahamdi.com/colon-bench/#leaderboard) on the project website will be updated to reflect the new results.
