# Skills

Portable, domain-neutral engineering skills extracted from this project's
codebase. Each skill targets **one reusable aspect** — coding patterns,
prompting techniques, and architecture — that you can drop into other projects.
Domain specifics of the original application are deliberately abstracted out;
these are about *how the code works*, not *what it analyzes*.

Each skill is a self-contained `SKILL.md` with YAML frontmatter (`name`,
`description`) and copy-pasteable, generic code patterns.

## MLLM techniques

- **[mllm-detection](mllm-detection/SKILL.md)** — object detection/localization
  with vision LLMs; the Gemini-vs-others coordinate-convention split and robust
  multi-fallback box parsing.
- **[mllm-video-input](mllm-video-input/SKILL.md)** — the four ways to feed video
  to MLLMs (File API upload, inline base64, remote URL, frame extraction), with
  duration-aware sampling and reference-counted upload caching.

## Provider / backend architecture

- **[provider-api-utils](provider-api-utils/SKILL.md)** — thin normalized wrapper
  modules around LLM providers: shared return contract, env-driven rate
  limiting + bounded concurrency + backoff, empty-response diagnostics, alias
  tables.
- **[interchangeable-model-backends](interchangeable-model-backends/SKILL.md)** —
  one pipeline that runs interchangeably against multiple hosted APIs *and* local
  models via a single `--model` flag, with capability-aware input fallback.

## Reliability & records

- **[fault-tolerant-api-runs](fault-tolerant-api-runs/SKILL.md)** — long batch API
  experiments that survive failures and resume cleanly; only-persist-successes so
  failures auto-retry on re-run.
- **[json-checkpoint-records](json-checkpoint-records/SKILL.md)** — JSON as the
  inspectable, verifiable, re-runnable contract between every pipeline stage.

## Interfaces & outputs

- **[interactive-review-app](interactive-review-app/SKILL.md)** — Streamlit app
  to run a pipeline on an upload and have a human review results item-by-item
  with feedback.
- **[cross-model-comparison-plots](cross-model-comparison-plots/SKILL.md)** —
  publication figures comparing many models: auto-discovery, logo/color metadata
  table, grouped bars, PDF+PNG, LaTeX tables, easy add/remove.
- **[distribution-plots](distribution-plots/SKILL.md)** — dataset-composition and
  multi-stage funnel distribution figures (multi-label category bars, overlaid
  per-stage histograms).

## Data & self-improvement

- **[hf-dataset-build-upload](hf-dataset-build-upload/SKILL.md)** — build typed
  multi-split Hugging Face datasets, push them with a card, and resolve asset URLs
  at read time without re-downloading.
- **[skill-extraction-from-errors](skill-extraction-from-errors/SKILL.md)** — mine
  shared cross-model mistakes into a compact, budgeted, reusable skill prompt that
  improves future runs.

---

The skills cross-reference each other with `[[skill-name]]` links. A typical
end-to-end flow: [[interchangeable-model-backends]] runs a model selected by name,
calling [[provider-api-utils]] wrappers with [[mllm-video-input]] content,
producing [[mllm-detection]] boxes; [[fault-tolerant-api-runs]] keeps the sweep
alive and writes [[json-checkpoint-records]]; those feed
[[cross-model-comparison-plots]] / [[distribution-plots]],
[[interactive-review-app]], [[hf-dataset-build-upload]], and
[[skill-extraction-from-errors]].
