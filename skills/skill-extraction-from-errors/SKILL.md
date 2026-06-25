---
name: skill-extraction-from-errors
description: Mine the shared mistakes across multiple models on a benchmark and distill them into a compact, reusable "skill" prompt that improves future model runs. Covers majority-wrong aggregation, severity weighting, payload compaction to fit a budget, LLM-based synthesis into fixed sections, and deterministic line/token budget enforcement.
---

# Extracting a Reusable Skill Prompt from Cross-Model Error Patterns

A self-improvement loop: run many models on a benchmark, find the questions that
*most models got wrong*, and have an LLM distill those shared failures into a
short prompt-appendable "skill" guide that future runs prepend. Domain-neutral —
the inputs are (question, correct answer, per-model predictions); the output is a
budgeted guidance string.

## 1. Aggregate by "majority wrong", carry severity

A mistake is signal only if it's *shared*. Collect questions where a majority of
evaluated models failed, and attach a **severity** so synthesis can weight
all-models-wrong cases above most-models-wrong cases:

```python
item = {
    "question": ex["question"],
    "correct_answer": ex["correct_answer"],
    "severity": f"{n_models_wrong}/{n_models_total} models wrong",
    "wrong_models": [...],
    "predicted_text_by_model": {model: pred, ...},   # what each model said
}
```

Keeping *which* models were right vs wrong (and what the right ones answered) is
the highest-value signal — it shows the synthesizer the discriminating cue.

## 2. Compact the payload to a budget before synthesis

Error dumps are huge; you can't send them all. Rank categories by frequency, take
the top-K, and distribute a fixed example budget across them (most-frequent first
until the budget is spent). Strip each example to only the fields the synthesizer
needs:

```python
remaining = MAX_EXAMPLES
for cat in sorted(categories, key=lambda c: -c["count"])[:MAX_CATEGORIES]:
    take = min(len(cat["examples"]), max(0, remaining))
    rows.append({"category": cat["category"], "count": cat["count"],
                 "examples": compact(cat["examples"][:take])})
    remaining -= take
    if remaining <= 0: break
```

## 3. Synthesize with a structure-locked prompt

Ask the LLM for **fixed headings** and hard constraints (max lines, target
tokens, no tables, short bullets). Fixed structure makes the output reusable and
comparable across regenerations:

```
Write a concise, high-signal skill guide derived from shared model mistakes.
Hard constraints: <= {max_lines} lines, target <= {max_tokens} tokens,
short bullets, no markdown tables.
Output structure (exact headings):
## Universal Anti-Error Rules
## <Domain> Cues by Category
## Common Confusion Traps and How to Resolve Them
## Fast Decision Checklist
Data:
{compact_payload_json}
```

Tell it explicitly how to read `severity` ("pay extra attention to cases where
ALL models failed; also learn from majority-wrong cases — subtle traps").

## 4. Enforce the budget deterministically (don't trust the model)

LLMs overshoot length limits. Post-process to *guarantee* the budget: truncate to
max lines, then drop trailing bullets one at a time until under the token
estimate, then hard-trim by characters as a last resort. Record before/after
stats:

```python
def estimate_tokens(t): return max(1, len(t) // 4)   # ~4 chars/token

while estimate_tokens(text) > max_tokens and len(lines) > MIN_LINES:
    # remove the last bullet line for deterministic compaction
    drop_last_bullet(lines)
    text = "\n".join(lines)
```

## 5. Emit three artifacts

- **Markdown** (`.md`) — human review.
- **Plain text** (`.txt`) — strip markdown to bare lines, ready to prepend to a
  prompt at inference.
- **Metadata** (`.json`) — model used, constraints, before/after line/token
  counts, source files, input overview. This is the audit trail
  ([[json-checkpoint-records]]).

## 6. Close the loop

Run the same benchmark again with the extracted skill prepended to the prompt and
compare scores (skill vs no-skill ablation). The error-pattern inputs come
straight from the per-model result JSONs produced by
[[interchangeable-model-backends]]; the synthesis call uses the wrappers in
[[provider-api-utils]]. The result is a portable, model-agnostic prompt
improvement mined entirely from observed failures.
