---
name: distribution-plots
description: Plot dataset composition and multi-stage funnel distributions for papers — multi-label category bar charts with percentage annotations, and overlaid per-stage histograms where each stage is a subset of the previous. Covers keyword/regex multi-label classification of records, Counter aggregation, n-counts in legends, and dual PDF+PNG export.
---

# Distribution & Funnel Plots

Reusable matplotlib patterns for the "what's in my dataset / what survived each
stage" figures, distinct from model-vs-model bars (see
[[cross-model-comparison-plots]]). Domain-neutral.

## Multi-label category distribution

Records often belong to several categories at once. Classify each record by
regex/keyword matching into zero-or-more categories, count with `Counter`, and
plot percentages-of-total (which sum to >100% under multi-label — note that in
the caption):

```python
from collections import Counter

PATTERNS = {cat: [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in kws]
            for cat, kws in CATEGORY_KEYWORDS.items()}

def classify(record):
    text = build_search_text(record)               # concat the searchable fields
    return [cat for cat, pats in PATTERNS.items() if any(p.search(text) for p in pats)]

counts = Counter({c: 0 for c in PATTERNS})          # seed all categories at 0
labels_per_record = []
for r in records:
    cats = classify(r)
    labels_per_record.append(len(cats))
    for c in cats: counts[c] += 1

percentages = np.array([counts[c] for c in order]) / total * 100.0
```

Seeding the `Counter` with every category at 0 guarantees empty categories still
appear (no silently-dropped bars). Annotate each bar with both count and percent,
and print a text summary alongside the figure for the paper's prose.

## Multi-stage funnel: overlaid histograms of subsets

For a pipeline where each stage *filters* the previous one, load the surviving
records at each stage and overlay their histograms (e.g. of a duration/size
field). Put the surviving count `n=` in each legend label so the funnel is
readable:

```python
stages = {}                                          # label -> np.array of values
stages[f"Combined (n={len(a)})"]   = durations(a)
stages[f"Verified (n={len(b)})"]   = durations(b)    # b ⊂ a
stages[f"Confirmed (n={len(c)})"]  = durations(c)    # c ⊂ b

all_max = max(arr.max() for arr in stages.values() if len(arr))
bins = np.linspace(0, all_max, 30)
for label, arr in stages.items():
    if len(arr):
        ax.hist(arr, bins=bins, alpha=0.5, label=label)
ax.legend()
```

Shared bins across stages make the overlay comparable; descending alpha or a
fixed color ramp reads as "progressively filtered."

## Conventions shared with all plot scripts

- Load source data by globbing per-stage/per-record JSONs — never import the
  pipeline ([[json-checkpoint-records]]).
- A single `savefig_pdf_png(fig, path)` helper writes vector **PDF** + raster
  **PNG** from one figure with a shared basename (PDF for the paper, PNG for
  slides/READMEs).
- CLI: `--json <dir>` for input, `--save <path>` for output (display
  interactively if omitted).
- Print a numeric summary to stdout so the figure's numbers are quotable without
  re-reading the plot.
