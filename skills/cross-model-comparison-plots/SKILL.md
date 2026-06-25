---
name: cross-model-comparison-plots
description: Produce publication-quality cross-model comparison figures from per-model result JSONs. Covers auto-discovery of result files, a model→(display name, logo, color-family) metadata table, grouped bar charts with brand logos under each group, dual PDF+PNG export, --exclude-models, and auto-generated LaTeX tables.
---

# Cross-Model Comparison Plots

A reusable plotting layer that turns a directory of per-model result JSONs into
grouped bar charts, completion reports, and LaTeX tables — with near-zero effort
to add or remove a model. Domain-neutral: works for any "models × metrics"
comparison.

## Decouple plotting from evaluation via the filesystem

The plotter **discovers** results by globbing a directory and parsing model names
out of filenames — it never imports the eval code. Adding a model = dropping its
JSON in the folder; removing one = `--exclude-models slug`.

```python
def parse_filename(fname):                     # llm_eval_<model>_det_<ts>.json
    m = re.match(r"llm_eval_(.+?)_det_\d{8}_\d{6}\.json$", fname)
    return m.group(1) if m else None

def load_results(results_dir):
    data = {}
    for fname in sorted(os.listdir(results_dir)):
        slug = parse_filename(fname)
        if not slug: continue
        content = json.load(open(os.path.join(results_dir, fname)))
        entry = {**content["metrics"], **content["metadata"]}   # metrics + completion info
        prev = data.get(slug)                                    # keep the most-complete file
        if prev is None or entry["items_evaluated"] > prev["items_evaluated"]:
            data[slug] = entry
    return data
```

When multiple files exist for one model (re-runs), keep the one with the most
items evaluated. Read `complete`/`items_failed` from the metadata header
([[json-checkpoint-records]]) to flag partial models.

## One metadata table drives names, logos, and colors

Centralize all per-model presentation in a single dict so the plot, the report,
and the LaTeX table stay consistent. Map each model to a display label, a logo
file, and a color *family* (so a vendor's models share a hue):

```python
MODEL_META = {                       # slug -> (display name, logo file)
    "fast-vl":  ("Fast-VL\n8B",  "vendor_a.png"),
    "big-vl":   ("Big-VL\n235B", "vendor_a.png"),
    "other-vl": ("Other-VL",     "vendor_b.png"),
}
FAMILY_COLORS = {"vendor_a": "#4285F4", "vendor_b": "#E67E22"}
def family(slug):  return next((f for f in FAMILY_COLORS if f in slug), "other")
def bar_color(slug): return FAMILY_COLORS.get(family(slug), "#95A5A6")
```

Unknown slugs fall back to the slug as its own label and a neutral gray — so a
brand-new model still plots, just unstyled.

## Brand logos under each x-axis group

Place each model's logo beneath its bar group with `AnnotationBbox` in
axes-fraction coordinates, caching loaded logos so repeats are cheap:

```python
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def load_logo(path, h=36):
    img = Image.open(path).convert("RGBA"); a = img.width / img.height
    return np.array(img.resize((int(h * a), h), Image.LANCZOS))

for i, slug in enumerate(slugs):
    arr = logo_cache.setdefault(meta[slug][1], load_logo(logo_path))
    ab = AnnotationBbox(OffsetImage(arr, zoom=0.85), (i, 0),
                        xybox=(0, -56), xycoords=("data", "axes fraction"),
                        boxcoords="offset points", box_alignment=(0.5, 1.0), frameon=False)
    ax.add_artist(ab)
fig.subplots_adjust(bottom=0.20)     # leave room for logos
```

## Grouped bars: sort by primary metric, label values

Sort models by the primary metric (descending) so the figure reads as a ranking;
draw N metrics as offset bars per group; annotate each bar with its value:

```python
slugs = sorted(data, key=lambda s: data[s]["f1"], reverse=True)
x = np.arange(len(slugs)); bw = 0.75 / len(metrics)
for mi, m in enumerate(metrics):
    off = (mi - (len(metrics) - 1) / 2) * bw
    ax.bar(x + off, [data[s][m] * 100 for s in slugs], width=bw,
           color=METRIC_COLORS[m], label=METRIC_LABELS[m])
```

## Always export both PDF and PNG

Papers need vector PDF; slides/READMEs need PNG. Write both from one figure with
a shared basename:

```python
def savefig_pdf_png(fig, save_path, *, dpi=300):
    base = os.path.splitext(save_path)[0]
    fig.savefig(base + ".pdf", bbox_inches="tight", format="pdf")
    fig.savefig(base + ".png", bbox_inches="tight", format="png", dpi=dpi)
    return base + ".pdf", base + ".png"
```

## Free LaTeX tables + completion reports

From the same `data` dict, emit a `table*` with per-metric **bold-best**
highlighting and a console completion report (evaluated/failed/status per model).
The plot, the table, and the report are three views of one source — no manual
copying of numbers into the paper.

CLI surface to standardize across all plot scripts: `--results <dir>`,
`--logos <dir>`, `--save <path>`, `--exclude-models <slug...>`. This reads
directly from the per-model JSONs written by [[interchangeable-model-backends]].
