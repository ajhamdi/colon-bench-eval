"""
plot_detection_metrics.py

Plots grouped bar charts of detection metrics (F1, AP50, mAP50-95) for all
evaluated models on the colon-bench detection benchmark.

Metrics are read directly from result JSON files produced by
baselines/llm_evaluate_det_seg.py.

Model logos from ablations/logos/ are placed under each group.

Usage:
    python ablations/plot_detection_metrics.py
    python ablations/plot_detection_metrics.py --results data/colon-bench/det_results
    python ablations/plot_detection_metrics.py --save ablations/detection_metrics.pdf
"""

import os
import sys
import json
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ─── Defaults ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "colon-bench", "det_results")
DEFAULT_LOGOS_DIR = os.path.join(SCRIPT_DIR, "logos")

TOTAL_VIDEOS = 272

# ─── Model display names and logo mapping ────────────────────────────────────
MODEL_META = {
    "gemini-2.5-flash-lite":          ("Gemini 2.5\nFlash Lite",     "gemini.png"),
    "gemini-3-flash-preview":         ("Gemini 3\nFlash",            "gemini.png"),
    "gemini-3-pro-preview":           ("Gemini 3\nPro",              "gemini.png"),
    "gemini-3.1-pro-preview":         ("Gemini 3.1\nPro",            "gemini.png"),
    "gemini-3.1-flash-lite-preview":  ("Gemini 3.1\nFlash Lite",     "gemini.png"),
    "glm-4.6v":                       ("GLM-4.6V",                   "glm.png"),
    "molmo-2-8b":                     ("Molmo\n2-8B",                "molmo.png"),
    "nemotron-nano-12b-v2-vl-free":   ("Nemotron\nNano 12B",        "nemotron.png"),
    "qwen3-vl-8b":                    ("Qwen3-VL\n8B",              "qwen.png"),
    "qwen3-vl-32b":                   ("Qwen3-VL\n32B",             "qwen.png"),
    "qwen3-vl-235b":                  ("Qwen3-VL\n235B",            "qwen.png"),
    "qwen3-vl-plus":                  ("Qwen3-VL\nPlus",            "qwen.png"),
    "qwen-vl-max":                    ("Qwen-VL\nMax",              "qwen.png"),
    "qwen3.5-397b-a17b":             ("Qwen3.5\n397B",             "qwen.png"),
    "qwen3.5-plus":                   ("Qwen3.5\nPlus",             "qwen.png"),
    "seed-1.6":                       ("Seed 1.6",                   "seed.png"),
    "seed-1.6-flash":                 ("Seed 1.6\nFlash",            "seed.png"),
    "gpt-4o":                         ("GPT-4o",                     "gpt.png"),
    "gpt-5.2":                        ("GPT-5.2",                    "gpt.png"),
    "gpt-5.4":                        ("GPT-5.4",                    "gpt.png"),
    "nova-premier":                   ("Nova\nPremier",              "nova.png"),
    "claude-opus-4.6":                ("Claude\nOpus 4.6",           "opus.png"),
}

# ─── Colour palette ──────────────────────────────────────────────────────────
FAMILY_COLORS = {
    "gemini":   "#4285F4",
    "seed":     "#34A853",
    "glm":      "#8E44AD",
    "qwen":     "#E67E22",
    "molmo":    "#E74C3C",
    "nemotron": "#76B900",
    "gpt":      "#10A37F",
    "nova":     "#FF9900",
    "claude":   "#D97706",
}


def _family(slug: str) -> str:
    for fam in FAMILY_COLORS:
        if fam in slug:
            return fam
    return "other"


def get_bar_color(slug: str) -> str:
    return FAMILY_COLORS.get(_family(slug), "#95A5A6")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_det_filename(filename: str):
    """
    Parse a detection results filename like
        llm_eval_gemini-3-flash-preview_det_20260215_184125.json
    Returns model_slug or None on failure.
    """
    m = re.match(r"llm_eval_(.+?)_det_\d{8}_\d{6}\.json$", filename)
    if not m:
        return None
    return m.group(1)


def load_results(results_dir: str) -> dict:
    """
    Load all detection eval JSONs.  Metrics are read from the full
    ``metrics`` block (recomputed by the eval script over all results).
    Completion info (evaluated, failed, complete) is read from the
    JSON metadata header.

    Returns dict: { model_slug: { 'f1': float, 'ap50': float,
                    'map50_95': float, 'mean_iou': float,
                    'precision': float, 'recall': float,
                    'videos_evaluated': int, 'videos_failed': int,
                    'total_videos': int, 'complete': bool } }
    """
    data = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        model_slug = parse_det_filename(fname)
        if model_slug is None:
            continue

        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, "r") as f:
                content = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] Skipping {fname}: {e}")
            continue

        metrics = content.get("metrics", {})
        meta = content.get("metadata", {})

        entry = {
            "f1": metrics.get("f1", 0.0),
            "ap50": metrics.get("ap50", 0.0),
            "map50_95": metrics.get("map50_95", 0.0),
            "mean_iou": metrics.get("mean_iou", 0.0),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            # Completion info from metadata header
            "videos_evaluated": meta.get("videos_evaluated", 0),
            "videos_failed": meta.get("videos_failed", 0),
            "total_videos": meta.get("total_videos", TOTAL_VIDEOS),
            "complete": meta.get("complete", False),
        }

        prev = data.get(model_slug)
        if prev is None or entry["videos_evaluated"] > prev["videos_evaluated"]:
            data[model_slug] = entry

    return data


def load_logo(logo_path: str, target_height: int = 40) -> np.ndarray | None:
    if not os.path.isfile(logo_path):
        return None
    try:
        img = Image.open(logo_path).convert("RGBA")
        aspect = img.width / img.height
        new_w = int(target_height * aspect)
        img = img.resize((new_w, target_height), Image.LANCZOS)
        return np.array(img)
    except Exception:
        return None


# ─── Plotting ─────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "f1":        "F1 Score",
    "ap50":      "AP@50",
    "map50_95":  "mAP@50-95",
}

METRIC_COLORS = {
    "f1":        "#2196F3",
    "ap50":      "#FF9800",
    "map50_95":  "#4CAF50",
}


def plot_detection(
    data: dict,
    logos_dir: str,
    metrics_to_plot: list[str] = ("f1", "ap50", "map50_95"),
    save_path: str | None = None,
) -> None:
    if not data:
        print("  [WARN] No detection data found, skipping.")
        return

    # Sort models by F1 (primary metric), descending
    slugs = sorted(data.keys(), key=lambda s: data[s]["f1"], reverse=True)
    n = len(slugs)
    n_metrics = len(metrics_to_plot)

    fig_width = max(14, n * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 7.5))

    x = np.arange(n)
    total_bar_width = 0.75
    bar_width = total_bar_width / n_metrics

    for mi, metric in enumerate(metrics_to_plot):
        offset = (mi - (n_metrics - 1) / 2) * bar_width
        values = [data[s][metric] * 100 for s in slugs]
        bars = ax.bar(
            x + offset, values, width=bar_width,
            color=METRIC_COLORS[metric], edgecolor="white", linewidth=0.6,
            alpha=0.88, zorder=3, label=METRIC_LABELS[metric],
        )
        for bar, val in zip(bars, values):
            if val > 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#333333", rotation=0,
                )

    # Model names + logos on x-axis
    ax.set_xticks(x)
    display_names = []
    for s in slugs:
        meta = MODEL_META.get(s)
        display_names.append(meta[0] if meta else s)
    ax.set_xticklabels(display_names, fontsize=15, ha="center", linespacing=1.1)

    logo_cache: dict[str, np.ndarray | None] = {}
    for i, slug in enumerate(slugs):
        meta = MODEL_META.get(slug)
        if meta is None:
            continue
        logo_file = meta[1]
        if logo_file not in logo_cache:
            logo_cache[logo_file] = load_logo(
                os.path.join(logos_dir, logo_file), target_height=36
            )
        logo_arr = logo_cache[logo_file]
        if logo_arr is None:
            continue
        imagebox = OffsetImage(logo_arr, zoom=0.85)
        ab = AnnotationBbox(
            imagebox, (i, 0),
            xybox=(0, -56),
            xycoords=("data", "axes fraction"),
            boxcoords="offset points",
            box_alignment=(0.5, 1.0),
            frameon=False, pad=0,
        )
        ax.add_artist(ab)

    # Axes formatting
    ax.set_ylabel("Score (%)", fontsize=18)
    max_val = max(
        data[s][m] * 100 for s in slugs for m in metrics_to_plot
    )
    ax.set_ylim(0, min(max_val + 12, 105))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    ax.set_title(
        "Colon-Bench Detection Benchmark",
        fontsize=20, fontweight="bold", pad=16,
    )

    ax.legend(loc="upper right", fontsize=15, framealpha=0.9, ncol=n_metrics)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(bottom=0.20)

    if save_path:
        save_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(save_path, bbox_inches="tight", format="pdf")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─── Completion report ────────────────────────────────────────────────────────

def print_completion_report(data: dict) -> None:
    """Print a per-model completion report showing evaluation success rates."""
    slugs = sorted(data.keys(), key=lambda s: data[s]["f1"], reverse=True)

    print(f"\n{'─'*90}")
    print(f"  COMPLETION REPORT — Detection Benchmark ({TOTAL_VIDEOS} videos)")
    print(f"{'─'*90}")
    print(f"  {'Model':<30s}  {'Evaluated':>12s}  {'Failed':>8s}  "
          f"{'Success':>8s}  {'Status'}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*12}")

    for slug in slugs:
        d = data[slug]
        meta = MODEL_META.get(slug)
        name = meta[0].replace("\n", " ") if meta else slug
        evald = d["videos_evaluated"]
        total = d["total_videos"]
        failed = d["videos_failed"]
        pct = evald / total * 100 if total > 0 else 0.0
        status = "COMPLETE" if d["complete"] else "INCOMPLETE"
        print(f"  {name:<30s}  {evald:>5d}/{total:<5d}  {failed:>8d}  "
              f"{pct:>7.1f}%  {status}")


# ─── LaTeX table ──────────────────────────────────────────────────────────────

def print_latex_table(data: dict) -> None:
    """Print a LaTeX table* (double-column) summarising detection results."""
    # Sort ascending so worst model is first row, best model is last row
    slugs = sorted(data.keys(), key=lambda s: data[s]["f1"])

    metric_keys = ["precision", "recall", "f1", "ap50", "map50_95", "mean_iou"]
    best = {}
    for mk in metric_keys:
        vals = [data[s][mk] for s in slugs]
        best[mk] = max(vals) if vals else 0

    incomplete_slugs = [s for s in slugs if not data[s]["complete"]]

    def _fmt(val: float, is_best: bool) -> str:
        s = f"{val * 100:.1f}"
        return f"\\textbf{{{s}}}" if is_best else s

    print(f"\n{'='*80}")
    print("  LaTeX Table (copy-paste into your paper)")
    print(f"{'='*80}\n")

    dagger_note = ""

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{{Object detection results (\%) on the Colon-Bench "
        r"detection benchmark ({n_videos} videos). "
        r"Precision, Recall, and F1 are computed at IoU$\geq$0.50. "
        r"AP@50 is the average precision at IoU$\geq$0.50; "
        r"mAP@50-95 averages AP across IoU thresholds 0.50--0.95. "
        r"Mean IoU is the average intersection-over-union of matched "
        r"predicted and ground-truth boxes. "
        r"Best results per metric are shown in \textbf{{bold}}.{note}}}".format(
            n_videos=TOTAL_VIDEOS, note=dagger_note))
    lines.append(r"\label{tab:detection_results}")
    lines.append(r"\begin{tabular}{l c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Prec. & Rec. & F1 & AP@50 & mAP@50-95 & mIoU \\")
    lines.append(r"\midrule")

    for slug in slugs:
        d = data[slug]
        meta = MODEL_META.get(slug)
        name = meta[0].replace("\n", " ") if meta else slug
        name_tex = name.replace("_", r"\_")
        if slug == slugs[-1]:
            name_tex = f"\\textbf{{{name_tex}}}"

        cols = []
        for mk in metric_keys:
            cols.append(_fmt(d[mk], abs(d[mk] - best[mk]) < 1e-9 and best[mk] > 0))

        lines.append(f"{name_tex} & {' & '.join(cols)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    print("\n".join(lines))
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot detection metrics bar charts for colon-bench"
    )
    parser.add_argument(
        "--results", type=str, default=DEFAULT_RESULTS_DIR,
        help=f"Directory with llm_eval_*_det_*.json files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--logos", type=str, default=DEFAULT_LOGOS_DIR,
        help=f"Directory with model logo PNGs (default: {DEFAULT_LOGOS_DIR})",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save figure to this path (e.g. --save ablations/detection_metrics.pdf). "
             "If omitted, displays interactively.",
    )
    parser.add_argument(
        "--exclude-models", nargs="+", metavar="SLUG", default=[],
        help="Model slugs to exclude from plots, tables, and reports "
             "(e.g. --exclude-models qwen3.5-plus molmo-2-8b).",
    )
    args = parser.parse_args()

    print(f"Loading detection results from: {args.results}")
    data = load_results(args.results)

    if args.exclude_models:
        exclude = set(args.exclude_models)
        for slug in exclude:
            data.pop(slug, None)
        print(f"  Excluded models: {', '.join(sorted(exclude))}")

    print(f"\n{'='*60}")
    print(f"  Detection benchmark — {len(data)} models")
    print(f"{'='*60}")
    for slug in sorted(data.keys(), key=lambda s: data[s]["f1"], reverse=True):
        d = data[slug]
        meta = MODEL_META.get(slug)
        name = meta[0].replace("\n", " ") if meta else slug
        print(
            f"  {name:30s}  F1={d['f1']*100:5.1f}%  "
            f"AP50={d['ap50']*100:5.1f}%  "
            f"mAP={d['map50_95']*100:5.1f}%  "
            f"mIoU={d['mean_iou']*100:5.1f}%  "
            f"({d['videos_evaluated']}/{d['total_videos']} videos)"
        )

    print_completion_report(data)

    print("Plotting detection metrics...")
    plot_detection(data, args.logos, save_path=args.save)
    print("\nDone!")


if __name__ == "__main__":
    main()
