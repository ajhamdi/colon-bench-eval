"""
plot_classification_metrics.py

Plots bar charts of classification metrics (Accuracy, F1, Precision, Recall)
for all evaluated models on the colon-bench classification benchmark.

Metrics are read directly from result JSON files produced by
baselines/llm_evaluate_cls.py.

Two figures are generated:
  1. Accuracy bar chart (single bars, sorted by accuracy)
  2. Grouped bar chart with Precision, Recall, and F1

Model logos from ablations/logos/ are placed under each bar.

Usage:
    python ablations/plot_classification_metrics.py
    python ablations/plot_classification_metrics.py --results data/colon-bench/cls_results
    python ablations/plot_classification_metrics.py --save ablations/cls_accuracy.pdf ablations/cls_prf1.pdf
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

DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "colon-bench", "cls_results")
DEFAULT_LOGOS_DIR = os.path.join(SCRIPT_DIR, "logos")

TOTAL_RECORDS = 790

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
    # CLIP / ViCLIP baselines
    "Clip":                           ("CLIP",                       "gpt.png"),
    "Endo-Clip":                      ("Endo-CLIP",                  "gpt.png"),
    "ViClip":                         ("ViCLIP",                     "viclip.jpeg"),
    "Colon-ViClip":                   ("Colon-\nViCLIP",             "colon.jpg"),
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
    "clip":     "#1B9E77",
    "viclip":   "#D95F02",
}


def _family(slug: str) -> str:
    slug_lower = slug.lower()
    # Check longer keys first so "viclip" matches before "clip"
    for fam in sorted(FAMILY_COLORS, key=len, reverse=True):
        if fam in slug_lower:
            return fam
    return "other"


def get_bar_color(slug: str) -> str:
    return FAMILY_COLORS.get(_family(slug), "#95A5A6")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_cls_filename(filename: str):
    """
    Parse a classification results filename like
        llm_eval_gemini-3-flash-preview_cls_20260216_130212.json
    Returns model_slug or None on failure.
    """
    m = re.match(r"llm_eval_(.+?)_cls_\d{8}_\d{6}\.json$", filename)
    if not m:
        return None
    return m.group(1)


def load_results(results_dir: str) -> dict:
    """
    Load all classification eval JSONs.  Accuracy is recomputed as
    correct / total_records so that unevaluated records count as wrong.
    Completion info (evaluated, failed, complete) is read directly from the
    JSON metadata header.

    Returns dict: { model_slug: { 'accuracy': float, 'precision': float,
                    'recall': float, 'f1': float, 'correct': int,
                    'total_records': int, 'records_evaluated': int,
                    'records_failed': int, 'complete': bool,
                    'confusion_matrix': dict } }
    """
    data = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        model_slug = parse_cls_filename(fname)
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

        total_records = meta.get("total_records", TOTAL_RECORDS)

        # Recompute accuracy so unevaluated records count as wrong
        correct = metrics.get("correct", 0)
        accuracy = correct / total_records if total_records > 0 else 0.0

        # Completion info from metadata header
        records_evaluated = meta.get("records_evaluated", 0)
        records_failed = meta.get("records_failed", 0)
        complete = meta.get("complete", False)

        entry = {
            "accuracy": accuracy,
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "correct": correct,
            "wrong": metrics.get("wrong", 0),
            "total_records": total_records,
            "records_evaluated": records_evaluated,
            "records_failed": records_failed,
            "complete": complete,
            "confusion_matrix": metrics.get("confusion_matrix", {}),
        }

        prev = data.get(model_slug)
        if prev is None or entry["records_evaluated"] > prev["records_evaluated"]:
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


def _add_logos(ax, slugs, logos_dir):
    """Place model logos under the x-axis tick labels."""
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


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_accuracy(
    data: dict,
    logos_dir: str,
    save_path: str | None = None,
) -> None:
    """Plot accuracy bar chart (single bars, colour-coded by model family)."""
    if not data:
        print("  [WARN] No classification data found, skipping.")
        return

    slugs = sorted(data.keys(), key=lambda s: data[s]["accuracy"], reverse=True)
    accuracies = [data[s]["accuracy"] * 100 for s in slugs]
    colors = [get_bar_color(s) for s in slugs]
    n = len(slugs)

    fig_width = max(14, n * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    x = np.arange(n)
    bar_width = 0.65
    bars = ax.bar(x, accuracies, width=bar_width, color=colors,
                  edgecolor="white", linewidth=0.8, alpha=0.90, zorder=3)

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.0,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="#333333",
        )

    # Random chance baseline (binary classification → 50%)
    ax.axhline(y=50, color="#C44E52", linewidth=1.5, linestyle="--",
               alpha=0.7, zorder=2, label="Random chance (50%)")

    # X-axis labels
    ax.set_xticks(x)
    display_names = []
    for s in slugs:
        meta = MODEL_META.get(s)
        display_names.append(meta[0] if meta else s)
    ax.set_xticklabels(display_names, fontsize=15, ha="center", linespacing=1.1)
    _add_logos(ax, slugs, logos_dir)

    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.set_ylim(0, min(max(accuracies) + 14, 105))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    ax.set_title(
        "Colon-Bench Lesion Classification — Accuracy",
        fontsize=20, fontweight="bold", pad=16,
    )

    ax.legend(loc="upper right", fontsize=15, framealpha=0.9)
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


def plot_prf1(
    data: dict,
    logos_dir: str,
    save_path: str | None = None,
) -> None:
    """Plot grouped bar chart with Precision, Recall, and F1."""
    if not data:
        print("  [WARN] No classification data found, skipping.")
        return

    metrics_to_plot = ("precision", "recall", "f1")
    metric_labels = {"precision": "Precision", "recall": "Recall", "f1": "F1 Score"}
    metric_colors = {"precision": "#2196F3", "recall": "#FF9800", "f1": "#4CAF50"}

    # Sort by F1 descending
    slugs = sorted(data.keys(), key=lambda s: data[s]["f1"], reverse=True)
    n = len(slugs)
    n_metrics = len(metrics_to_plot)

    fig_width = max(14, n * 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.5))

    x = np.arange(n)
    total_bar_width = 0.75
    bar_width = total_bar_width / n_metrics

    for mi, metric in enumerate(metrics_to_plot):
        offset = (mi - (n_metrics - 1) / 2) * bar_width
        values = [data[s][metric] * 100 for s in slugs]
        bars = ax.bar(
            x + offset, values, width=bar_width,
            color=metric_colors[metric], edgecolor="white", linewidth=0.6,
            alpha=0.88, zorder=3, label=metric_labels[metric],
        )
        for bar, val in zip(bars, values):
            if val > 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                    color="#333333",
                )

    # X-axis labels + logos
    ax.set_xticks(x)
    display_names = []
    for s in slugs:
        meta = MODEL_META.get(s)
        display_names.append(meta[0] if meta else s)
    ax.set_xticklabels(display_names, fontsize=15, ha="center", linespacing=1.1)
    _add_logos(ax, slugs, logos_dir)

    ax.set_ylabel("Score (%)", fontsize=18)
    max_val = max(
        data[s][m] * 100 for s in slugs for m in metrics_to_plot
    )
    ax.set_ylim(0, min(max_val + 14, 105))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    ax.set_title(
        "Colon-Bench Lesion Classification — Precision / Recall / F1",
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
    slugs = sorted(data.keys(), key=lambda s: data[s]["accuracy"], reverse=True)

    print(f"\n{'─'*90}")
    print(f"  COMPLETION REPORT — Classification Benchmark ({TOTAL_RECORDS} records)")
    print(f"{'─'*90}")
    print(f"  {'Model':<30s}  {'Evaluated':>12s}  {'Failed':>8s}  "
          f"{'Success':>8s}  {'Status'}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*12}")

    for slug in slugs:
        d = data[slug]
        meta = MODEL_META.get(slug)
        name = meta[0].replace("\n", " ") if meta else slug
        evald = d["records_evaluated"]
        total = d["total_records"]
        failed = d["records_failed"]
        pct = evald / total * 100 if total > 0 else 0.0
        status = "COMPLETE" if d["complete"] else "INCOMPLETE"
        print(f"  {name:<30s}  {evald:>5d}/{total:<5d}  {failed:>8d}  "
              f"{pct:>7.1f}%  {status}")


# ─── LaTeX table ──────────────────────────────────────────────────────────────

def print_latex_table(data: dict) -> None:
    """Print a LaTeX table* (double-column) summarising classification results."""
    # Sort ascending so worst model is first row, best model is last row
    slugs = sorted(data.keys(), key=lambda s: data[s]["accuracy"])

    metric_keys = ["accuracy", "precision", "recall", "f1"]
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
        r"\caption{{Binary lesion classification results (\%) on the Colon-Bench "
        r"classification benchmark ({n_records} records). "
        r"Each model receives a video clip and predicts whether a lesion "
        r"is present (positive) or absent (negative). "
        r"Accuracy is computed over all records; unevaluated records count "
        r"as incorrect. Precision, Recall, and F1 are reported for the "
        r"positive (lesion-present) class. "
        r"Best results per metric are shown in \textbf{{bold}}.{note}}}".format(
            n_records=TOTAL_RECORDS, note=dagger_note))
    lines.append(r"\label{tab:classification_results}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Acc. & Prec. & Rec. & F1 \\")
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
        description="Plot classification metrics bar charts for colon-bench"
    )
    parser.add_argument(
        "--results", type=str, default=DEFAULT_RESULTS_DIR,
        help=f"Directory with llm_eval_*_cls_*.json files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--logos", type=str, default=DEFAULT_LOGOS_DIR,
        help=f"Directory with model logo PNGs (default: {DEFAULT_LOGOS_DIR})",
    )
    parser.add_argument(
        "--save", nargs=2, metavar=("ACC_PATH", "PRF1_PATH"),
        default=None,
        help="Save figures to these paths (e.g. --save cls_acc.pdf cls_prf1.pdf). "
             "If omitted, displays interactively.",
    )
    parser.add_argument(
        "--exclude-models", nargs="+", metavar="SLUG", default=[],
        help="Model slugs to exclude from plots, tables, and reports "
             "(e.g. --exclude-models nemotron-nano-12b-v2-vl-free qwen3.5-plus).",
    )
    args = parser.parse_args()

    print(f"Loading classification results from: {args.results}")
    data = load_results(args.results)

    if args.exclude_models:
        exclude = set(args.exclude_models)
        for slug in exclude:
            data.pop(slug, None)
        print(f"  Excluded models: {', '.join(sorted(exclude))}")

    print(f"\n{'='*60}")
    print(f"  Classification benchmark — {len(data)} models")
    print(f"{'='*60}")
    for slug in sorted(data.keys(), key=lambda s: data[s]["accuracy"], reverse=True):
        d = data[slug]
        meta = MODEL_META.get(slug)
        name = meta[0].replace("\n", " ") if meta else slug
        print(
            f"  {name:30s}  Acc={d['accuracy']*100:5.1f}%  "
            f"P={d['precision']*100:5.1f}%  R={d['recall']*100:5.1f}%  "
            f"F1={d['f1']*100:5.1f}%  "
            f"({d['records_evaluated']}/{d['total_records']} records)"
        )

    print_completion_report(data)

    acc_save = args.save[0] if args.save else None
    prf1_save = args.save[1] if args.save else None

    print("Plotting classification accuracy...")
    plot_accuracy(data, args.logos, save_path=acc_save)

    print("Plotting classification P/R/F1...")
    plot_prf1(data, args.logos, save_path=prf1_save)

    print("\nDone!")


if __name__ == "__main__":
    main()
