"""
plot_vqa_accuracy.py

Plots bar charts of VQA accuracy for all evaluated models on the colon-bench
prompted and unprompted benchmarks. Accuracy is recalculated from individual results
(correct / total_questions) rather than using the pre-computed metric, so that
models with failed answers are scored fairly (failures count as wrong).

Model logos from ablations/logos/ are placed under each bar.

Usage:
    python scripts/plot_vqa_accuracy.py
    python scripts/plot_vqa_accuracy.py --results data/colon-bench/eval_results
    python scripts/plot_vqa_accuracy.py --save plots/vqa_accuracy_prompted.pdf plots/vqa_accuracy_unprompted.pdf
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

DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "colon-bench", "eval_results")
DEFAULT_BENCH_DIR = os.path.join(PROJECT_ROOT, "data", "colon-bench")
DEFAULT_LOGOS_DIR = os.path.join(SCRIPT_DIR, "logos")

# Total questions per benchmark (used as denominator so failures count as wrong)
BENCH_TOTALS = {
    "prompted": 1485,
    "unprompted": 2740,
}

# ─── Model display names and logo mapping ────────────────────────────────────
# Maps the model slug (from filename) to (display_name, logo_filename).
# Models sharing a family share a logo.
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
    # Future models with logos already available
    "gpt-4o":                         ("GPT-4o",                     "gpt.png"),
    "gpt-5.4":                        ("GPT-5.4",                    "gpt.png"),
    "nova-premier":                   ("Nova\nPremier",              "nova.png"),
}

# Desired display order (models will appear in this order on the x-axis).
# Models not listed here are appended alphabetically at the end.
DISPLAY_ORDER = [
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "seed-1.6",
    "seed-1.6-flash",
    "glm-4.6v",
    "qwen3-vl-235b",
    "qwen3-vl-plus",
    "qwen-vl-max",
    "qwen3.5-397b-a17b",
    "qwen3.5-plus",
    "qwen3-vl-32b",
    "qwen3-vl-8b",
    "molmo-2-8b",
    "nemotron-nano-12b-v2-vl-free",
    "gpt-4o",
    "gpt-5.4",
    "nova-premier",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_eval_filename(filename: str):
    """
    Parse an eval-results filename like
        llm_eval_qwen3-vl-8b_prompted_20260213_043311.json
    Returns (model_slug, mode) or (None, None) on failure.
    """
    m = re.match(r"llm_eval_(.+?)_(prompted|unprompted)_\d{8}_\d{6}\.json$", filename)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def load_results(results_dir: str):
    """
    Load all eval JSONs.  Accuracy is recomputed from individual results
    (correct / total_questions) so that failures count as wrong.
    Completion info (evaluated, failed, complete) is read directly from the
    JSON metadata header.

    Returns dict: { mode: { model_slug: { 'correct': int,
                    'bench_total': int, 'accuracy': float,
                    'total_answered': int,
                    'questions_evaluated': int, 'questions_failed': int,
                    'complete': bool } } }
    where mode is 'prompted' or 'unprompted'.
    """
    data = {"prompted": {}, "unprompted": {}}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        model_slug, mode = parse_eval_filename(fname)
        if model_slug is None:
            continue
        if mode not in data:
            continue

        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath, "r") as f:
                content = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] Skipping {fname}: {e}")
            continue

        meta = content.get("metadata", {})
        results_list = content.get("results", [])

        # Recompute metrics from raw results
        total_answered = len(results_list)
        correct = sum(1 for r in results_list if r.get("is_correct"))
        bench_total = BENCH_TOTALS.get(mode, total_answered)
        accuracy = correct / bench_total if bench_total > 0 else 0.0

        # Completion info from metadata header
        questions_evaluated = meta.get("questions_evaluated", 0)
        questions_failed = meta.get("questions_failed", 0)
        complete = meta.get("complete", False)

        # Keep the result with the most answers if duplicates exist
        prev = data[mode].get(model_slug)
        if prev is None or total_answered > prev["total_answered"]:
            data[mode][model_slug] = {
                "correct": correct,
                "total_answered": total_answered,
                "bench_total": bench_total,
                "accuracy": accuracy,
                "questions_evaluated": questions_evaluated,
                "questions_failed": questions_failed,
                "complete": complete,
            }

    return data


def load_logo(logo_path: str, target_height: int = 40) -> np.ndarray | None:
    """Load a logo image and resize to target height, preserving aspect ratio."""
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


def sort_models(model_slugs: list[str], mode_data: dict | None = None) -> list[str]:
    """Sort model slugs by accuracy (highest first). Falls back to DISPLAY_ORDER."""
    if mode_data is not None:
        return sorted(model_slugs,
                      key=lambda s: mode_data[s]["accuracy"], reverse=True)
    order_map = {slug: i for i, slug in enumerate(DISPLAY_ORDER)}
    def key_fn(slug):
        return (order_map.get(slug, 9999), slug)
    return sorted(model_slugs, key=key_fn)


# ─── Colour palette ──────────────────────────────────────────────────────────
# Assign a consistent colour per model family.
FAMILY_COLORS = {
    "gemini":   "#4285F4",  # Google blue
    "seed":     "#34A853",  # green
    "glm":      "#8E44AD",  # purple
    "qwen":     "#E67E22",  # orange
    "molmo":    "#E74C3C",  # red
    "nemotron": "#76B900",  # NVIDIA green
    "gpt":      "#10A37F",  # OpenAI teal
    "nova":     "#FF9900",  # AWS orange
}


def _family(slug: str) -> str:
    """Derive family key from model slug."""
    for fam in FAMILY_COLORS:
        if fam in slug:
            return fam
    return "other"


def get_bar_color(slug: str) -> str:
    fam = _family(slug)
    return FAMILY_COLORS.get(fam, "#95A5A6")


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_accuracy(
    mode_data: dict,
    mode: str,
    logos_dir: str,
    save_path: str | None = None,
) -> None:
    """
    Plot a horizontal-style vertical bar chart for one benchmark mode.
    """
    if not mode_data:
        print(f"  [WARN] No data for mode={mode}, skipping.")
        return

    slugs = sort_models(list(mode_data.keys()), mode_data)
    accuracies = [mode_data[s]["accuracy"] * 100 for s in slugs]
    corrects = [mode_data[s]["correct"] for s in slugs]
    totals = [mode_data[s]["bench_total"] for s in slugs]
    colors = [get_bar_color(s) for s in slugs]

    n = len(slugs)
    fig_width = max(14, n * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    x = np.arange(n)
    bar_width = 0.65
    bars = ax.bar(x, accuracies, width=bar_width, color=colors,
                  edgecolor="white", linewidth=0.8, alpha=0.90, zorder=3)

    # ── Value labels on top of bars ──────────────────────────────────────
    for i, (bar, acc, cor, tot) in enumerate(zip(bars, accuracies, corrects, totals)):
        # Percentage label
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            color="#333333",
        )

    # ── Random-chance baseline ───────────────────────────────────────────
    ax.axhline(y=20, color="#C44E52", linewidth=1.5, linestyle="--",
               alpha=0.7, zorder=2, label="Random chance (20%)")

    # ── Logos + model names on x-axis ────────────────────────────────────
    ax.set_xticks(x)
    # We'll place logos below, so use display names as tick labels
    display_names = []
    for s in slugs:
        meta = MODEL_META.get(s)
        display_names.append(meta[0] if meta else s)
    ax.set_xticklabels(display_names, fontsize=15, ha="center",
                       linespacing=1.1)

    # Place logos below tick labels
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
            imagebox,
            (i, 0),
            xybox=(0, -56),
            xycoords=("data", "axes fraction"),
            boxcoords="offset points",
            box_alignment=(0.5, 1.0),
            frameon=False,
            pad=0,
        )
        ax.add_artist(ab)

    # ── Axes formatting ──────────────────────────────────────────────────
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.set_ylim(0, min(max(accuracies) + 14, 105))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)

    mode_label = "Prompted VQA" if mode == "prompted" else "Unprompted VQA"
    ax.set_title(
        f"Colon-Bench — {mode_label}",
        fontsize=20, fontweight="bold", pad=16,
    )

    ax.legend(loc="upper right", fontsize=15, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Extra bottom margin for logos
    fig.subplots_adjust(bottom=0.20)

    if save_path:
        save_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(save_path, bbox_inches="tight", format="pdf")
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ─── Completion report ────────────────────────────────────────────────────────

def print_completion_report(all_data: dict) -> None:
    """Print a per-model completion report showing evaluation success rates."""
    for mode in ("prompted", "unprompted"):
        mode_data = all_data[mode]
        if not mode_data:
            continue
        slugs = sort_models(list(mode_data.keys()), mode_data)
        mode_label = "Prompted VQA" if mode == "prompted" else "Unprompted VQA"
        bench_total = BENCH_TOTALS.get(mode, "?")

        print(f"\n{'─'*80}")
        print(f"  COMPLETION REPORT — {mode_label} Benchmark ({bench_total} questions)")
        print(f"{'─'*80}")
        print(f"  {'Model':<30s}  {'Evaluated':>10s}  {'Failed':>8s}  "
              f"{'Success':>8s}  {'Status'}")
        print(f"  {'─'*30}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*12}")

        for slug in slugs:
            d = mode_data[slug]
            meta = MODEL_META.get(slug)
            name = meta[0].replace("\n", " ") if meta else slug
            evald = d["questions_evaluated"]
            total = d["bench_total"]
            failed = d["questions_failed"]
            pct = evald / total * 100 if total > 0 else 0.0
            status = "COMPLETE" if d["complete"] else f"INCOMPLETE"
            print(f"  {name:<30s}  {evald:>5d}/{total:<4d}  {failed:>8d}  "
                  f"{pct:>7.1f}%  {status}")


# ─── LaTeX table ──────────────────────────────────────────────────────────────

def print_latex_table(all_data: dict) -> None:
    """Print a LaTeX table* (double-column) summarising VQA accuracy."""
    # Collect all model slugs across both modes
    all_slugs = set()
    for mode in ("prompted", "unprompted"):
        all_slugs.update(all_data[mode].keys())
    # Sort ascending so worst model is first row, best model is last row
    slugs = sorted(all_slugs, key=lambda s: (
        all_data["prompted"].get(s, {}).get("accuracy", 0),
        all_data["unprompted"].get(s, {}).get("accuracy", 0),
    ))

    # Find best values for bolding
    best = {}
    for mode in ("prompted", "unprompted"):
        vals = [all_data[mode][s]["accuracy"] for s in slugs if s in all_data[mode]]
        best[mode] = max(vals) if vals else 0

    incomplete_slugs = []
    for slug in slugs:
        for mode in ("prompted", "unprompted"):
            d = all_data[mode].get(slug)
            if d and not d["complete"]:
                incomplete_slugs.append(slug)
                break

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
        r"\caption{{VQA accuracy (\%) on the Colon-Bench Prompted "
        r"({prompted} questions) and Unprompted ({unprompted} questions) benchmarks. "
        r"Accuracy is computed as correct answers divided by total questions; "
        r"unanswered questions count as incorrect. "
        r"Best results per benchmark are shown in \textbf{{bold}}.{note}}}".format(
            prompted=BENCH_TOTALS["prompted"], unprompted=BENCH_TOTALS["unprompted"], note=dagger_note))
    lines.append(r"\label{tab:vqa_accuracy}")
    lines.append(r"\begin{tabular}{l c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Prompted Acc. (\%) & Unprompted Acc. (\%) \\")
    lines.append(r"\midrule")

    for slug in slugs:
        meta = MODEL_META.get(slug)
        name = meta[0].replace("\n", " ") if meta else slug
        name_tex = name.replace("_", r"\_")
        if slug == slugs[-1]:
            name_tex = f"\\textbf{{{name_tex}}}"

        easy_d = all_data["prompted"].get(slug)
        hard_d = all_data["unprompted"].get(slug)

        easy_acc = _fmt(easy_d["accuracy"], abs(easy_d["accuracy"] - best["prompted"]) < 1e-9) if easy_d else "---"
        hard_acc = _fmt(hard_d["accuracy"], abs(hard_d["accuracy"] - best["unprompted"]) < 1e-9) if hard_d else "---"

        lines.append(f"{name_tex} & {easy_acc} & {hard_acc} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    print("\n".join(lines))
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot VQA accuracy bar charts for ColonBench prompted and unprompted splits"
    )
    parser.add_argument(
        "--results", type=str, default=DEFAULT_RESULTS_DIR,
        help=f"Directory with llm_eval_*.json files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--logos", type=str, default=DEFAULT_LOGOS_DIR,
        help=f"Directory with model logo PNGs (default: {DEFAULT_LOGOS_DIR})",
    )
    parser.add_argument(
        "--save", nargs=2, metavar=("PROMPTED_PATH", "UNPROMPTED_PATH"),
        default=None,
        help="Save figures to these paths (e.g. --save prompted.pdf unprompted.pdf). "
             "If omitted, displays interactively.",
    )
    parser.add_argument(
        "--exclude-models", nargs="+", metavar="SLUG", default=[],
        help="Model slugs to exclude from plots, tables, and reports "
             "(e.g. --exclude-models nemotron-nano-12b-v2-vl-free qwen3.5-plus).",
    )
    args = parser.parse_args()

    print(f"Loading eval results from: {args.results}")
    all_data = load_results(args.results)

    if args.exclude_models:
        exclude = set(args.exclude_models)
        for mode in all_data:
            for slug in exclude:
                all_data[mode].pop(slug, None)
        print(f"  Excluded models: {', '.join(sorted(exclude))}")

    for mode in ("prompted", "unprompted"):
        models_found = list(all_data[mode].keys())
        print(f"\n{'='*60}")
        print(f"  {mode.upper()} benchmark — {len(models_found)} models")
        print(f"{'='*60}")
        for slug in sort_models(models_found, all_data[mode]):
            d = all_data[mode][slug]
            meta = MODEL_META.get(slug)
            name = meta[0].replace("\n", " ") if meta else slug
            print(f"  {name:30s}  {d['accuracy']*100:5.1f}%  "
                  f"({d['correct']}/{d['bench_total']})")

    print_completion_report(all_data)

    prompted_save = args.save[0] if args.save else None
    unprompted_save = args.save[1] if args.save else None

    print("Plotting prompted benchmark...")
    plot_accuracy(all_data["prompted"], "prompted", args.logos, save_path=prompted_save)

    print("Plotting unprompted benchmark...")
    plot_accuracy(all_data["unprompted"], "unprompted", args.logos, save_path=unprompted_save)

    print("\nDone!")


if __name__ == "__main__":
    main()
