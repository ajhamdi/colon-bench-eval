#!/usr/bin/env python3
"""Public ColonBench classification evaluation runner."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from colonbench_eval.hf_video import (  # noqa: E402
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_DATASET_REVISION,
    HFDatasetAssetResolver,
)
from colonbench_eval.io_utils import ensure_dir, find_previous_results, load_json, sanitize_model_name, save_json_atomic  # noqa: E402
from colonbench_eval.openrouter_utils import (  # noqa: E402
    DEFAULT_EXTRACTION_MODEL,
    VideoFrameCache,
    list_openrouter_models,
    resolve_openrouter_model,
    run_openrouter_completion,
    run_openrouter_text_completion,
)


BENCHMARK_FILENAME = "benchmark_cls.json"
VIDEOS_SUBDIR = "videos"
MODE_TAG = "cls"

CLASSIFICATION_PROMPT_TEMPLATE = """\
You are a medical AI expert analyzing a colonoscopy video.

Task:
Determine whether a lesion is visible in this video.

Label definition:
- 1 = lesion visible
- 0 = no lesion visible

Instructions:
- Watch the video carefully.
- Return ONLY a single character: 0 or 1.
- Do not add any explanation.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
You are an answer extractor. Extract the final binary prediction from the text.

Valid outputs:
- 1 = lesion visible
- 0 = no lesion visible

Text:
{raw_response}

Return ONLY one character: 0 or 1.
"""


def build_classification_prompt() -> str:
    return CLASSIFICATION_PROMPT_TEMPLATE


def make_record_id(item: Dict, idx: int) -> str:
    for key in ("record_id", "sample_id", "id", "question_id"):
        if item.get(key):
            return str(item[key])
    if item.get("video_id"):
        return str(item["video_id"])
    return f"idx_{idx:06d}"


def extract_binary_prediction(raw_response: str) -> Tuple[Optional[int], Optional[float]]:
    text = (raw_response or "").strip()
    if not text:
        return None, None
    lower = text.lower()

    strict_patterns = [
        r"^\s*\(?([01])\)?\s*\.?\s*$",
        r"(?:^|\n)\s*(?:answer|prediction|label|class)\s*[:=\-]?\s*\(?([01])\)?\s*(?:$|\n)",
        r"(?:^|\n)\s*(?:final\s+answer|final\s+prediction)\s*[:=\-]?\s*\(?([01])\)?\s*(?:$|\n)",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return int(match.group(1)), _extract_probability(lower)

    positive_patterns = [
        r"\blesion\s+(?:is\s+)?(?:present|visible|detected)\b",
        r"\b(there\s+is|contains)\s+(?:a\s+)?lesion\b",
        r"\bpositive\b",
        r"\byes\b",
    ]
    negative_patterns = [
        r"\bno\s+lesion\b",
        r"\blesion\s+(?:is\s+)?(?:absent|not\s+visible|not\s+detected)\b",
        r"\bnegative\b",
        r"\bno\b",
    ]

    pos_match = any(re.search(pattern, lower) for pattern in positive_patterns)
    neg_match = any(re.search(pattern, lower) for pattern in negative_patterns)
    if pos_match and not neg_match:
        return 1, _extract_probability(lower)
    if neg_match and not pos_match:
        return 0, _extract_probability(lower)

    for token in re.findall(r"[01]", text):
        return int(token), _extract_probability(lower)
    return None, _extract_probability(lower)


def _extract_probability(lower_text: str) -> Optional[float]:
    patterns = [
        r"(?:p(?:robability)?|confidence)\s*(?:of\s+lesion|for\s+lesion)?\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        r"(?:lesion\s+probability)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower_text)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            return value
    return None


def extract_prediction_with_llm_fallback(
    raw_response: str,
    extraction_model: str,
) -> Tuple[int, str, Optional[float]]:
    label, score = extract_binary_prediction(raw_response)
    if label in (0, 1):
        return label, "regex_semantic", score

    try:
        extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(raw_response=raw_response)
        _, extraction_text = run_openrouter_text_completion(extraction_model, extraction_prompt)
        label, score2 = extract_binary_prediction(extraction_text)
        if label in (0, 1):
            return label, "llm_fallback", score2 if score2 is not None else score
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN] Extraction LLM failed: {exc}")

    return 0, "default", score


def _resolve_local_video_path(
    *,
    video_id: str,
    videos_dir: Optional[str],
    asset_resolver: HFDatasetAssetResolver,
) -> str:
    if videos_dir:
        candidate = os.path.join(videos_dir, video_id)
        if os.path.isfile(candidate):
            return candidate
    return asset_resolver.download_video(video_id, local_videos_dir=videos_dir)


def evaluate_single_item(
    *,
    item: Dict,
    idx: int,
    model: str,
    frame_cache: VideoFrameCache,
    asset_resolver: HFDatasetAssetResolver,
    videos_dir: Optional[str],
    extraction_model: str,
    use_video: bool,
    max_retries: int,
    verbose: bool,
) -> Optional[Dict]:
    record_id = make_record_id(item, idx)
    video_name = item.get("video_id", "")
    correct_label = item.get("lesion", None)
    if correct_label not in (0, 1) or not video_name:
        return None

    prompt_text = build_classification_prompt()
    effective_use_video = use_video
    remote_video_url: Optional[str] = None
    local_video_path: Optional[str] = None
    frames_b64: Optional[List[str]] = None

    if effective_use_video:
        try:
            remote_video_url = asset_resolver.resolve_video_url(video_name)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(
                    f"  [WARN] Record {record_id}: remote video URL failed ({exc}); "
                    "falling back to frame extraction"
                )
            effective_use_video = False

    if not effective_use_video:
        try:
            local_video_path = _resolve_local_video_path(
                video_id=video_name,
                videos_dir=videos_dir,
                asset_resolver=asset_resolver,
            )
            frames_b64 = frame_cache.get_or_extract(local_video_path)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [ERROR] Record {record_id}: local video prep failed ({exc})")
            return None

    raw_response = ""
    completion = None
    last_error = None
    for attempt in range(max_retries):
        try:
            completion, raw_response = run_openrouter_completion(
                model,
                prompt_text,
                video_path=(
                    local_video_path
                    if (effective_use_video and remote_video_url is None)
                    else None
                ),
                video_url=remote_video_url,
                frames_b64=frames_b64,
                use_video=effective_use_video,
                max_retries=max_retries,
            )
            if raw_response:
                break
            last_error = "Empty response"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if effective_use_video:
                effective_use_video = False
                remote_video_url = None
                try:
                    local_video_path = _resolve_local_video_path(
                        video_id=video_name,
                        videos_dir=videos_dir,
                        asset_resolver=asset_resolver,
                    )
                    frames_b64 = frame_cache.get_or_extract(local_video_path)
                    completion, raw_response = run_openrouter_completion(
                        model,
                        prompt_text,
                        frames_b64=frames_b64,
                        use_video=False,
                        max_retries=max_retries,
                    )
                    if raw_response:
                        break
                except Exception as fallback_exc:  # noqa: BLE001
                    last_error = f"{exc}; frame fallback failed: {fallback_exc}"
        if attempt < max_retries - 1 and not raw_response:
            time.sleep(2 * (attempt + 1))

    if not raw_response:
        if verbose:
            print(f"  [FAIL] Record {record_id}: {last_error}")
        return None

    predicted_label, extraction_tier, predicted_score = extract_prediction_with_llm_fallback(
        raw_response,
        extraction_model,
    )
    is_correct = predicted_label == correct_label

    prompt_tokens = 0
    completion_tokens = 0
    if completion is not None and hasattr(completion, "usage"):
        prompt_tokens = int(getattr(completion.usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(completion.usage, "completion_tokens", 0) or 0)

    result = {
        "record_id": record_id,
        "video_id": video_name,
        "correct_label": int(correct_label),
        "predicted_label": int(predicted_label),
        "is_correct": is_correct,
        "raw_response": raw_response,
        "extraction_tier": extraction_tier,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if predicted_score is not None:
        result["predicted_score"] = predicted_score
    return result


def _collect_scores(results: List[Dict]) -> Optional[List[float]]:
    scores: List[float] = []
    for result in results:
        score = result.get("predicted_score")
        if score is None:
            return None
        try:
            value = float(score)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= value <= 1.0):
            return None
        scores.append(value)
    return scores


def compute_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0])
    rank_sum_pos = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum_pos += rank
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def compute_average_precision(y_true: List[int], y_scores: List[float]) -> float:
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    n_pos = sum(y_true)
    if n_pos == 0:
        return 0.0
    tp = 0
    ap_sum = 0.0
    for idx, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            tp += 1
            ap_sum += tp / idx
    return ap_sum / n_pos


def compute_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "total": 0,
            "correct": 0,
            "wrong": 0,
            "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            "predicted_distribution": {"0": 0, "1": 0},
            "token_usage": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    y_true = [int(result["correct_label"]) for result in results]
    y_pred = [int(result["predicted_label"]) for result in results]
    total = len(results)
    correct = sum(1 for result in results if result.get("is_correct"))

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    total_prompt_tokens = sum(int(result.get("prompt_tokens") or 0) for result in results)
    total_completion_tokens = sum(int(result.get("completion_tokens") or 0) for result in results)

    metrics = {
        "accuracy": correct / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "predicted_distribution": {
            "0": sum(1 for pred in y_pred if pred == 0),
            "1": sum(1 for pred in y_pred if pred == 1),
        },
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }

    scores = _collect_scores(results)
    if scores is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = compute_roc_auc(y_true, scores)
        metrics["average_precision"] = compute_average_precision(y_true, scores)
    return metrics


def save_results_file(
    path: str,
    results: List[Dict],
    total_records: int,
    model: str,
) -> Dict:
    metrics = compute_metrics(results)
    records_failed = total_records - len(results)
    payload = {
        "metadata": {
            "model": model,
            "last_updated": datetime.now().isoformat(),
            "total_records": total_records,
            "records_evaluated": len(results),
            "records_failed": records_failed,
            "complete": records_failed == 0,
        },
        "metrics": metrics,
        "results": results,
    }
    save_json_atomic(payload, path)
    return metrics


def print_evaluation_report(metrics: Dict, model: str) -> None:
    print("\n" + "=" * 64)
    print(f"  ColonBench Classification Report - {model}")
    print("=" * 64)
    print(f"  Accuracy:  {metrics.get('accuracy', 0.0):.4f} ({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
    print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"  F1:        {metrics.get('f1', 0.0):.4f}")
    print("=" * 64)


def resolve_benchmark_paths(
    benchmark: str,
    videos: Optional[str],
) -> Tuple[str, Optional[str], str]:
    if os.path.isdir(benchmark):
        benchmark_dir = benchmark
        benchmark_json_path = os.path.join(benchmark_dir, BENCHMARK_FILENAME)
        videos_dir = videos or os.path.join(benchmark_dir, VIDEOS_SUBDIR)
    elif os.path.isfile(benchmark):
        benchmark_json_path = benchmark
        benchmark_dir = os.path.dirname(benchmark_json_path)
        videos_dir = videos or os.path.join(benchmark_dir, VIDEOS_SUBDIR)
    else:
        raise FileNotFoundError(f"Benchmark path not found: {benchmark}")

    if not os.path.isfile(benchmark_json_path):
        raise FileNotFoundError(f"Classification benchmark not found: {benchmark_json_path}")
    if videos_dir and not os.path.isdir(videos_dir):
        videos_dir = None
    return benchmark_json_path, videos_dir, benchmark_dir


def run_evaluation(
    *,
    benchmark: List[Dict],
    videos_dir: Optional[str],
    asset_resolver: HFDatasetAssetResolver,
    model: str,
    output_dir: str,
    extraction_model: str,
    parallel: bool,
    max_workers: int,
    max_retries: int,
    new_run: bool,
    num_frames: int,
    use_frames: bool,
    verbose: bool,
) -> Tuple[List[Dict], Dict, str]:
    ensure_dir(output_dir)
    total_records = len(benchmark)
    model_safe = sanitize_model_name(model)
    results_prefix = f"llm_eval_{model_safe}_{MODE_TAG}_"

    existing_results: Dict[str, Dict] = {}
    results_path: Optional[str] = None
    if not new_run:
        prev_path = find_previous_results(output_dir, results_prefix)
        if prev_path:
            prev_data = load_json(prev_path)
            for result in prev_data.get("results", []):
                record_id = result.get("record_id") or result.get("video_id")
                if record_id:
                    existing_results[str(record_id)] = result
            results_path = prev_path
            print(f"  Resuming from: {prev_path}")
            print(f"  Previously completed: {len(existing_results)}/{total_records}")

    if results_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"{results_prefix}{timestamp}.json")
        if not new_run:
            print("  No previous results found - starting fresh.")

    to_process: List[Tuple[int, Dict]] = []
    results: List[Dict] = []
    for idx, item in enumerate(benchmark):
        record_id = make_record_id(item, idx)
        if record_id in existing_results:
            results.append(existing_results[record_id])
        else:
            to_process.append((idx, item))

    if not to_process:
        metrics = save_results_file(results_path, results, total_records, model)
        return results, metrics, results_path

    frame_cache = VideoFrameCache(num_frames=num_frames)
    _, supports_video = resolve_openrouter_model(model)
    send_video = supports_video and not use_frames
    print(f"  Video input: {'remote video_url' if send_video else f'frames ({num_frames})'}")

    def process_record(pair: Tuple[int, Dict]) -> Optional[Dict]:
        idx, item = pair
        return evaluate_single_item(
            item=item,
            idx=idx,
            model=model,
            frame_cache=frame_cache,
            asset_resolver=asset_resolver,
            videos_dir=videos_dir,
            extraction_model=extraction_model,
            use_video=send_video,
            max_retries=max_retries,
            verbose=verbose,
        )

    completed_this_run = 0
    failed_this_run = 0
    metrics: Dict = {}
    try:
        if parallel and len(to_process) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_record, pair): pair
                    for pair in to_process
                }
                for future in as_completed(futures):
                    completed_this_run += 1
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        failed_this_run += 1
                    if (completed_this_run % 10 == 0) or (completed_this_run == len(to_process)):
                        metrics = save_results_file(results_path, results, total_records, model)
        else:
            for pair in to_process:
                completed_this_run += 1
                result = process_record(pair)
                if result is not None:
                    results.append(result)
                else:
                    failed_this_run += 1
                if (completed_this_run % 10 == 0) or (completed_this_run == len(to_process)):
                    metrics = save_results_file(results_path, results, total_records, model)
    finally:
        try:
            save_results_file(results_path, results, total_records, model)
        except Exception:
            pass

    missing = total_records - len(results)
    if missing == 0:
        print(f"\n  COMPLETE - all {total_records} records evaluated (fail=0)")
    else:
        print(f"\n  Done: {len(results)}/{total_records} records ({missing} missing). Re-run to retry.")
    if failed_this_run:
        print(f"  Failed in this run: {failed_this_run}")

    metrics = save_results_file(results_path, results, total_records, model)
    return results, metrics, results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ColonBench classification evaluation.")
    parser.add_argument("--benchmark", type=str, default="data/colon-bench", help="Benchmark directory or benchmark JSON path.")
    parser.add_argument("--videos", type=str, default=None, help="Optional local videos directory.")
    parser.add_argument("--model", type=str, default="qwen3-vl-8b", help="OpenRouter model name.")
    parser.add_argument("--extraction-model", type=str, default=DEFAULT_EXTRACTION_MODEL, help="Model used for answer extraction fallback.")
    parser.add_argument("--dataset-repo-id", type=str, default=DEFAULT_DATASET_REPO_ID, help="HF dataset repo for asset resolution.")
    parser.add_argument("--dataset-revision", type=str, default=DEFAULT_DATASET_REVISION, help="HF dataset revision.")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HF token override.")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: <benchmark_dir>/cls_results).")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel evaluation.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per record.")
    parser.add_argument("--new", action="store_true", help="Ignore previous results and start fresh.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N records.")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames to sample when frame mode is used.")
    parser.add_argument("--use-frames", action="store_true", help="Force frame mode even for video-capable models.")
    parser.add_argument("--verbose", action="store_true", help="Print debug details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_json_path, videos_dir, benchmark_dir = resolve_benchmark_paths(args.benchmark, args.videos)
    output_dir = args.output or os.path.join(benchmark_dir, "cls_results")
    ensure_dir(output_dir)

    benchmark = load_json(benchmark_json_path)
    if args.limit is not None:
        benchmark = benchmark[: args.limit]

    asset_resolver = HFDatasetAssetResolver(
        repo_id=args.dataset_repo_id,
        revision=args.dataset_revision,
        token=args.hf_token,
    )
    resolved_id, supports_video = resolve_openrouter_model(args.model)
    print("\n" + "=" * 72)
    print("  ColonBench Classification Evaluation")
    print("=" * 72)
    print(f"  Benchmark:         {benchmark_json_path}")
    print(f"  Model:             {args.model}")
    print(f"  OpenRouter ID:     {resolved_id}")
    print(f"  Dataset repo:      {args.dataset_repo_id}@{args.dataset_revision}")
    print(f"  Local videos dir:  {videos_dir or '(not provided - resolve/download from HF)'}")
    print(f"  Output dir:        {output_dir}")
    print(f"  Records:           {len(benchmark)}")
    print(f"  Parallel:          {args.parallel} (workers: {args.max_workers})")
    print(f"  Video input:       {'remote video_url' if (supports_video and not args.use_frames) else f'frames ({args.num_frames})'}")
    print("=" * 72)

    results, metrics, results_path = run_evaluation(
        benchmark=benchmark,
        videos_dir=videos_dir,
        asset_resolver=asset_resolver,
        model=args.model,
        output_dir=output_dir,
        extraction_model=args.extraction_model,
        parallel=args.parallel,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        new_run=args.new,
        num_frames=args.num_frames,
        use_frames=args.use_frames,
        verbose=args.verbose,
    )
    _ = results
    print_evaluation_report(metrics, args.model)
    print(f"\n  Results JSON: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
