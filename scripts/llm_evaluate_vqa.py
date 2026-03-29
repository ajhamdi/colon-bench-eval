#!/usr/bin/env python3
"""
Public ColonBench VQA evaluation runner.

Key differences from the internal project version:
- uses public naming: `prompted` / `unprompted`
- uses OpenRouter/liteLLM for API evaluation
- adds `--local` for local Hugging Face Qwen3-VL evaluation
- resolves Hugging Face dataset videos directly instead of using GCS uploads

Examples:
  python scripts/llm_evaluate_vqa.py --benchmark data/colon-bench --mode prompted --model qwen3-vl-8b
  python scripts/llm_evaluate_vqa.py --benchmark data/colon-bench --mode unprompted --model gpt-4o --parallel
  python scripts/llm_evaluate_vqa.py --benchmark data/colon-bench --mode prompted --local
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
from colonbench_eval.io_utils import (  # noqa: E402
    ensure_dir,
    find_previous_results,
    load_json,
    project_root_from_script,
    sanitize_model_name,
    save_json_atomic,
)
from colonbench_eval.local_vlm import LocalQwenVideoVLM  # noqa: E402
from colonbench_eval.openrouter_utils import (  # noqa: E402
    DEFAULT_EXTRACTION_MODEL,
    VideoFrameCache,
    get_max_video_bytes,
    list_openrouter_models,
    resolve_openrouter_model,
    run_openrouter_completion,
    run_openrouter_text_completion,
)


VALID_ANSWER_KEYS = ["A", "B", "C", "D", "E"]
PROMPTED_BENCHMARK_FILENAME = "benchmark_vqa_prompted.json"
UNPROMPTED_BENCHMARK_FILENAME = "benchmark_vqa_unprompted.json"
VIDEOS_SUBDIR = "videos"

VQA_PROMPT_TEMPLATE = """\
You are a medical AI expert analyzing a colonoscopy video. Watch the video carefully \
and answer the following multiple-choice question.

Question: {question}

Options:
{options_text}

Instructions:
- Watch the entire video before answering.
- Select the single best answer based on what you observe in the video.
- Reply with ONLY the letter of your answer (e.g., "B").
"""

EXTRACTION_PROMPT_TEMPLATE = """\
You are an answer extractor. Extract the FINAL answer option from the text below. \
The text may discuss multiple options, but identify what the author concluded as \
their final answer. Only return the single letter of the answer. \
Valid options are: {valid_options}.

Text:
{raw_response}

Final answer letter:\
"""


def build_vqa_prompt(question: str, choices: Dict[str, str], skill_text: Optional[str] = None) -> str:
    options_text = "\n".join(
        f"  {key}. {choices[key]}" for key in sorted(choices.keys())
    )
    prompt_text = VQA_PROMPT_TEMPLATE.format(
        question=question,
        options_text=options_text,
    )
    if skill_text:
        prompt_text = f"{prompt_text}\n\n{skill_text.strip()}"
    return prompt_text


def extract_final_answer(raw_response: str, valid_options: List[str]) -> Optional[str]:
    text = (raw_response or "").strip()
    if len(text) <= 5:
        for char in text.upper():
            if char in valid_options:
                return char

    opts = "|".join(valid_options)
    patterns = [
        (rf"\\boxed\{{({opts})\}}", re.MULTILINE),
        (
            rf"(?:the\s+)?(?:correct|final|best)\s+(?:answer|option)\s+is[:\s]*\(?({opts})\)?",
            re.IGNORECASE | re.MULTILINE,
        ),
        (
            rf"(?:^|\n)\s*answer[:\s]+\(?({opts})\)?",
            re.IGNORECASE | re.MULTILINE,
        ),
        (
            rf"(?:i\s+)?(?:choose|select|pick)\s+\(?({opts})\)?",
            re.IGNORECASE | re.MULTILINE,
        ),
        (
            rf"option\s+\(?({opts})\)?\s+is\s+(?:correct|right|the\s+answer)",
            re.IGNORECASE | re.MULTILINE,
        ),
        (rf"(?:^|\n)\s*\(?({opts})\)?\s*\.?\s*$", re.MULTILINE),
        (
            rf"(?:[Tt]herefore|[Tt]hus|[Hh]ence|[Ss]o)[,:\s]+\(?({opts})\)?(?!\w)",
            re.MULTILINE,
        ),
        (
            rf"my\s+(?:answer|choice)\s+is[:\s]*\(?({opts})\)?",
            re.IGNORECASE | re.MULTILINE,
        ),
    ]
    for pattern, flags in patterns:
        match = re.search(pattern, text, flags)
        if match and match.group(1).upper() in valid_options:
            return match.group(1).upper()
    return None


def extract_answer_with_fallback(
    raw_response: str,
    valid_options: List[str],
    call_extraction_llm: Optional[Callable[[str], str]] = None,
) -> Tuple[str, str]:
    answer = extract_final_answer(raw_response, valid_options)
    if answer is not None:
        return answer, "regex"

    if call_extraction_llm is not None:
        try:
            extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
                raw_response=raw_response,
                valid_options=", ".join(valid_options),
            )
            extracted = call_extraction_llm(extraction_prompt)
            answer = extract_final_answer(extracted, valid_options)
            if answer is not None:
                return answer, "llm_fallback"
        except Exception:
            pass

    for char in (raw_response or "").upper():
        if char in valid_options:
            return char, "brute_force"
    return valid_options[0], "default"


def compute_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0, "wrong": 0}

    total = len(results)
    correct = sum(1 for result in results if result["is_correct"])
    answer_breakdown: Dict[str, Dict[str, int]] = {}
    pred_distribution: Dict[str, int] = {}
    video_stats: Dict[str, Dict[str, int]] = {}

    total_prompt_tokens = 0
    total_completion_tokens = 0
    for result in results:
        gt = result["correct_answer"]
        pred_distribution[result["predicted_answer"]] = pred_distribution.get(
            result["predicted_answer"], 0
        ) + 1
        answer_breakdown.setdefault(gt, {"total": 0, "correct": 0})
        answer_breakdown[gt]["total"] += 1
        if result["is_correct"]:
            answer_breakdown[gt]["correct"] += 1

        video_id = result["video"]
        video_stats.setdefault(video_id, {"total": 0, "correct": 0})
        video_stats[video_id]["total"] += 1
        if result["is_correct"]:
            video_stats[video_id]["correct"] += 1

        total_prompt_tokens += int(result.get("prompt_tokens") or 0)
        total_completion_tokens += int(result.get("completion_tokens") or 0)

    per_answer_accuracy = {
        key: {
            "total": info["total"],
            "correct": info["correct"],
            "accuracy": (info["correct"] / info["total"]) if info["total"] else 0.0,
        }
        for key, info in sorted(answer_breakdown.items())
    }

    videos_perfect = sum(
        1 for value in video_stats.values() if value["correct"] == value["total"]
    )
    videos_zero = sum(
        1 for value in video_stats.values() if value["correct"] == 0
    )
    return {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "per_answer_accuracy": per_answer_accuracy,
        "predicted_distribution": dict(sorted(pred_distribution.items())),
        "token_usage": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "video_stats": {
            "total_videos": len(video_stats),
            "videos_all_correct": videos_perfect,
            "videos_all_wrong": videos_zero,
        },
    }


def save_results_file(
    path: str,
    results: List[Dict],
    total_questions: int,
    model_name: str,
    mode: str,
    backend: str,
) -> Dict:
    metrics = compute_metrics(results)
    questions_failed = total_questions - len(results)
    payload = {
        "metadata": {
            "model": model_name,
            "mode": mode,
            "backend": backend,
            "last_updated": datetime.now().isoformat(),
            "total_questions": total_questions,
            "questions_evaluated": len(results),
            "questions_failed": questions_failed,
            "complete": questions_failed == 0,
        },
        "metrics": metrics,
        "results": results,
    }
    save_json_atomic(payload, path)
    return metrics


def print_evaluation_report(metrics: Dict, model_name: str, mode: str, backend: str) -> None:
    print("\n" + "=" * 64)
    print(f"  ColonBench VQA Report - {model_name}")
    print("=" * 64)
    print(f"  Mode: {mode}")
    print(f"  Backend: {backend}")
    print(f"  Accuracy: {metrics.get('accuracy', 0.0):.4f} ({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
    tokens = metrics.get("token_usage", {})
    if tokens:
        print(f"  Prompt tokens: {tokens.get('total_prompt_tokens', 0):,}")
        print(f"  Completion tokens: {tokens.get('total_completion_tokens', 0):,}")
        print(f"  Total tokens: {tokens.get('total_tokens', 0):,}")
    print("=" * 64)


def resolve_benchmark_paths(
    benchmark: str,
    mode: str,
    videos: Optional[str],
) -> Tuple[str, Optional[str], str]:
    if mode == "prompted":
        candidates = [PROMPTED_BENCHMARK_FILENAME]
    else:
        candidates = [UNPROMPTED_BENCHMARK_FILENAME]

    if os.path.isdir(benchmark):
        benchmark_dir = benchmark
        benchmark_json_path = None
        for candidate in candidates:
            test_path = os.path.join(benchmark_dir, candidate)
            if os.path.isfile(test_path):
                benchmark_json_path = test_path
                break
        if benchmark_json_path is None:
            raise FileNotFoundError(
                f"Could not find a {mode} benchmark JSON in {benchmark_dir}"
            )
        videos_dir = videos or os.path.join(benchmark_dir, VIDEOS_SUBDIR)
    elif os.path.isfile(benchmark):
        benchmark_json_path = benchmark
        benchmark_dir = os.path.dirname(benchmark_json_path)
        videos_dir = videos or os.path.join(benchmark_dir, VIDEOS_SUBDIR)
    else:
        raise FileNotFoundError(f"Benchmark path not found: {benchmark}")

    if videos_dir and not os.path.isdir(videos_dir):
        videos_dir = None
    return benchmark_json_path, videos_dir, benchmark_dir


def resolve_local_video_path(
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


def evaluate_single_question_openrouter(
    *,
    question: Dict,
    model: str,
    frame_cache: VideoFrameCache,
    asset_resolver: HFDatasetAssetResolver,
    videos_dir: Optional[str],
    extraction_model: str,
    skill_text: Optional[str],
    use_video: bool,
    max_retries: int,
    verbose: bool,
) -> Optional[Dict]:
    question_id = question.get("question_id", "unknown")
    video_name = question.get("video", "")
    question_text = question.get("question", "")
    choices = question.get("choices", {})
    correct_answer = question.get("answer", "")
    valid_options = sorted(choices.keys())

    if not question_text or not choices or not video_name:
        return None

    prompt_text = build_vqa_prompt(question_text, choices, skill_text=skill_text)
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
                    f"  [WARN] Q={question_id}: could not resolve remote video URL ({exc}); "
                    "falling back to local frame extraction"
                )
            effective_use_video = False

    if not effective_use_video:
        try:
            local_video_path = resolve_local_video_path(
                video_id=video_name,
                videos_dir=videos_dir,
                asset_resolver=asset_resolver,
            )
            frames_b64 = frame_cache.get_or_extract(local_video_path)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [ERROR] Q={question_id}: could not prepare local video ({exc})")
            return None

    raw_response = ""
    completion = None
    last_error = None
    used_runtime_frame_fallback = False

    def _call_openrouter() -> Tuple[object, str]:
        return run_openrouter_completion(
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

    for attempt in range(max_retries):
        try:
            completion, raw_response = _call_openrouter()
            if raw_response:
                break
            last_error = "Empty response"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if effective_use_video:
                used_runtime_frame_fallback = True
                effective_use_video = False
                remote_video_url = None
                try:
                    local_video_path = resolve_local_video_path(
                        video_id=video_name,
                        videos_dir=videos_dir,
                        asset_resolver=asset_resolver,
                    )
                    frames_b64 = frame_cache.get_or_extract(local_video_path)
                except Exception as fallback_exc:  # noqa: BLE001
                    last_error = f"{exc}; frame fallback failed: {fallback_exc}"
                    break
                try:
                    completion, raw_response = _call_openrouter()
                    if raw_response:
                        break
                    last_error = "Empty response after frame fallback"
                except Exception as fallback_exc:  # noqa: BLE001
                    last_error = str(fallback_exc)

        if attempt < max_retries - 1 and not raw_response:
            time.sleep(2 * (attempt + 1))

    if not raw_response:
        if verbose:
            print(f"  [FAIL] Q={question_id}: {last_error}")
        return None

    def call_extraction_llm(prompt: str) -> str:
        _, text = run_openrouter_text_completion(extraction_model, prompt)
        return text

    predicted_answer, extraction_tier = extract_answer_with_fallback(
        raw_response,
        valid_options,
        call_extraction_llm=call_extraction_llm,
    )
    is_correct = predicted_answer == correct_answer

    prompt_tokens = 0
    completion_tokens = 0
    if completion is not None and hasattr(completion, "usage"):
        prompt_tokens = int(getattr(completion.usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(completion.usage, "completion_tokens", 0) or 0)

    if verbose:
        status = "CORRECT" if is_correct else "WRONG"
        fallback_note = " [frame-fallback]" if used_runtime_frame_fallback else ""
        print(
            f"  [{status}] Q={question_id} pred={predicted_answer} gt={correct_answer}{fallback_note}"
        )

    return {
        "question_id": question_id,
        "video": video_name,
        "question": question_text,
        "choices": choices,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "raw_response": raw_response,
        "extraction_tier": extraction_tier,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def evaluate_single_question_local(
    *,
    question: Dict,
    evaluator: LocalQwenVideoVLM,
    asset_resolver: HFDatasetAssetResolver,
    videos_dir: Optional[str],
    skill_text: Optional[str],
    verbose: bool,
) -> Optional[Dict]:
    question_id = question.get("question_id", "unknown")
    video_name = question.get("video", "")
    question_text = question.get("question", "")
    choices = question.get("choices", {})
    correct_answer = question.get("answer", "")
    valid_options = sorted(choices.keys())

    if not question_text or not choices or not video_name:
        return None

    try:
        video_path = resolve_local_video_path(
            video_id=video_name,
            videos_dir=videos_dir,
            asset_resolver=asset_resolver,
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [ERROR] Q={question_id}: could not download local video ({exc})")
        return None

    prompt_text = build_vqa_prompt(question_text, choices, skill_text=skill_text)
    try:
        raw_response, usage = evaluator.answer(video_path=video_path, prompt_text=prompt_text)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [ERROR] Q={question_id}: local model failed ({exc})")
        return None

    predicted_answer, extraction_tier = extract_answer_with_fallback(
        raw_response,
        valid_options,
        call_extraction_llm=None,
    )
    is_correct = predicted_answer == correct_answer
    if verbose:
        status = "CORRECT" if is_correct else "WRONG"
        print(f"  [{status}] Q={question_id} pred={predicted_answer} gt={correct_answer}")
    return {
        "question_id": question_id,
        "video": video_name,
        "question": question_text,
        "choices": choices,
        "correct_answer": correct_answer,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "raw_response": raw_response,
        "extraction_tier": extraction_tier,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
    }


def run_evaluation(
    *,
    benchmark: List[Dict],
    mode: str,
    output_dir: str,
    videos_dir: Optional[str],
    asset_resolver: HFDatasetAssetResolver,
    skill_text: Optional[str],
    parallel: bool,
    max_workers: int,
    max_retries: int,
    new_run: bool,
    local: bool,
    model: str,
    local_model_id: str,
    num_frames: int,
    use_frames: bool,
    local_nframes: int,
    local_max_new_tokens: int,
    verbose: bool,
) -> Tuple[List[Dict], Dict, str]:
    ensure_dir(output_dir)
    total_questions = len(benchmark)
    result_model_name = local_model_id if local else model
    model_safe = sanitize_model_name(result_model_name)
    results_prefix = f"llm_eval_{model_safe}_{mode}_"

    existing_results: Dict[str, Dict] = {}
    results_path: Optional[str] = None
    if not new_run:
        prev_path = find_previous_results(output_dir, results_prefix)
        if prev_path:
            prev_data = load_json(prev_path)
            for result in prev_data.get("results", []):
                question_id = result.get("question_id")
                if question_id:
                    existing_results[question_id] = result
            results_path = prev_path
            print(f"  Resuming from: {prev_path}")
            print(f"  Previously completed: {len(existing_results)}/{total_questions}")

    if results_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"{results_prefix}{timestamp}.json")
        if not new_run:
            print("  No previous results found - starting fresh.")

    to_process: List[Dict] = []
    results: List[Dict] = []
    for question in benchmark:
        question_id = question.get("question_id")
        if question_id in existing_results:
            results.append(existing_results[question_id])
        else:
            to_process.append(question)

    if not to_process:
        metrics = save_results_file(
            results_path,
            results,
            total_questions,
            result_model_name,
            mode,
            "local" if local else "openrouter",
        )
        return results, metrics, results_path

    if local and parallel and len(to_process) > 1:
        print("  [WARN] Local evaluation runs sequentially; disabling --parallel.")
        parallel = False

    evaluator: Optional[LocalQwenVideoVLM] = None
    frame_cache: Optional[VideoFrameCache] = None
    send_video = False
    if local:
        evaluator = LocalQwenVideoVLM(
            model_id=local_model_id,
            nframes=local_nframes,
            max_new_tokens=local_max_new_tokens,
            verbose=verbose,
        )
    else:
        frame_cache = VideoFrameCache(num_frames=num_frames)
        _, supports_video = resolve_openrouter_model(model)
        send_video = supports_video and not use_frames
        if send_video:
            print("  API mode: native remote video_url")
        else:
            print(f"  API mode: frame extraction ({num_frames} frames)")

    def process_question(question: Dict) -> Optional[Dict]:
        if local:
            return evaluate_single_question_local(
                question=question,
                evaluator=evaluator,  # type: ignore[arg-type]
                asset_resolver=asset_resolver,
                videos_dir=videos_dir,
                skill_text=skill_text,
                verbose=verbose,
            )
        return evaluate_single_question_openrouter(
            question=question,
            model=model,
            frame_cache=frame_cache,  # type: ignore[arg-type]
            asset_resolver=asset_resolver,
            videos_dir=videos_dir,
            extraction_model=DEFAULT_EXTRACTION_MODEL,
            skill_text=skill_text,
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
                    executor.submit(process_question, question): question
                    for question in to_process
                }
                for future in as_completed(futures):
                    completed_this_run += 1
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        failed_this_run += 1
                    if (completed_this_run % 10 == 0) or (completed_this_run == len(to_process)):
                        metrics = save_results_file(
                            results_path,
                            results,
                            total_questions,
                            result_model_name,
                            mode,
                            "local" if local else "openrouter",
                        )
        else:
            for question in to_process:
                completed_this_run += 1
                result = process_question(question)
                if result is not None:
                    results.append(result)
                else:
                    failed_this_run += 1
                if (completed_this_run % 10 == 0) or (completed_this_run == len(to_process)):
                    metrics = save_results_file(
                        results_path,
                        results,
                        total_questions,
                        result_model_name,
                        mode,
                        "local" if local else "openrouter",
                    )
    finally:
        try:
            save_results_file(
                results_path,
                results,
                total_questions,
                result_model_name,
                mode,
                "local" if local else "openrouter",
            )
        except Exception:
            pass

    missing = total_questions - len(results)
    if missing == 0:
        print(f"\n  COMPLETE - all {total_questions} questions evaluated (fail=0)")
    else:
        print(f"\n  Done: {len(results)}/{total_questions} questions ({missing} missing). Re-run to retry.")
    if failed_this_run:
        print(f"  Failed in this run: {failed_this_run}")

    metrics = save_results_file(
        results_path,
        results,
        total_questions,
        result_model_name,
        mode,
        "local" if local else "openrouter",
    )
    return results, metrics, results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ColonBench VQA evaluation.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/colon-bench",
        help="Benchmark directory or benchmark JSON path.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Optional local videos directory. If omitted, videos are resolved from Hugging Face.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prompted",
        choices=["prompted", "unprompted"],
        help="Which VQA split to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-vl-8b",
        help=f"OpenRouter model name (choices include: {', '.join(list_openrouter_models()[:8])}, ...)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run local Hugging Face evaluation instead of API evaluation.",
    )
    parser.add_argument(
        "--local-model-id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Local Hugging Face Qwen3-VL checkpoint to load when --local is set.",
    )
    parser.add_argument(
        "--skill-file",
        type=str,
        default=None,
        help="Optional plain-text skill/context appended to every question prompt.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=DEFAULT_DATASET_REPO_ID,
        help="Hugging Face dataset repo used to resolve videos and masks.",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=DEFAULT_DATASET_REVISION,
        help="Hugging Face dataset revision.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token override. Otherwise uses HF_TOKEN / HUGGINGFACE_API_KEY / local login.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for result JSONs (default: <benchmark_dir>/eval_results).",
    )
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per question.")
    parser.add_argument("--new", action="store_true", help="Ignore previous results and start fresh.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    parser.add_argument("--num-frames", type=int, default=16, help="Frames to sample in API frame mode.")
    parser.add_argument("--use-frames", action="store_true", help="Force frame mode even for video-capable models.")
    parser.add_argument("--local-nframes", type=int, default=32, help="Frames to feed into the local Qwen model.")
    parser.add_argument(
        "--local-max-new-tokens",
        type=int,
        default=32,
        help="Generation budget for the local Qwen model.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_json_path, videos_dir, benchmark_dir = resolve_benchmark_paths(
        args.benchmark,
        args.mode,
        args.videos,
    )
    output_dir = args.output or os.path.join(benchmark_dir, "eval_results")
    ensure_dir(output_dir)

    benchmark = load_json(benchmark_json_path)
    if args.limit is not None:
        benchmark = benchmark[: args.limit]

    skill_text: Optional[str] = None
    skill_file_path: Optional[str] = None
    if args.skill_file:
        skill_candidate = Path(args.skill_file)
        if not skill_candidate.is_absolute():
            skill_candidate = project_root_from_script(__file__) / skill_candidate
        skill_file_path = str(skill_candidate.resolve())
        if not os.path.isfile(skill_file_path):
            raise FileNotFoundError(f"Skill file not found: {skill_file_path}")
        with open(skill_file_path, "r", encoding="utf-8") as handle:
            skill_text = handle.read().strip()

    asset_resolver = HFDatasetAssetResolver(
        repo_id=args.dataset_repo_id,
        revision=args.dataset_revision,
        token=args.hf_token,
    )

    backend_label = "local" if args.local else "openrouter"
    print("\n" + "=" * 72)
    print("  ColonBench VQA Evaluation")
    print("=" * 72)
    print(f"  Benchmark:         {benchmark_json_path}")
    print(f"  Mode:              {args.mode}")
    print(f"  Backend:           {backend_label}")
    print(f"  Model:             {args.local_model_id if args.local else args.model}")
    print(f"  Dataset repo:      {args.dataset_repo_id}@{args.dataset_revision}")
    print(f"  Local videos dir:  {videos_dir or '(not provided - resolve/download from HF)'}")
    print(f"  Output dir:        {output_dir}")
    print(f"  Questions:         {len(benchmark)}")
    print(f"  Parallel:          {args.parallel} (workers: {args.max_workers})")
    if not args.local:
        resolved_id, supports_video = resolve_openrouter_model(args.model)
        video_mode = "remote video_url" if (supports_video and not args.use_frames) else f"frames ({args.num_frames})"
        print(f"  OpenRouter ID:     {resolved_id}")
        print(f"  Video input:       {video_mode}")
    if skill_file_path:
        print(f"  Skill file:        {skill_file_path}")
    print("=" * 72)

    results, metrics, results_path = run_evaluation(
        benchmark=benchmark,
        mode=args.mode,
        output_dir=output_dir,
        videos_dir=videos_dir,
        asset_resolver=asset_resolver,
        skill_text=skill_text,
        parallel=args.parallel,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        new_run=args.new,
        local=args.local,
        model=args.model,
        local_model_id=args.local_model_id,
        num_frames=args.num_frames,
        use_frames=args.use_frames,
        local_nframes=args.local_nframes,
        local_max_new_tokens=args.local_max_new_tokens,
        verbose=args.verbose,
    )

    _ = results
    print_evaluation_report(
        metrics,
        args.local_model_id if args.local else args.model,
        args.mode,
        backend_label,
    )
    print(f"\n  Results JSON: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
