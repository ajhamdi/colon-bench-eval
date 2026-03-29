#!/usr/bin/env python3
"""
LLM evaluation for detection benchmark and segmentation evaluation.

Without --seg: evaluates model-agnostic LLM localization quality on
`benchmark_detection.json`.

With --seg: runs EdgeTAM segmentation only, loading existing detection
results from the output directory. Exits with an error if no detection
results are found. Run detection first (without --seg), then segmentation
(with --seg).

Detection output filename pattern:
  llm_eval_{model_safe}_det_{timestamp}.json

Segmentation output filename pattern:
  llm_eval_{model_safe}_seg_{timestamp}.json
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

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

from colonbench_eval.hf_video import (  # pyright: ignore[reportMissingImports]
    DEFAULT_DATASET_REPO_ID,
    DEFAULT_DATASET_REVISION,
    HFDatasetAssetResolver,
)
from colonbench_eval.openrouter_utils import (  # pyright: ignore[reportMissingImports]
    list_openrouter_models,
    resolve_openrouter_model,
    run_openrouter_completion,
)


DETECTION_BENCHMARK_FILENAME = "benchmark_detection.json"
SEGMENTATION_BENCHMARK_FILENAME = "benchmark_segmentation.json"
VIDEOS_SUBDIR = "videos"
DEFAULT_OUTPUT_SUBDIR = "det_results"
DEFAULT_MODEL = "qwen3-vl-8b"


def atomic_json_save(data: Any, path: str) -> None:
    """Write JSON atomically via temp file + rename."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=dir_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_json_safe(path: str) -> Optional[Any]:
    """Load JSON and return None for missing/invalid files."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def find_previous_results(output_dir: str, prefix: str) -> Optional[str]:
    """Find latest JSON result matching prefix in output_dir."""
    if not os.path.isdir(output_dir):
        return None
    pattern = os.path.join(output_dir, f"{prefix}*.json")
    candidates = [p for p in glob.glob(pattern) if not p.endswith("_progress.json")]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two bboxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 >= x2 or y1 >= y2:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def greedy_match(
    pred_boxes: List[List[int]],
    gt_boxes: List[List[int]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy one-to-one matching by IoU (descending)."""
    if not pred_boxes or not gt_boxes:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    pairs: List[Tuple[int, int, float]] = []
    for p_i, p_box in enumerate(pred_boxes):
        for g_i, g_box in enumerate(gt_boxes):
            iou = compute_iou(p_box, g_box)
            if iou >= iou_threshold:
                pairs.append((p_i, g_i, iou))

    pairs.sort(key=lambda x: -x[2])
    matched_preds = set()
    matched_gts = set()
    matches: List[Tuple[int, int, float]] = []
    for p_i, g_i, iou in pairs:
        if p_i in matched_preds or g_i in matched_gts:
            continue
        matched_preds.add(p_i)
        matched_gts.add(g_i)
        matches.append((p_i, g_i, iou))

    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts]
    return matches, unmatched_preds, unmatched_gts


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _build_pred_entries(
    pred_by_frame: Dict[Hashable, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for frame_key, boxes in pred_by_frame.items():
        for box_info in boxes:
            bbox = box_info.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            conf = float(box_info.get("confidence", 1.0))
            entries.append(
                {
                    "frame_key": frame_key,
                    "bbox": [int(v) for v in bbox],
                    "confidence": max(0.0, min(conf, 1.0)),
                }
            )
    return entries


def compute_ap(
    pred_entries: List[Dict[str, Any]],
    gt_by_frame: Dict[Hashable, List[List[int]]],
    iou_threshold: float,
) -> float:
    """Compute AP for a single IoU threshold using confidence-ranked predictions."""
    total_gt = sum(len(v) for v in gt_by_frame.values())
    if total_gt <= 0:
        return 0.0
    if not pred_entries:
        return 0.0

    sorted_preds = sorted(pred_entries, key=lambda x: x["confidence"], reverse=True)
    gt_matched: Dict[Hashable, List[bool]] = {
        k: [False] * len(v) for k, v in gt_by_frame.items()
    }

    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for pred in sorted_preds:
        frame_key = pred["frame_key"]
        pred_box = pred["bbox"]
        gt_boxes = gt_by_frame.get(frame_key, [])

        best_iou = -1.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[frame_key][gt_idx]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_matched[frame_key][best_gt_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    cum_tp: List[int] = []
    cum_fp: List[int] = []
    running_tp = 0
    running_fp = 0
    for i in range(len(tp_flags)):
        running_tp += tp_flags[i]
        running_fp += fp_flags[i]
        cum_tp.append(running_tp)
        cum_fp.append(running_fp)

    recalls: List[float] = []
    precisions: List[float] = []
    for i in range(len(cum_tp)):
        recall = cum_tp[i] / total_gt
        precision = cum_tp[i] / (cum_tp[i] + cum_fp[i]) if (cum_tp[i] + cum_fp[i]) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)

    # Pascal-style interpolation.
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def compute_detection_metrics(
    pred_by_frame: Dict[Hashable, List[Dict[str, Any]]],
    gt_by_frame: Dict[Hashable, List[List[int]]],
    iou_thresholds: List[float] | None = None,
) -> Dict[str, Any]:
    """Compute detection metrics including AP and IoU summaries."""
    if iou_thresholds is None:
        iou_thresholds = [0.25, 0.5, 0.75]

    per_threshold: Dict[str, Any] = {}
    ious_at_05: List[float] = []

    all_frame_keys = set(gt_by_frame.keys()) | set(pred_by_frame.keys())
    for thr in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0
        thr_ious: List[float] = []

        for frame_key in all_frame_keys:
            pred_boxes = [p["bbox"] for p in pred_by_frame.get(frame_key, []) if "bbox" in p]
            gt_boxes = gt_by_frame.get(frame_key, [])
            matches, unmatched_preds, unmatched_gts = greedy_match(pred_boxes, gt_boxes, iou_threshold=thr)
            tp += len(matches)
            fp += len(unmatched_preds)
            fn += len(unmatched_gts)
            thr_ious.extend([m[2] for m in matches])

        precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
        avg_iou = sum(thr_ious) / len(thr_ious) if thr_ious else 0.0
        per_threshold[f"iou_{thr:.2f}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "avg_matched_iou": round(avg_iou, 4),
            "num_matched": len(thr_ious),
        }
        if abs(thr - 0.5) < 1e-9:
            ious_at_05 = thr_ious

    pred_entries = _build_pred_entries(pred_by_frame)
    ap50 = compute_ap(pred_entries, gt_by_frame, iou_threshold=0.5)

    coco_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap_values = [compute_ap(pred_entries, gt_by_frame, iou_threshold=t) for t in coco_thresholds]
    map_50_95 = sum(ap_values) / len(ap_values) if ap_values else 0.0
    mean_iou = sum(ious_at_05) / len(ious_at_05) if ious_at_05 else 0.0

    all_gt = sum(len(v) for v in gt_by_frame.values())
    all_pred = sum(len(v) for v in pred_by_frame.values())
    p50 = per_threshold.get("iou_0.50", {})
    return {
        "num_frames": len(all_frame_keys),
        "num_gt_boxes": all_gt,
        "num_pred_boxes": all_pred,
        "precision": float(p50.get("precision", 0.0)),
        "recall": float(p50.get("recall", 0.0)),
        "f1": float(p50.get("f1", 0.0)),
        "mean_iou": round(mean_iou, 4),
        "ap50": round(ap50, 4),
        "map50_95": round(map_50_95, 4),
        "ap_by_threshold": {f"{t:.2f}": round(v, 4) for t, v in zip(coco_thresholds, ap_values)},
        "per_threshold": per_threshold,
    }


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, gt).sum()
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    inter = np.logical_and(pred, gt).sum()
    return float((2.0 * inter) / denom)


def read_gt_mask(mask_path: str) -> np.ndarray:
    with Image.open(mask_path) as img:
        arr = np.array(img.convert("L"))
    return arr > 127


def encode_frame_to_base64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    return base64.b64encode(buf).decode("utf-8")


def get_video_frames_at_indices(
    video_path: str, indices: List[int]
) -> List[Dict[str, Any]]:
    """Read specific frame indices from a video and return b64 + geometry."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[Dict[str, Any]] = []
    for target_idx in indices:
        if target_idx < 0 or (total_frames > 0 and target_idx >= total_frames):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        frames.append(
            {
                "frame_index": target_idx,
                "width": int(w),
                "height": int(h),
                "frame_b64": encode_frame_to_base64(frame),
            }
        )

    cap.release()
    return frames


def compute_tracking_window_sample_indices(
    record: Dict[str, Any],
    frames_count: int,
) -> List[int]:
    """
    Compute equally-spaced frame indices within the tracking window.

    The tracking window is defined by the ``bbox_track`` entries in the
    benchmark record.  Sampling starts at the *first* tracking-window
    frame (``min(frame_index)``) and extends to the *last* tracking-window
    frame (``max(frame_index)``).  ``frames_count`` indices are placed at
    equal intervals across that range.

    Falls back to ``first_localization_frame`` if ``bbox_track`` is empty.
    """
    bbox_track = record.get("bbox_track", [])
    track_frame_indices = sorted(
        {
            int(e["frame_index"])
            for e in bbox_track
            if isinstance(e, dict) and isinstance(e.get("frame_index"), int)
        }
    )

    if track_frame_indices:
        win_start = track_frame_indices[0]
        win_end = track_frame_indices[-1]
    else:
        # Fallback: use first_localization_frame as a single-frame window.
        first_loc = record.get("first_localization_frame")
        if first_loc is not None:
            win_start = int(first_loc)
            win_end = win_start
        else:
            return list(range(frames_count))  # ultimate fallback

    if frames_count <= 1:
        return [win_start]

    span = win_end - win_start
    if span <= 0:
        return [win_start]

    # Equally-spaced indices covering [win_start, win_end]
    indices: List[int] = []
    for i in range(frames_count):
        idx = win_start + int(round(i * span / (frames_count - 1)))
        idx = clamp(idx, win_start, win_end)
        if not indices or idx != indices[-1]:
            indices.append(idx)
    return indices


def decode_video_to_frame_dir(video_path: str, output_dir: str) -> int:
    """Decode video to sequential JPG frames in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = os.path.join(output_dir, f"{idx:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        idx += 1
    cap.release()
    return idx


def build_detection_prompt(description: str, frame_index: int) -> str:
    return (
        "You are an expert colorectal surgeon analyzing a colonoscopy frame.\n"
        f"Known lesion description from benchmark: {description}\n"
        f"Current frame index: {frame_index}\n\n"
        "Task: detect any visible lesion and output bounding boxes.\n"
        "Preferred output format (one line per lesion):\n"
        "FOUND | x_min,y_min,x_max,y_max | confidence(0-1) | short description\n"
        "If no lesion is visible:\n"
        "NONE | no lesion visible\n"
        "Coordinates should be in image pixel coordinates."
    )


def _parse_bbox_numbers(text: str) -> Optional[List[float]]:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) < 4:
        return None
    return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]


def _parse_confidence(text: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    value = float(m.group(1))
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return max(0.0, min(value, 1.0))


def _convert_bbox_to_pixels(
    raw_bbox: List[float],
    width: int,
    height: int,
    model_is_gemini: bool,
) -> List[int]:
    x1, y1, x2, y2 = raw_bbox
    if model_is_gemini and max(raw_bbox) <= 1000.0:
        x1 = x1 * width / 1000.0
        y1 = y1 * height / 1000.0
        x2 = x2 * width / 1000.0
        y2 = y2 * height / 1000.0

    x1_i = clamp(int(round(x1)), 0, width)
    y1_i = clamp(int(round(y1)), 0, height)
    x2_i = clamp(int(round(x2)), 0, width)
    y2_i = clamp(int(round(y2)), 0, height)

    if x2_i < x1_i:
        x1_i, x2_i = x2_i, x1_i
    if y2_i < y1_i:
        y1_i, y2_i = y2_i, y1_i
    return [x1_i, y1_i, x2_i, y2_i]


def parse_detection_response(
    raw_text: str,
    width: int,
    height: int,
    model_is_gemini: bool,
) -> List[Dict[str, Any]]:
    """Parse detection text into standardized boxes."""
    text = (raw_text or "").strip()
    if not text:
        return []
    if text.upper().startswith("NONE"):
        return []

    out: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Format: FOUND | x1,y1,x2,y2 | conf | desc
    for line in lines:
        if not line.upper().startswith("FOUND"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        raw_bbox = _parse_bbox_numbers(parts[1])
        if raw_bbox is None:
            continue
        bbox = _convert_bbox_to_pixels(raw_bbox, width, height, model_is_gemini)
        conf = 1.0
        desc = ""
        if len(parts) >= 3:
            maybe_conf = _parse_confidence(parts[2])
            if maybe_conf is not None:
                conf = maybe_conf
            else:
                desc = parts[2]
        if len(parts) >= 4:
            desc = parts[3]
        out.append({"bbox": bbox, "confidence": conf, "description": desc})

    if out:
        return out

    # Fallback: parse JSON-like bboxes.
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            payload = json.loads(text[start : end + 1])
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        raw_bbox = item.get("bbox_2d") or item.get("bbox")
                        if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                            bbox = _convert_bbox_to_pixels(
                                [float(v) for v in raw_bbox],
                                width,
                                height,
                                model_is_gemini,
                            )
                            conf = float(item.get("confidence", 1.0))
                            conf = max(0.0, min(conf, 1.0))
                            out.append(
                                {
                                    "bbox": bbox,
                                    "confidence": conf,
                                    "description": str(item.get("description", "")),
                                }
                            )
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

    if out:
        return out

    # Last fallback: first 4 numbers in the whole response.
    raw_bbox = _parse_bbox_numbers(text)
    if raw_bbox is not None:
        bbox = _convert_bbox_to_pixels(raw_bbox, width, height, model_is_gemini)
        out.append({"bbox": bbox, "confidence": 1.0, "description": ""})
    return out


def call_model_for_frame(
    *,
    model: str,
    prompt_text: str,
    frame_b64: str,
    max_retries: int = 3,
) -> Tuple[str, int, int]:
    """
    Run one model call on one frame and return raw text + token usage.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            completion, response_text = run_openrouter_completion(
                model,
                prompt_text,
                frames_b64=[frame_b64],
                use_video=False,
                max_retries=max_retries,
            )

            prompt_tokens = 0
            completion_tokens = 0
            if completion is not None and hasattr(completion, "usage"):
                prompt_tokens = int(getattr(completion.usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(completion.usage, "completion_tokens", 0) or 0)
            return response_text or "", prompt_tokens, completion_tokens
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Model call failed after retries: {last_error}")


def merge_bboxes_to_encompassing(bboxes: List[List[int]]) -> Optional[List[int]]:
    if not bboxes:
        return None
    if len(bboxes) == 1:
        return bboxes[0]
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


def deduplicate_prompts_by_frame(
    pred_by_frame: Dict[int, List[Dict[str, Any]]]
) -> List[Tuple[int, List[int]]]:
    prompts: List[Tuple[int, List[int]]] = []
    for frame_idx in sorted(pred_by_frame.keys()):
        frame_boxes = [p.get("bbox") for p in pred_by_frame[frame_idx] if isinstance(p.get("bbox"), list)]
        frame_boxes = [b for b in frame_boxes if len(b) == 4]
        merged = merge_bboxes_to_encompassing(frame_boxes)
        if merged is not None:
            prompts.append((frame_idx, merged))
    return prompts


def save_mask(mask: np.ndarray, output_path: str) -> None:
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_u8 = mask.astype(np.uint8) * 255
    Image.fromarray(mask_u8, mode="L").save(output_path)


def init_edgetam_predictor(model_cfg: str, checkpoint: str):
    """Lazily initialize EdgeTAM predictor."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor  # pyright: ignore[reportMissingImports]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  [EdgeTAM] device={device}")
    return build_sam2_video_predictor(model_cfg, checkpoint, device=device)


def run_edgetam_masks_single_pass(
    predictor: Any,
    video_path: str,
    pred_by_frame: Dict[int, List[Dict[str, Any]]],
    output_mask_dir: str,
    max_prompts: int,
    verbose: bool = False,
) -> Dict[int, str]:
    """
    Run one EdgeTAM pass for one video using detection predictions as prompts.
    Returns mapping frame_idx -> saved predicted mask path.
    """
    prompts = deduplicate_prompts_by_frame(pred_by_frame)
    if not prompts:
        return {}

    if len(prompts) > max_prompts:
        sampled: List[Tuple[int, List[int]]] = []
        for i in range(max_prompts):
            idx = int(i * (len(prompts) - 1) / max(1, max_prompts - 1))
            sampled.append(prompts[idx])
        prompts = sampled
        if verbose:
            print(f"  [EdgeTAM] Prompt count capped to {max_prompts}")

    with tempfile.TemporaryDirectory() as td:
        frame_dir = os.path.join(td, "frames")
        total_frames = decode_video_to_frame_dir(video_path, frame_dir)
        if total_frames <= 0:
            return {}

        inference_state = predictor.init_state(video_path=frame_dir)
        predictor.reset_state(inference_state)

        ann_obj_id = 1
        prompts_added = 0
        for frame_idx, bbox in prompts:
            if frame_idx < 0 or frame_idx >= total_frames:
                continue
            try:
                box = np.array(bbox, dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=ann_obj_id,
                    box=box,
                )
                prompts_added += 1
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"  [EdgeTAM][WARN] Failed prompt frame={frame_idx}: {exc}")

        if prompts_added == 0:
            return {}

        os.makedirs(output_mask_dir, exist_ok=True)
        saved: Dict[int, str] = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            for i, out_obj_id in enumerate(out_obj_ids):
                if int(out_obj_id) != ann_obj_id:
                    continue
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                out_path = os.path.join(output_mask_dir, f"{out_frame_idx:06d}.png")
                save_mask(mask, out_path)
                saved[out_frame_idx] = out_path
        return saved


def build_frame_maps_from_detection_record(
    record: Dict[str, Any],
    frame_indices: List[int],
) -> Dict[int, List[List[int]]]:
    gt_by_frame: Dict[int, List[List[int]]] = {fi: [] for fi in frame_indices}
    for item in record.get("bbox_track", []):
        if not isinstance(item, dict):
            continue
        fi = item.get("frame_index")
        bbox = item.get("bbox")
        if isinstance(fi, int) and isinstance(bbox, list) and len(bbox) == 4 and fi in gt_by_frame:
            gt_by_frame[fi].append([int(v) for v in bbox])
    return gt_by_frame


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


def evaluate_single_video_detection(
    record: Dict[str, Any],
    videos_dir: Optional[str],
    asset_resolver: HFDatasetAssetResolver,
    model: str,
    frames_count: int,
    max_retries: int = 3,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    video_id = record.get("video_id")
    description = str(record.get("description", "")).strip()
    if not video_id or not description:
        return None

    try:
        video_path = resolve_local_video_path(
            video_id=video_id,
            videos_dir=videos_dir,
            asset_resolver=asset_resolver,
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [SKIP] Missing or undownloadable video {video_id}: {exc}")
        return None

    # Compute equally-spaced indices within the tracking window
    sample_indices = compute_tracking_window_sample_indices(record, frames_count)
    if verbose:
        print(f"  video={video_id}: sampling frames {sample_indices} from tracking window")

    try:
        frames = get_video_frames_at_indices(video_path, sample_indices)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [SKIP] Could not read frames for video={video_id}: {exc}")
        return None
    if not frames:
        if verbose:
            print(f"  [SKIP] No readable frames in video: {video_id}")
        return None

    frame_indices = [f["frame_index"] for f in frames]
    gt_by_frame = build_frame_maps_from_detection_record(record, frame_indices)
    pred_by_frame: Dict[int, List[Dict[str, Any]]] = {fi: [] for fi in frame_indices}
    frame_results: List[Dict[str, Any]] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    for f in frames:
        frame_idx = int(f["frame_index"])
        prompt = build_detection_prompt(description, frame_idx)
        try:
            raw_response, p_tok, c_tok = call_model_for_frame(
                model=model,
                prompt_text=prompt,
                frame_b64=f["frame_b64"],
                max_retries=max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  [WARN] model call failed video={video_id} frame={frame_idx}: {exc}")
            raw_response, p_tok, c_tok = "", 0, 0
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        preds = parse_detection_response(
            raw_text=raw_response,
            width=int(f["width"]),
            height=int(f["height"]),
            model_is_gemini=False,
        )
        pred_by_frame[frame_idx] = preds
        frame_results.append(
            {
                "frame_index": frame_idx,
                "frame_width": int(f["width"]),
                "frame_height": int(f["height"]),
                "ground_truth_boxes": gt_by_frame.get(frame_idx, []),
                "predicted_boxes": preds,
                "raw_response": raw_response,
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
            }
        )

    metrics = compute_detection_metrics(pred_by_frame, gt_by_frame)

    # Tracking window metadata for transparency
    bbox_track = record.get("bbox_track", [])
    track_fis = sorted(
        {int(e["frame_index"]) for e in bbox_track if isinstance(e, dict) and isinstance(e.get("frame_index"), int)}
    )
    tracking_window_start = track_fis[0] if track_fis else None
    tracking_window_end = track_fis[-1] if track_fis else None

    result = {
        "video_id": video_id,
        "description": description,
        "frames_count_requested": frames_count,
        "frames_evaluated": len(frame_indices),
        "first_localization_frame_gt": record.get("first_localization_frame"),
        "tracking_window_start": tracking_window_start,
        "tracking_window_end": tracking_window_end,
        "sample_indices_requested": sample_indices,
        "frame_indices_evaluated": frame_indices,
        "frame_results": frame_results,
        "metrics": metrics,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
    }
    return result


def aggregate_detection_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    pred_global: Dict[str, List[Dict[str, Any]]] = {}
    gt_global: Dict[str, List[List[int]]] = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for item in results:
        video_id = item.get("video_id", "unknown")
        total_prompt_tokens += int(item.get("token_usage", {}).get("prompt_tokens", 0) or 0)
        total_completion_tokens += int(item.get("token_usage", {}).get("completion_tokens", 0) or 0)
        for frame in item.get("frame_results", []):
            fi = frame.get("frame_index")
            key = f"{video_id}:{fi}"
            gt_global[key] = frame.get("ground_truth_boxes", [])
            pred_global[key] = frame.get("predicted_boxes", [])

    det_metrics = compute_detection_metrics(pred_global, gt_global)
    det_metrics["token_usage"] = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
    }
    det_metrics["videos_evaluated"] = len(results)
    return det_metrics


def save_detection_results_file(
    path: str,
    results: List[Dict[str, Any]],
    total_videos: int,
    model: str,
    frames_count: int,
    parallel: bool,
    max_workers: int,
    max_retries: int,
) -> Dict[str, Any]:
    metrics = aggregate_detection_metrics(results)
    videos_failed = total_videos - len(results)
    output = {
        "metadata": {
            "task": "detection",
            "model": model,
            "frames_count": frames_count,
            "parallel": parallel,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "last_updated": datetime.now().isoformat(),
            "total_videos": total_videos,
            "videos_evaluated": len(results),
            "videos_failed": videos_failed,
            "complete": videos_failed == 0,
            "metrics_summary": {
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "ap50": metrics.get("ap50", 0.0),
                "map50_95": metrics.get("map50_95", 0.0),
                "mean_iou": metrics.get("mean_iou", 0.0),
            },
        },
        "metrics": metrics,
        "results": results,
    }
    atomic_json_save(output, path)
    return metrics


def build_seg_gt_index(seg_benchmark: List[Dict[str, Any]]) -> Dict[str, Dict[int, str]]:
    index: Dict[str, Dict[int, str]] = {}
    for row in seg_benchmark:
        video_id = row.get("video_id")
        if not video_id:
            continue
        masks = row.get("masks", [])
        frame_to_path: Dict[int, str] = {}
        for m in masks:
            fi = m.get("frame_index")
            rel = m.get("path")
            if isinstance(fi, int) and isinstance(rel, str):
                frame_to_path[fi] = rel
        index[video_id] = frame_to_path
    return index


def evaluate_single_video_segmentation(
    det_result: Dict[str, Any],
    gt_mask_paths_by_frame: Dict[int, str],
    videos_dir: Optional[str],
    masks_dir: Optional[str],
    asset_resolver: HFDatasetAssetResolver,
    predictor: Any,
    pred_mask_root: str,
    max_prompts: int,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    video_id = det_result.get("video_id")
    if not video_id:
        return None
    try:
        video_path = resolve_local_video_path(
            video_id=video_id,
            videos_dir=videos_dir,
            asset_resolver=asset_resolver,
        )
    except Exception:
        return None

    pred_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for fr in det_result.get("frame_results", []):
        fi = fr.get("frame_index")
        preds = fr.get("predicted_boxes", [])
        if isinstance(fi, int) and isinstance(preds, list):
            pred_by_frame[fi] = preds

    video_stem = os.path.splitext(video_id)[0]
    pred_mask_dir = os.path.join(pred_mask_root, video_stem)
    os.makedirs(pred_mask_dir, exist_ok=True)
    pred_mask_paths = run_edgetam_masks_single_pass(
        predictor=predictor,
        video_path=video_path,
        pred_by_frame=pred_by_frame,
        output_mask_dir=pred_mask_dir,
        max_prompts=max_prompts,
        verbose=verbose,
    )

    frame_metrics: List[Dict[str, Any]] = []
    iou_values: List[float] = []
    dice_values: List[float] = []
    for fi, gt_path in sorted(gt_mask_paths_by_frame.items(), key=lambda x: x[0]):
        try:
            resolved_gt_path = asset_resolver.download_mask(gt_path, local_masks_dir=masks_dir)
        except Exception:
            continue
        if not os.path.isfile(resolved_gt_path):
            continue
        gt_mask = read_gt_mask(resolved_gt_path)
        pred_path = pred_mask_paths.get(fi)
        if pred_path is not None and os.path.isfile(pred_path):
            pred_mask = read_gt_mask(pred_path)
        else:
            pred_mask = np.zeros_like(gt_mask, dtype=bool)

        iou = compute_mask_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        iou_values.append(iou)
        dice_values.append(dice)
        frame_metrics.append(
            {
                "frame_index": fi,
                "gt_mask_path": gt_path,
                "pred_mask_path": pred_path,
                "iou": round(iou, 4),
                "dice": round(dice, 4),
            }
        )

    mean_iou = sum(iou_values) / len(iou_values) if iou_values else 0.0
    mean_dice = sum(dice_values) / len(dice_values) if dice_values else 0.0
    return {
        "video_id": video_id,
        "gt_mask_frames": len(gt_mask_paths_by_frame),
        "pred_mask_frames": len(pred_mask_paths),
        "evaluated_frames": len(frame_metrics),
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
        "frame_metrics": frame_metrics,
    }


def aggregate_segmentation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_iou: List[float] = []
    all_dice: List[float] = []
    gt_frames = 0
    pred_frames = 0
    eval_frames = 0
    for r in results:
        gt_frames += int(r.get("gt_mask_frames", 0) or 0)
        pred_frames += int(r.get("pred_mask_frames", 0) or 0)
        eval_frames += int(r.get("evaluated_frames", 0) or 0)
        for fm in r.get("frame_metrics", []):
            all_iou.append(float(fm.get("iou", 0.0)))
            all_dice.append(float(fm.get("dice", 0.0)))

    mean_iou = sum(all_iou) / len(all_iou) if all_iou else 0.0
    mean_dice = sum(all_dice) / len(all_dice) if all_dice else 0.0
    return {
        "videos_evaluated": len(results),
        "gt_mask_frames": gt_frames,
        "pred_mask_frames": pred_frames,
        "evaluated_frames": eval_frames,
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
    }


def save_segmentation_results_file(
    path: str,
    results: List[Dict[str, Any]],
    total_videos: int,
    model: str,
    checkpoint: str,
    model_cfg: str,
    max_prompts: int,
) -> Dict[str, Any]:
    metrics = aggregate_segmentation_metrics(results)
    videos_failed = total_videos - len(results)
    output = {
        "metadata": {
            "task": "segmentation",
            "model": model,
            "checkpoint": checkpoint,
            "model_cfg": model_cfg,
            "max_prompts": max_prompts,
            "last_updated": datetime.now().isoformat(),
            "total_videos": total_videos,
            "videos_evaluated": len(results),
            "videos_failed": videos_failed,
            "complete": videos_failed == 0,
            "metrics_summary": {
                "mean_iou": metrics.get("mean_iou", 0.0),
                "mean_dice": metrics.get("mean_dice", 0.0),
            },
        },
        "metrics": metrics,
        "results": results,
    }
    atomic_json_save(output, path)
    return metrics


def resolve_benchmark_paths(
    benchmark: str,
    videos: Optional[str],
    seg: bool,
    seg_benchmark_override: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Resolve detection benchmark path, optional segmentation benchmark path,
    videos dir, and benchmark directory.
    """
    if os.path.isdir(benchmark):
        benchmark_dir = benchmark
        det_path = os.path.join(benchmark_dir, DETECTION_BENCHMARK_FILENAME)
        seg_path = os.path.join(benchmark_dir, SEGMENTATION_BENCHMARK_FILENAME) if seg else None
        videos_dir = videos or os.path.join(benchmark_dir, VIDEOS_SUBDIR)
    elif os.path.isfile(benchmark):
        det_path = benchmark
        benchmark_dir = os.path.dirname(det_path)
        seg_path = os.path.join(benchmark_dir, SEGMENTATION_BENCHMARK_FILENAME) if seg else None
        videos_dir = videos or os.path.join(benchmark_dir, VIDEOS_SUBDIR)
    else:
        raise FileNotFoundError(f"Benchmark path not found: {benchmark}")

    if seg and seg_benchmark_override:
        seg_path = seg_benchmark_override

    if not os.path.isfile(det_path):
        raise FileNotFoundError(f"Detection benchmark not found: {det_path}")
    if videos_dir and not os.path.isdir(videos_dir):
        videos_dir = None
    if seg and (seg_path is None or not os.path.isfile(seg_path)):
        raise FileNotFoundError(
            "Segmentation benchmark not found. Expected benchmark_segmentation.json "
            f"at: {seg_path}"
        )
    return det_path, seg_path, videos_dir, benchmark_dir


def run_detection_evaluation(
    benchmark: List[Dict[str, Any]],
    videos_dir: Optional[str],
    asset_resolver: HFDatasetAssetResolver,
    model: str,
    output_dir: str,
    frames_count: int,
    parallel: bool,
    max_workers: int,
    max_retries: int,
    new_run: bool,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    os.makedirs(output_dir, exist_ok=True)

    model_safe = sanitize_model_name(model)
    prefix = f"llm_eval_{model_safe}_det_"

    total_videos = len(benchmark)
    existing_results: Dict[str, Dict[str, Any]] = {}
    results_path: Optional[str] = None

    if not new_run:
        prev_path = find_previous_results(output_dir, prefix)
        if prev_path:
            prev_data = load_json_safe(prev_path)
            if isinstance(prev_data, dict):
                for r in prev_data.get("results", []):
                    video_id = r.get("video_id")
                    if video_id:
                        existing_results[video_id] = r
                results_path = prev_path
                print(f"  Resuming detection from: {prev_path}")
                print(f"  Previously completed: {len(existing_results)}/{total_videos}")

    if results_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"{prefix}{timestamp}.json")
        if not new_run:
            print("  No previous detection results found - starting fresh.")

    to_process: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for row in benchmark:
        video_id = row.get("video_id")
        if video_id in existing_results:
            results.append(existing_results[video_id])
        else:
            to_process.append(row)

    if not to_process:
        metrics = save_detection_results_file(
            results_path,
            results,
            total_videos,
            model,
            frames_count,
            parallel,
            max_workers,
            max_retries,
        )
        return results, metrics, results_path

    resolved_id, supports_video = resolve_openrouter_model(model)
    print(f"  OpenRouter ID: {resolved_id}")
    print(f"  Video capability: {supports_video} (detection uses frame mode)")

    completed_this_run = 0
    failed_this_run = 0
    metrics: Dict[str, Any] = {}

    def process_video(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return evaluate_single_video_detection(
                record=item,
                videos_dir=videos_dir,
                asset_resolver=asset_resolver,
                model=model,
                frames_count=frames_count,
                max_retries=max_retries,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                video_id = item.get("video_id", "unknown")
                print(f"  [ERROR] video={video_id} failed: {exc}")
            return None

    try:
        if parallel and len(to_process) > 1:
            print(
                f"  [PARALLEL] Processing {len(to_process)} videos "
                f"with {max_workers} workers"
            )
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_video, item): item for item in to_process}
                for future in as_completed(futures):
                    completed_this_run += 1
                    res = future.result()
                    if res is not None:
                        results.append(res)
                    else:
                        failed_this_run += 1

                    if completed_this_run % 5 == 0 or completed_this_run == len(to_process):
                        metrics = save_detection_results_file(
                            results_path,
                            results,
                            total_videos,
                            model,
                            frames_count,
                            parallel,
                            max_workers,
                            max_retries,
                        )

                    if completed_this_run % 10 == 0 or completed_this_run == len(to_process):
                        missing = total_videos - len(results)
                        print(
                            f"  Detection progress: {completed_this_run}/{len(to_process)} "
                            f"this run | {len(results)}/{total_videos} total (missing={missing})"
                        )
        else:
            for i, item in enumerate(to_process):
                _ = i
                completed_this_run += 1
                res = process_video(item)
                if res is not None:
                    results.append(res)
                else:
                    failed_this_run += 1

                if completed_this_run % 5 == 0 or completed_this_run == len(to_process):
                    metrics = save_detection_results_file(
                        results_path,
                        results,
                        total_videos,
                        model,
                        frames_count,
                        parallel,
                        max_workers,
                        max_retries,
                    )

                if completed_this_run % 10 == 0 or completed_this_run == len(to_process):
                    missing = total_videos - len(results)
                    print(
                        f"  Detection progress: {completed_this_run}/{len(to_process)} "
                        f"this run | {len(results)}/{total_videos} total (missing={missing})"
                    )

        metrics = save_detection_results_file(
            results_path,
            results,
            total_videos,
            model,
            frames_count,
            parallel,
            max_workers,
            max_retries,
        )
    finally:
        try:
            save_detection_results_file(
                results_path,
                results,
                total_videos,
                model,
                frames_count,
                parallel,
                max_workers,
                max_retries,
            )
        except Exception:
            pass

    missing = total_videos - len(results)
    if missing == 0:
        print(f"\n  DETECTION COMPLETE - all {total_videos} videos evaluated (fail=0)")
    else:
        print(f"\n  Detection done: {len(results)}/{total_videos} videos ({missing} missing). Re-run to retry.")
    print(f"  Detection results: {results_path}")
    if failed_this_run > 0:
        print(f"  Failed in this run: {failed_this_run}")
    return results, metrics, results_path


def run_segmentation_evaluation(
    *,
    det_results: List[Dict[str, Any]],
    seg_benchmark: List[Dict[str, Any]],
    videos_dir: Optional[str],
    masks_dir: Optional[str],
    asset_resolver: HFDatasetAssetResolver,
    model: str,
    output_dir: str,
    checkpoint: str,
    model_cfg: str,
    max_prompts: int,
    new_run: bool,
    verbose: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    os.makedirs(output_dir, exist_ok=True)
    model_safe = sanitize_model_name(model)
    prefix = f"llm_eval_{model_safe}_seg_"

    gt_index = build_seg_gt_index(seg_benchmark)
    total_videos = len(det_results)
    existing_results: Dict[str, Dict[str, Any]] = {}
    results_path: Optional[str] = None

    if not new_run:
        prev_path = find_previous_results(output_dir, prefix)
        if prev_path:
            prev_data = load_json_safe(prev_path)
            if isinstance(prev_data, dict):
                for r in prev_data.get("results", []):
                    video_id = r.get("video_id")
                    if video_id:
                        existing_results[video_id] = r
                results_path = prev_path
                print(f"  Resuming segmentation from: {prev_path}")
                print(f"  Previously completed: {len(existing_results)}/{total_videos}")

    if results_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"{prefix}{timestamp}.json")
        if not new_run:
            print("  No previous segmentation results found - starting fresh.")

    pred_mask_root = os.path.join(output_dir, f"{model_safe}_seg_pred_masks")
    os.makedirs(pred_mask_root, exist_ok=True)

    to_process: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for r in det_results:
        video_id = r.get("video_id")
        if not video_id:
            continue
        if video_id in existing_results:
            results.append(existing_results[video_id])
        elif video_id in gt_index:
            to_process.append(r)
        elif verbose:
            print(f"  [SEG][SKIP] No GT segmentation entry for video: {video_id}")

    if not to_process:
        metrics = save_segmentation_results_file(
            results_path,
            results,
            total_videos,
            model,
            checkpoint,
            model_cfg,
            max_prompts,
        )
        return results, metrics, results_path

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"EdgeTAM checkpoint not found: {checkpoint}")

    edgetam_path = os.path.join(PROJECT_ROOT, "edgetam")
    if edgetam_path not in sys.path:
        sys.path.insert(0, edgetam_path)
    predictor = init_edgetam_predictor(model_cfg=model_cfg, checkpoint=checkpoint)

    completed_this_run = 0
    failed_this_run = 0
    metrics: Dict[str, Any] = {}
    for det_item in to_process:
        completed_this_run += 1
        video_id = det_item.get("video_id")
        try:
            seg_result = evaluate_single_video_segmentation(
                det_result=det_item,
                gt_mask_paths_by_frame=gt_index.get(video_id, {}),
                videos_dir=videos_dir,
                masks_dir=masks_dir,
                asset_resolver=asset_resolver,
                predictor=predictor,
                pred_mask_root=pred_mask_root,
                max_prompts=max_prompts,
                verbose=verbose,
            )
            if seg_result is not None:
                results.append(seg_result)
            else:
                failed_this_run += 1
        except Exception as exc:  # noqa: BLE001
            failed_this_run += 1
            if verbose:
                print(f"  [SEG][ERROR] video={video_id}: {exc}")

        if completed_this_run % 3 == 0 or completed_this_run == len(to_process):
            metrics = save_segmentation_results_file(
                results_path,
                results,
                total_videos,
                model,
                checkpoint,
                model_cfg,
                max_prompts,
            )
            missing = total_videos - len(results)
            print(
                f"  Segmentation progress: {completed_this_run}/{len(to_process)} this run "
                f"| {len(results)}/{total_videos} total (missing={missing})"
            )

    metrics = save_segmentation_results_file(
        results_path,
        results,
        total_videos,
        model,
        checkpoint,
        model_cfg,
        max_prompts,
    )
    missing = total_videos - len(results)
    if missing == 0:
        print(f"\n  SEGMENTATION COMPLETE - all {total_videos} videos evaluated (fail=0)")
    else:
        print(f"\n  Segmentation done: {len(results)}/{total_videos} videos ({missing} missing). Re-run to retry.")
    print(f"  Segmentation results: {results_path}")
    if failed_this_run > 0:
        print(f"  Failed in this run: {failed_this_run}")
    return results, metrics, results_path


def print_detection_report(metrics: Dict[str, Any], model: str) -> None:
    print("\n" + "=" * 68)
    print(f"  Detection Evaluation Report - {model}")
    print("=" * 68)
    print(f"  Precision@0.5: {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall@0.5:    {metrics.get('recall', 0.0):.4f}")
    print(f"  F1@0.5:        {metrics.get('f1', 0.0):.4f}")
    print(f"  AP50:          {metrics.get('ap50', 0.0):.4f}")
    print(f"  mAP50-95:      {metrics.get('map50_95', 0.0):.4f}")
    print(f"  Mean IoU:      {metrics.get('mean_iou', 0.0):.4f}")
    tok = metrics.get("token_usage", {})
    if tok:
        print(f"  Prompt tokens: {int(tok.get('total_prompt_tokens', 0)):,}")
        print(f"  Completion tokens: {int(tok.get('total_completion_tokens', 0)):,}")
        print(f"  Total tokens:  {int(tok.get('total_tokens', 0)):,}")
    print("=" * 68)


def print_segmentation_report(metrics: Dict[str, Any], model: str) -> None:
    print("\n" + "=" * 68)
    print(f"  Segmentation Evaluation Report - {model}")
    print("=" * 68)
    print(f"  Mean IoU:      {metrics.get('mean_iou', 0.0):.4f}")
    print(f"  Mean Dice:     {metrics.get('mean_dice', 0.0):.4f}")
    print(f"  Eval frames:   {int(metrics.get('evaluated_frames', 0))}")
    print(f"  GT mask frames:{int(metrics.get('gt_mask_frames', 0))}")
    print(f"  Pred mask frames:{int(metrics.get('pred_mask_frames', 0))}")
    print("=" * 68)


def parse_args() -> argparse.Namespace:
    openrouter_hint = ", ".join(list_openrouter_models()[:8])
    parser = argparse.ArgumentParser(
        description="Evaluate ColonBench detection with optional EdgeTAM segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/llm_evaluate_det_seg.py --benchmark data/colon-bench\n"
            "  python scripts/llm_evaluate_det_seg.py --benchmark data/colon-bench --model gpt-4o --parallel --max-workers 4\n"
            "  python scripts/llm_evaluate_det_seg.py --benchmark data/colon-bench --seg --checkpoint edgetam/checkpoints/edgetam.pt\n\n"
            f"OpenRouter aliases (subset): {openrouter_hint}"
        ),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/colon-bench",
        help=(
            "Path to benchmark directory containing benchmark_detection.json "
            "or direct path to benchmark_detection.json"
        ),
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Optional local videos directory. If omitted, videos are downloaded from HF as needed.",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default=None,
        help="Optional local masks directory used in segmentation mode before falling back to HF.",
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
        help="Optional HF token override for gated dataset access.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to evaluate (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory for det/seg results "
            f"(default: <benchmark_dir>/{DEFAULT_OUTPUT_SUBDIR})"
        ),
    )
    parser.add_argument(
        "--frames-count",
        type=int,
        default=3,
        help=(
            "Number of local frames to localize per video, starting from frame 0 "
            "of the tracking window clip (default: 3)"
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Enable parallel processing over videos for detection evaluation only. "
            "Segmentation always runs sequentially (GPU cannot be shared across workers)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel workers for detection mode (default: 8)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for each frame-level model call (default: 3)",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a fresh run, ignoring previous results (default: auto-resume)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N benchmark videos (for testing)",
    )
    parser.add_argument(
        "--seg",
        action="store_true",
        help=(
            "Run segmentation only (no detection). Requires existing detection "
            "results in the output directory. Exits with error if none found."
        ),
    )
    parser.add_argument(
        "--seg-benchmark",
        type=str,
        default=None,
        help=(
            "Optional explicit path to benchmark_segmentation.json "
            "(default: auto-detected from benchmark directory)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "edgetam" / "checkpoints" / "edgetam.pt"),
        help="Path to EdgeTAM checkpoint (used when --seg is set)",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="edgetam.yaml",
        help="EdgeTAM model config (default: edgetam.yaml)",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=1400,
        help="Maximum detection prompts forwarded to EdgeTAM in seg mode (default: 1400)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug details",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames_count <= 0:
        print("[ERROR] --frames-count must be > 0")
        return 1
    if args.max_workers <= 0:
        print("[ERROR] --max-workers must be > 0")
        return 1
    if args.max_retries <= 0:
        print("[ERROR] --max-retries must be > 0")
        return 1
    if args.max_prompts <= 0:
        print("[ERROR] --max-prompts must be > 0")
        return 1

    try:
        det_path, seg_path, videos_dir, benchmark_dir = resolve_benchmark_paths(
            benchmark=args.benchmark,
            videos=args.videos,
            seg=args.seg,
            seg_benchmark_override=args.seg_benchmark,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    with open(det_path, "r") as f:
        detection_benchmark = json.load(f)
    if not isinstance(detection_benchmark, list) or not detection_benchmark:
        print(f"[ERROR] Detection benchmark is empty/invalid: {det_path}")
        return 1

    if args.limit and args.limit > 0:
        detection_benchmark = detection_benchmark[: args.limit]
        print(f"  [LIMITED] Evaluating first {len(detection_benchmark)} videos")

    seg_benchmark: List[Dict[str, Any]] = []
    if args.seg and seg_path:
        with open(seg_path, "r") as f:
            seg_benchmark = json.load(f)
        if not isinstance(seg_benchmark, list) or not seg_benchmark:
            print(f"[ERROR] Segmentation benchmark is empty/invalid: {seg_path}")
            return 1

    if args.output is None:
        args.output = os.path.join(benchmark_dir, DEFAULT_OUTPUT_SUBDIR)
    os.makedirs(args.output, exist_ok=True)
    asset_resolver = HFDatasetAssetResolver(
        repo_id=args.dataset_repo_id,
        revision=args.dataset_revision,
        token=args.hf_token,
    )

    print("=" * 72)
    if args.seg:
        print("  EdgeTAM Segmentation Evaluation (from existing detection)")
    else:
        print("  ColonBench Detection Evaluation")
    print("=" * 72)
    print(f"  Detection benchmark: {det_path}")
    if args.seg:
        print(f"  Segmentation benchmark: {seg_path}")
    print(f"  Benchmark dir: {benchmark_dir}")
    print(f"  Dataset repo: {args.dataset_repo_id}@{args.dataset_revision}")
    print(f"  Videos dir: {videos_dir or '(not provided - download from HF)'}")
    if args.seg:
        print(f"  Masks dir: {args.masks_dir or '(not provided - download from HF)'}")
    print(f"  Output dir: {args.output}")
    print(f"  Model: {args.model}")
    if not args.seg:
        resolved_id, supports_video = resolve_openrouter_model(args.model)
        print("  Backend: OpenRouter")
        print(f"  OpenRouter ID: {resolved_id}")
        print(f"  Native video support: {supports_video} (detection uses image frames)")
        print(f"  Frames count: {args.frames_count}")
        print(f"  Parallel (det): {args.parallel} (workers: {args.max_workers})")
        print(f"  Resume: {'No (--new)' if args.new else 'Yes (auto)'}")
    print(f"  Mode: {'Segmentation only (--seg)' if args.seg else 'Detection only'}")
    if args.seg:
        print(f"  Seg resume: {'No (--new)' if args.new else 'Yes (auto)'}")
        print(f"  Seg parallel: False (GPU - always sequential)")
        print(f"  EdgeTAM checkpoint: {args.checkpoint}")
        print(f"  EdgeTAM cfg: {args.model_cfg}")
        print(f"  Max prompts: {args.max_prompts}")
    print(f"  Start time: {datetime.now().isoformat()}")
    print("=" * 72)

    if args.seg:
        # --seg mode: load existing detection results (no detection run).
        model_safe = sanitize_model_name(args.model)
        det_prefix = f"llm_eval_{model_safe}_det_"
        det_results_path = find_previous_results(args.output, det_prefix)
        if det_results_path is None:
            print(
                f"[ERROR] --seg requires existing detection results but none "
                f"found in {args.output} matching '{det_prefix}*.json'. "
                f"Run detection first (without --seg), then re-run with --seg."
            )
            return 1

        prev_data = load_json_safe(det_results_path)
        if not isinstance(prev_data, dict) or not prev_data.get("results"):
            print(
                f"[ERROR] Detection results file is invalid or has no results: "
                f"{det_results_path}"
            )
            return 1

        det_results: List[Dict[str, Any]] = prev_data["results"]
        det_metrics = prev_data.get("metrics", {})
        print(f"  Loaded detection results: {det_results_path}")
        print(f"  Detection videos loaded: {len(det_results)}")
        print_detection_report(det_metrics, args.model)

        seg_results, seg_metrics, seg_results_path = run_segmentation_evaluation(
            det_results=det_results,
            seg_benchmark=seg_benchmark,
            videos_dir=videos_dir,
            masks_dir=args.masks_dir,
            asset_resolver=asset_resolver,
            model=args.model,
            output_dir=args.output,
            checkpoint=args.checkpoint,
            model_cfg=args.model_cfg,
            max_prompts=args.max_prompts,
            new_run=args.new,
            verbose=args.verbose,
        )
        _ = seg_results
        print_segmentation_report(seg_metrics, args.model)
        print(f"\n  Segmentation JSON: {seg_results_path}")
        print(f"\n  Detection JSON (loaded): {det_results_path}")
    else:
        det_results, det_metrics, det_results_path = run_detection_evaluation(
            benchmark=detection_benchmark,
            videos_dir=videos_dir,
            asset_resolver=asset_resolver,
            model=args.model,
            output_dir=args.output,
            frames_count=args.frames_count,
            parallel=args.parallel,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            new_run=args.new,
            verbose=args.verbose,
        )
        print_detection_report(det_metrics, args.model)
        print(f"\n  Detection JSON: {det_results_path}")

    print(f"  End time: {datetime.now().isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
