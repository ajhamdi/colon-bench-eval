"""
visualize_benchmark.py

Streamlit viewer for all colon-bench benchmark modes:
- VQA prompted
- VQA unprompted
- Classification
- Detection
- Segmentation

Primary usage:
    streamlit run visualize_benchmark.py -- --benchmark /path/to/data/colon-bench --mode vqa_prompted

Mode examples:
    streamlit run visualize_benchmark.py -- --benchmark /path/to/data/colon-bench --mode classification
    streamlit run visualize_benchmark.py -- --benchmark /path/to/data/colon-bench --mode segmentation
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from constants import DEFAULT_CLIP_FPS, MASK_ALPHA, MASK_COLOR_BGR

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from colonbench_eval.hf_video import HFDatasetAssetResolver  # noqa: E402


# ========================= CONSTANTS =========================================

MODE_ORDER = ["vqa_prompted", "vqa_unprompted", "classification", "segmentation"]
MODE_TO_FILENAME = {
    "vqa_prompted": "benchmark_vqa_prompted.json",
    "vqa_unprompted": "benchmark_vqa_unprompted.json",
    "classification": "benchmark_cls.json",
    "segmentation": "benchmark_segmentation.json",
}
MODE_LABELS = {
    "vqa_prompted": "VQA Prompted",
    "vqa_unprompted": "VQA Unprompted",
    "classification": "Classification",
    "segmentation": "Segmentation",
}
QUIZ_MODES = {"vqa_prompted", "vqa_unprompted", "classification"}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "data", "colon-bench")
TEMP_VIDEO_DIR = os.path.join(tempfile.gettempdir(), "benchmark_viewer_videos")
os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)


# ========================= ARGUMENT PARSING ==================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments, ignoring unknown Streamlit arguments."""
    parser = argparse.ArgumentParser(description="Colon-Bench Visualization")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Path to colon-bench directory containing benchmark JSONs (videos/masks streamed from HF)",
    )
    parser.add_argument(
        "--mode",
        default="vqa_prompted",
        choices=MODE_ORDER,
        help="Initial mode to open in the UI",
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        dest="benchmark_json",
        help="Optional direct benchmark JSON path (overrides selected mode file only)",
    )
    parser.add_argument(
        "--videos",
        default=None,
        help="Optional direct path to videos directory",
    )
    parser.add_argument(
        "--masks",
        default=None,
        help="Optional direct path to masks directory",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        dest="hf_token",
        help="Optional HF token override. Otherwise uses HF_TOKEN env var or local login.",
    )

    try:
        args, _ = parser.parse_known_args(sys.argv[1:])
        return args
    except SystemExit:
        return parser.parse_args([])


ARGS = parse_args()


# ========================= PAGE CONFIG =======================================

st.set_page_config(
    page_title="Colon-Bench Viewer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    .question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 1rem 0 1.5rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .question-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }

    .choice-btn {
        display: block;
        width: 100%;
        padding: 14px 20px;
        margin: 8px 0;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 500;
        text-align: left;
        border: 2px solid #e2e8f0;
        background: #f8fafc;
        color: #1e293b;
        line-height: 1.5;
    }

    .choice-correct {
        border-color: #16a34a !important;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        color: #166534 !important;
        box-shadow: 0 4px 16px rgba(22, 163, 74, 0.2) !important;
    }

    .choice-correct .choice-key {
        background: #16a34a !important;
        color: white !important;
    }

    .choice-wrong {
        border-color: #dc2626 !important;
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%) !important;
        color: #991b1b !important;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.2) !important;
    }

    .choice-wrong .choice-key {
        background: #dc2626 !important;
        color: white !important;
    }

    .choice-dimmed {
        opacity: 0.5;
        border-color: #e2e8f0 !important;
        background: #f1f5f9 !important;
    }

    .choice-key {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        border-radius: 8px;
        background: #e2e8f0;
        color: #475569;
        font-weight: 700;
        margin-right: 12px;
        font-size: 0.95rem;
    }

    .instructions-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #475569;
    }

    .stButton > button {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        border: none;
    }

    .stButton > button {
        background: #9ca3af !important;
        color: white !important;
    }

    .video-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #e2e8f0;
        color: #475569;
        margin-bottom: 0.5rem;
    }

    .error-container {
        text-align: center;
        padding: 60px 40px;
        background: #fff5f5;
        border-radius: 16px;
        border: 2px solid #feb2b2;
        margin: 2rem auto;
        max-width: 720px;
    }

    .error-container h2 {
        color: #c53030;
        margin-bottom: 1rem;
    }

    .error-container code {
        background: #1a1a2e;
        color: #68d391;
        padding: 1rem;
        border-radius: 8px;
        display: block;
        margin: 1rem 0;
        font-size: 0.9rem;
        text-align: left;
        white-space: pre-wrap;
    }
</style>
""",
    unsafe_allow_html=True,
)

BUTTON_STYLE_JS = """
<script>
function styleButtons() {
    const doc = window.parent.document;
    const buttons = doc.querySelectorAll('.stButton button');
    buttons.forEach(btn => {
        const text = btn.textContent.trim();
        if (text.includes('Previous') || text.includes('Next')) {
            btn.style.setProperty('background', 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 'important');
            btn.style.setProperty('border', 'none', 'important');
        }
    });
}
setTimeout(styleButtons, 100);
const observer = new MutationObserver(() => setTimeout(styleButtons, 50));
observer.observe(window.parent.document.body, { childList: true, subtree: true });
</script>
"""


# ========================= DATA LOADING ======================================

def abs_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (ValueError, TypeError):
        return None


def ensure_mp4_name(video_id: Any) -> str:
    value = str(video_id or "").strip()
    if not value:
        return ""
    if value.endswith(".mp4"):
        return value
    return f"{value}.mp4"


def resolve_assets(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve benchmark directory, mode files, and related asset directories."""
    bench_dir = abs_path(args.benchmark)
    if not bench_dir and os.path.isdir(DEFAULT_BENCHMARK_DIR):
        bench_dir = DEFAULT_BENCHMARK_DIR
    if bench_dir and not os.path.isdir(bench_dir):
        bench_dir = None

    mode_files: Dict[str, str] = {}

    if bench_dir:
        for mode, filename in MODE_TO_FILENAME.items():
            candidate = os.path.join(bench_dir, filename)
            if os.path.isfile(candidate):
                mode_files[mode] = candidate

        legacy_benchmark = os.path.join(bench_dir, "benchmark.json")
        if os.path.isfile(legacy_benchmark) and "vqa_prompted" not in mode_files:
            mode_files["vqa_prompted"] = legacy_benchmark

    benchmark_json = abs_path(args.benchmark_json)
    if benchmark_json:
        mode_files[args.mode] = benchmark_json
        if not bench_dir:
            bench_dir = os.path.dirname(benchmark_json)

    videos_dir = abs_path(args.videos)
    if not videos_dir and bench_dir:
        candidate = os.path.join(bench_dir, "videos")
        if os.path.isdir(candidate):
            videos_dir = candidate

    masks_dir = abs_path(args.masks)
    if not masks_dir and bench_dir:
        candidate = os.path.join(bench_dir, "masks")
        if os.path.isdir(candidate):
            masks_dir = candidate

    available_modes = [m for m in MODE_ORDER if m in mode_files]

    hf_resolver = HFDatasetAssetResolver(token=getattr(args, "hf_token", None))

    return {
        "benchmark_dir": bench_dir,
        "mode_files": mode_files,
        "available_modes": available_modes,
        "videos_dir": videos_dir,
        "masks_dir": masks_dir,
        "hf_resolver": hf_resolver,
    }


@st.cache_data(show_spinner=False)
def load_json_list(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return []


def normalize_vqa_records(records: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        video_id = ensure_mp4_name(rec.get("video"))
        question = str(rec.get("question", "")).strip()
        choices = rec.get("choices", {})
        answer = str(rec.get("answer", "")).strip()
        question_id = str(rec.get("question_id", "")).strip()
        if not video_id or not question or not isinstance(choices, dict):
            continue

        clean_choices = {str(k): str(v) for k, v in choices.items()}
        item_id = question_id if question_id else f"{mode}_{idx}"
        items.append(
            {
                "item_id": item_id,
                "mode": mode,
                "video_id": video_id,
                "prompt_text": question,
                "choices": clean_choices,
                "correct_answer": answer,
                "aux": {"question_id": question_id},
            }
        )
    return items


def normalize_classification_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        video_id = ensure_mp4_name(rec.get("video_id"))
        lesion = safe_int(rec.get("lesion"))
        if not video_id or lesion not in (0, 1):
            continue

        choices = {
            "A": "Lesion present",
            "B": "No lesion detected",
        }
        correct_answer = "A" if lesion == 1 else "B"
        items.append(
            {
                "item_id": f"classification_{idx}_{video_id}",
                "mode": "classification",
                "video_id": video_id,
                "prompt_text": "Is there a visible lesion in this clip?",
                "choices": choices,
                "correct_answer": correct_answer,
                "aux": {"lesion": lesion},
            }
        )
    return items


def normalize_segmentation_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        video_id = ensure_mp4_name(rec.get("video_id"))
        if not video_id:
            continue

        description = str(rec.get("description", "")).strip()
        raw_masks = rec.get("masks", [])
        clean_masks: List[Dict[str, Any]] = []
        if isinstance(raw_masks, list):
            for m in raw_masks:
                if not isinstance(m, dict):
                    continue
                frame_index = safe_int(m.get("frame_index"))
                path = str(m.get("path", "")).strip()
                if frame_index is None:
                    continue
                clean_masks.append({"frame_index": frame_index, "path": path})
        clean_masks.sort(key=lambda e: e["frame_index"])

        items.append(
            {
                "item_id": f"segmentation_{idx}_{video_id}",
                "mode": "segmentation",
                "video_id": video_id,
                "prompt_text": "Segmentation benchmark sample",
                "choices": {},
                "correct_answer": "",
                "aux": {
                    "description": description,
                    "masks": clean_masks,
                },
            }
        )
    return items


def group_items_by_video(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for item in items:
        video_id = item.get("video_id", "")
        if video_id not in groups:
            groups[video_id] = []
            order.append(video_id)
        groups[video_id].append(item)
    return {video_id: groups[video_id] for video_id in order}


# ========================= MEDIA HELPERS =====================================

@st.cache_resource(show_spinner=False)
def _load_video_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def cleanup_old_videos(keep_recent: int = 80) -> None:
    if not os.path.isdir(TEMP_VIDEO_DIR):
        return
    video_paths: List[Tuple[str, float]] = []
    for name in os.listdir(TEMP_VIDEO_DIR):
        if not name.endswith(".mp4"):
            continue
        path = os.path.join(TEMP_VIDEO_DIR, name)
        try:
            video_paths.append((path, os.path.getmtime(path)))
        except OSError:
            continue
    video_paths.sort(key=lambda entry: entry[1], reverse=True)
    for old_path, _ in video_paths[keep_recent:]:
        try:
            os.remove(old_path)
        except OSError:
            pass


def _short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _reencode_to_h264(src_path: str) -> str:
    """Re-encode an mp4v video to H.264 so browsers can play it.

    Returns the path to the H.264 file (replaces the original).
    Falls back to the original path if ffmpeg is unavailable or fails.
    """
    tmp_path = src_path + ".h264.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                tmp_path,
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        os.replace(tmp_path, src_path)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return src_path


def apply_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = MASK_COLOR_BGR,
    alpha: float = MASK_ALPHA,
) -> np.ndarray:
    if mask is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = np.zeros_like(out)
    overlay[:] = color
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    return np.where(mask_3ch > 0, cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0), out)


def load_binary_mask(mask_path: str) -> Optional[np.ndarray]:
    if not mask_path or not os.path.isfile(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


def _resolve_mask_local(
    mask_record: Dict[str, Any],
    benchmark_dir: Optional[str],
    masks_dir: Optional[str],
    video_id: str,
) -> Optional[str]:
    """Try to find a mask on local disk only (no network)."""
    path_value = str(mask_record.get("path", "")).strip()
    frame_index = safe_int(mask_record.get("frame_index"))
    candidates: List[str] = []

    if path_value:
        if os.path.isabs(path_value):
            candidates.append(path_value)
        else:
            if benchmark_dir:
                candidates.append(os.path.join(benchmark_dir, path_value))
            if masks_dir:
                normalized_rel = path_value
                if normalized_rel.startswith("masks/"):
                    normalized_rel = normalized_rel[len("masks/"):]
                candidates.append(os.path.join(masks_dir, normalized_rel))
                candidates.append(
                    os.path.join(
                        masks_dir,
                        os.path.basename(os.path.dirname(path_value)),
                        os.path.basename(path_value),
                    )
                )

    if masks_dir and frame_index is not None:
        stem = os.path.splitext(video_id)[0]
        candidates.append(os.path.join(masks_dir, stem, f"{frame_index:06d}.png"))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


_MASK_DOWNLOAD_WORKERS = 16


def build_segmentation_mask_index(item: Dict[str, Any], assets: Dict[str, Any]) -> Dict[int, str]:
    benchmark_dir = assets.get("benchmark_dir")
    masks_dir = assets.get("masks_dir")
    hf_resolver: Optional[HFDatasetAssetResolver] = assets.get("hf_resolver")
    video_id = item.get("video_id", "")
    records = item.get("aux", {}).get("masks", [])

    index: Dict[int, str] = {}
    need_hf: List[Tuple[int, str]] = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        frame_index = safe_int(rec.get("frame_index"))
        if frame_index is None:
            continue

        local = _resolve_mask_local(rec, benchmark_dir, masks_dir, video_id)
        if local:
            index[frame_index] = local
        else:
            path_value = str(rec.get("path", "")).strip()
            if path_value:
                repo_path = path_value if path_value.startswith("masks/") else f"masks/{path_value}"
                need_hf.append((frame_index, repo_path))

    if need_hf and hf_resolver:
        def _download(entry: Tuple[int, str]) -> Tuple[int, Optional[str]]:
            fi, rp = entry
            try:
                return fi, hf_resolver.download_mask(rp, local_masks_dir=masks_dir)
            except Exception:
                return fi, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MASK_DOWNLOAD_WORKERS) as pool:
            for fi, path in pool.map(_download, need_hf):
                if path:
                    index[fi] = path

    return index


def _download_video_from_hf(video_id: str, assets: Dict[str, Any]) -> Optional[str]:
    hf_resolver: Optional[HFDatasetAssetResolver] = assets.get("hf_resolver")
    if not hf_resolver or not video_id:
        return None
    try:
        return hf_resolver.download_video(video_id, local_videos_dir=assets.get("videos_dir"))
    except Exception:
        return None


def _resolve_base_video(video_id: str, assets: Dict[str, Any]) -> Optional[str]:
    """Find the base video locally or download from HF."""
    videos_dir = assets.get("videos_dir")
    if videos_dir:
        candidate = os.path.join(videos_dir, video_id)
        if os.path.isfile(candidate):
            return candidate
    return _download_video_from_hf(video_id, assets)


def generate_segmentation_overlay_video(item: Dict[str, Any], assets: Dict[str, Any]) -> Optional[str]:
    """Download video + masks from HF, overlay masks on frames, return H.264 video."""
    video_id = item.get("video_id", "")
    if not video_id:
        return None

    mask_index = build_segmentation_mask_index(item, assets)

    source_video_path = _resolve_base_video(video_id, assets)
    if not source_video_path:
        return None

    if not mask_index:
        return source_video_path

    cache_key = _short_hash(
        f"seg_overlay_v2|{video_id}|{json.dumps(sorted(mask_index.items()), sort_keys=True)}"
    )
    output_path = os.path.join(TEMP_VIDEO_DIR, f"seg_{cache_key}.mp4")
    if os.path.isfile(output_path):
        return output_path

    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        return source_video_path

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = float(DEFAULT_CLIP_FPS)

    writer = None
    frame_idx = 0
    written = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            mask_path = mask_index.get(frame_idx)
            if mask_path:
                mask = load_binary_mask(mask_path)
                if mask is not None:
                    frame = apply_mask_overlay(frame, mask)

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
                if not writer.isOpened():
                    writer = None
                    break

            writer.write(frame)
            written += 1
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if written > 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return _reencode_to_h264(output_path)

    try:
        if os.path.isfile(output_path):
            os.remove(output_path)
    except OSError:
        pass
    return source_video_path


def resolve_video_path_for_item(mode: str, item: Dict[str, Any], assets: Dict[str, Any]) -> Optional[str]:
    video_id = item.get("video_id", "")

    if mode == "segmentation":
        return generate_segmentation_overlay_video(item, assets)

    return _resolve_base_video(video_id, assets)


# ========================= SESSION STATE =====================================

def _default_mode_state() -> Dict[str, Any]:
    return {
        "flat_idx": 0,
        "current_video_idx": 0,
        "revealed": {},
        "visited": [],
    }


def get_mode_state(mode: str) -> Dict[str, Any]:
    mode_state = st.session_state.bench_mode_state
    if mode not in mode_state:
        mode_state[mode] = _default_mode_state()
    return mode_state[mode]


def init_session_state() -> None:
    if "bench_initialized" in st.session_state:
        return

    st.session_state.bench_initialized = True
    st.session_state.bench_assets = resolve_assets(ARGS)
    st.session_state.bench_mode_state = {}
    st.session_state.bench_items_cache = {}

    available_modes = st.session_state.bench_assets.get("available_modes", [])
    if ARGS.mode in available_modes:
        initial_mode = ARGS.mode
    elif available_modes:
        initial_mode = available_modes[0]
    else:
        initial_mode = ARGS.mode
    st.session_state.bench_mode = initial_mode
    st.session_state.bench_mode_select = initial_mode

    cleanup_old_videos(keep_recent=80)


def load_items_for_mode(mode: str) -> List[Dict[str, Any]]:
    cache = st.session_state.bench_items_cache
    if mode in cache:
        return cache[mode]

    mode_file = st.session_state.bench_assets.get("mode_files", {}).get(mode)
    if not mode_file or not os.path.isfile(mode_file):
        cache[mode] = []
        return []

    records = load_json_list(mode_file)
    if mode in ("vqa_prompted", "vqa_unprompted"):
        items = normalize_vqa_records(records, mode)
    elif mode == "classification":
        items = normalize_classification_records(records)
    elif mode == "segmentation":
        items = normalize_segmentation_records(records)
    else:
        items = []

    cache[mode] = items
    return items


def build_flat_item_list(mode: str) -> Tuple[List[Tuple[str, int, Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], List[str]]:
    items = load_items_for_mode(mode)
    groups = group_items_by_video(items)
    order = list(groups.keys())
    flat: List[Tuple[str, int, Dict[str, Any]]] = []
    for video_id in order:
        for q_idx, item in enumerate(groups[video_id]):
            flat.append((video_id, q_idx, item))
    return flat, groups, order


# ========================= CALLBACKS =========================================

def _cb_go_next(mode: str, total: int) -> None:
    state = get_mode_state(mode)
    state["flat_idx"] = min(total - 1, state.get("flat_idx", 0) + 1)
    st.session_state.bench_mode_state[mode] = state


def _cb_go_prev(mode: str) -> None:
    state = get_mode_state(mode)
    state["flat_idx"] = max(0, state.get("flat_idx", 0) - 1)
    st.session_state.bench_mode_state[mode] = state


def _cb_select_choice(mode: str, item_id: str, choice_key: str) -> None:
    state = get_mode_state(mode)
    revealed = dict(state.get("revealed", {}))
    revealed[item_id] = choice_key
    state["revealed"] = revealed
    st.session_state.bench_mode_state[mode] = state


def _cb_reset_choice(mode: str, item_id: str) -> None:
    state = get_mode_state(mode)
    revealed = dict(state.get("revealed", {}))
    revealed[item_id] = None
    state["revealed"] = revealed
    st.session_state.bench_mode_state[mode] = state


def _cb_jump_to_question(mode: str, total: int, input_key: str) -> None:
    state = get_mode_state(mode)
    target = st.session_state.get(input_key, 1)
    target = max(1, min(target, total))
    state["flat_idx"] = target - 1
    st.session_state.bench_mode_state[mode] = state


def _cb_jump_to_video(
    mode: str,
    flat_list: List[Tuple[str, int, Dict[str, Any]]],
    video_labels: List[str],
    select_key: str,
    video_order: List[str],
) -> None:
    selected = st.session_state.get(select_key)
    if not selected or selected not in video_labels:
        return
    new_idx = video_labels.index(selected)
    if new_idx < 0 or new_idx >= len(video_order):
        return

    target_video = video_order[new_idx]
    state = get_mode_state(mode)
    for flat_idx, (video_id, _, _) in enumerate(flat_list):
        if video_id == target_video:
            state["flat_idx"] = flat_idx
            state["current_video_idx"] = new_idx
            break
    st.session_state.bench_mode_state[mode] = state


# ========================= RENDERING =========================================

def render_error_screen(assets: Dict[str, Any]) -> None:
    available_files = assets.get("mode_files", {})
    found_modes = ", ".join(MODE_LABELS[m] for m in MODE_ORDER if m in available_files) or "None"
    st.markdown(
        f"""
    <div class="error-container">
        <h2>Benchmark Configuration Required</h2>
        <p>Please provide a valid colon-bench directory.</p>
        <code>streamlit run visualize_benchmark.py -- \\
    --benchmark /path/to/data/colon-bench --mode vqa_prompted</code>
        <p style="margin-top: 1rem;">
            Expected files in benchmark dir:<br>
            <code>benchmark_vqa_prompted.json</code>,
            <code>benchmark_vqa_unprompted.json</code>,
            <code>benchmark_cls.json</code>,
            <code>benchmark_segmentation.json</code>
        </p>
        <p style="margin-top: 1rem; color: #666;">
            Modes discovered from your paths: <strong>{found_modes}</strong>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_mode_instructions(mode: str) -> None:
    instructions = {
        "vqa_prompted": (
            "Watch the clip and answer the multiple-choice question. "
            "Selected answers are revealed as correct (green) or wrong (red)."
        ),
        "vqa_unprompted": (
            "Same VQA workflow as prompted mode, but with harder clip composition and options."
        ),
        "classification": (
            "Review each clip and classify whether a lesion is present."
        ),
        "segmentation": (
            "Review segmentation targets with mask overlays. "
            "Videos and masks are streamed from the Hugging Face dataset."
        ),
    }
    text = instructions.get(mode, "")
    st.markdown(
        f"""
        <div class="instructions-box">
            <strong>{MODE_LABELS.get(mode, mode)} mode:</strong> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mode_selector(available_modes: List[str]) -> str:
    with st.sidebar:
        st.markdown("### Viewer")
        if not available_modes:
            st.warning("No benchmark mode files found.")
            return st.session_state.bench_mode

        current_mode = st.session_state.bench_mode
        if current_mode not in available_modes:
            current_mode = available_modes[0]
            st.session_state.bench_mode = current_mode

        selected = st.selectbox(
            "Mode",
            options=available_modes,
            index=available_modes.index(current_mode),
            format_func=lambda m: MODE_LABELS.get(m, m),
            key="bench_mode_select",
        )
        if selected != st.session_state.bench_mode:
            st.session_state.bench_mode = selected
        return st.session_state.bench_mode


def render_sidebar_stats(
    mode: str,
    flat_list: List[Tuple[str, int, Dict[str, Any]]],
    video_groups: Dict[str, List[Dict[str, Any]]],
    video_order: List[str],
) -> None:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Statistics")

        total_items = len(flat_list)
        total_videos = len(video_order)
        state = get_mode_state(mode)

        if mode in QUIZ_MODES:
            revealed_count = sum(1 for v in state.get("revealed", {}).values() if v is not None)
        else:
            revealed_count = len(state.get("visited", []))

        st.metric("Total Items", total_items)
        st.metric("Total Videos", total_videos)
        st.metric("Reviewed", revealed_count)

        st.markdown("---")
        st.markdown("### Progress")
        if total_items > 0:
            progress = min(1.0, float(revealed_count) / float(total_items))
            st.progress(progress)
            st.markdown(f"**{revealed_count}** of **{total_items}** reviewed")

        if total_videos > 0:
            st.markdown("---")
            video_labels = []
            for i, video_id in enumerate(video_order):
                count = len(video_groups[video_id])
                short = video_id[:16] + "..." if len(video_id) > 19 else video_id
                video_labels.append(f"Video {i + 1}: {short} ({count} items)")

            select_key = f"sidebar_video_select_{mode}"
            state_idx = state.get("current_video_idx", 0)
            state_idx = max(0, min(state_idx, len(video_labels) - 1))
            default_label = video_labels[state_idx] if video_labels else None
            if default_label and select_key not in st.session_state:
                st.session_state[select_key] = default_label

            st.selectbox(
                "Jump to video",
                video_labels,
                key=select_key,
                on_change=_cb_jump_to_video,
                args=(mode, flat_list, video_labels, select_key, video_order),
            )


def render_choice_review(item: Dict[str, Any], mode: str) -> None:
    item_id = item["item_id"]
    prompt = item.get("prompt_text", "")
    choices = item.get("choices", {})
    correct_answer = item.get("correct_answer", "")
    state = get_mode_state(mode)
    selected_choice = state.get("revealed", {}).get(item_id)

    st.markdown(
        f"""
        <div class="question-box">
            <div class="question-label">Prompt</div>
            {prompt}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if selected_choice is None:
        st.markdown("**Select your answer:**")
        for key in sorted(choices.keys()):
            st.button(
                f"{key}.  {choices[key]}",
                key=f"choice_{mode}_{item_id}_{key}",
                use_container_width=True,
                on_click=_cb_select_choice,
                args=(mode, item_id, key),
            )
        return

    st.markdown("**Your answer is revealed:**")
    for key in sorted(choices.keys()):
        text = choices[key]
        if key == correct_answer and key == selected_choice:
            css_class = "choice-correct"
            icon = "&#10004;"
        elif key == correct_answer:
            css_class = "choice-correct"
            icon = "&#10004;"
        elif key == selected_choice:
            css_class = "choice-wrong"
            icon = "&#10008;"
        else:
            css_class = "choice-dimmed"
            icon = ""

        icon_html = f'<span style="margin-left:auto; font-size:1.2rem;">{icon}</span>' if icon else ""
        st.markdown(
            f"""
            <div class="choice-btn {css_class}" style="cursor:default; display:flex; align-items:center;">
                <span class="choice-key">{key}</span>
                <span>{text}</span>
                {icon_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.button(
        "Reset answer",
        key=f"reset_{mode}_{item_id}",
        on_click=_cb_reset_choice,
        args=(mode, item_id),
    )


def render_segmentation_details(item: Dict[str, Any]) -> None:
    aux = item.get("aux", {})
    description = str(aux.get("description", "")).strip()
    masks = aux.get("masks", [])
    mask_count = len(masks)

    content = description if description else "No lesion description available."
    st.markdown(
        f"""
        <div class="question-box">
            <div class="question-label">Segmentation Description</div>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.metric("Mask Frames", mask_count)


def render_main_viewer(
    mode: str,
    flat_list: List[Tuple[str, int, Dict[str, Any]]],
    video_order: List[str],
    assets: Dict[str, Any],
) -> None:
    if not flat_list:
        mode_file = assets.get("mode_files", {}).get(mode)
        if not mode_file:
            st.warning(f"Mode file not found for {MODE_LABELS.get(mode, mode)}.")
        else:
            st.warning(f"No records found in `{os.path.basename(mode_file)}`.")
        return

    state = get_mode_state(mode)
    flat_idx = max(0, min(state.get("flat_idx", 0), len(flat_list) - 1))
    state["flat_idx"] = flat_idx
    st.session_state.bench_mode_state[mode] = state

    video_id, q_idx_in_video, item = flat_list[flat_idx]
    item_id = item.get("item_id", f"{mode}_{flat_idx}")

    if video_id in video_order:
        state["current_video_idx"] = video_order.index(video_id)
    if item_id not in state.get("visited", []):
        visited = list(state.get("visited", []))
        visited.append(item_id)
        state["visited"] = visited
        st.session_state.bench_mode_state[mode] = state

    st.markdown(f"### {MODE_LABELS.get(mode, mode)} Item **{flat_idx + 1}** of {len(flat_list)}")

    col_video, col_content = st.columns([1, 1])

    with col_video:
        st.markdown(
            f'<span class="video-badge">Video: {video_id[:32]}{"..." if len(video_id) > 32 else ""}</span>',
            unsafe_allow_html=True,
        )

        video_path: Optional[str] = None
        with st.spinner("Loading video..."):
            video_path = resolve_video_path_for_item(mode, item, assets)

        if video_path and os.path.isfile(video_path):
            video_bytes = _load_video_bytes(video_path)
            st.video(video_bytes, format="video/mp4", loop=True, autoplay=True)
        else:
            st.warning(f"Video not available for: {video_id}")

        if mode in {"vqa_prompted", "vqa_unprompted", "classification"} and flat_idx + 1 < len(flat_list):
            next_video = flat_list[flat_idx + 1][0]
            if next_video != video_id:
                videos_dir = assets.get("videos_dir")
                if videos_dir:
                    next_path = os.path.join(videos_dir, next_video)
                    if os.path.isfile(next_path):
                        _load_video_bytes(next_path)

    with col_content:
        if mode in QUIZ_MODES:
            render_choice_review(item, mode)
        elif mode == "segmentation":
            render_segmentation_details(item)

    st.markdown("---")
    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])

    with nav_left:
        st.button(
            "← Previous",
            key=f"btn_prev_{mode}",
            use_container_width=True,
            disabled=(flat_idx <= 0),
            on_click=_cb_go_prev,
            args=(mode,),
        )

    with nav_mid:
        jump_key = f"jump_question_number_{mode}"
        if jump_key not in st.session_state:
            st.session_state[jump_key] = flat_idx + 1
        st.number_input(
            "Go to item #",
            min_value=1,
            max_value=len(flat_list),
            value=flat_idx + 1,
            step=1,
            key=jump_key,
            label_visibility="collapsed",
            on_change=_cb_jump_to_question,
            args=(mode, len(flat_list), jump_key),
        )
        st.button(
            "Go",
            key=f"btn_jump_{mode}",
            use_container_width=True,
            on_click=_cb_jump_to_question,
            args=(mode, len(flat_list), jump_key),
        )
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:0.2rem;">
                <span style="font-size:0.9rem; color:#94a3b8;">
                    {flat_idx + 1} / {len(flat_list)}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_right:
        st.button(
            "Next →",
            key=f"btn_next_{mode}",
            use_container_width=True,
            disabled=(flat_idx >= len(flat_list) - 1),
            on_click=_cb_go_next,
            args=(mode, len(flat_list)),
        )


# ========================= MAIN ==============================================

def main() -> None:
    init_session_state()
    assets = st.session_state.bench_assets
    available_modes = assets.get("available_modes", [])

    st.markdown(
        '<h1 class="main-title">Colonoscopy Multi-Mode Benchmark Viewer</h1>',
        unsafe_allow_html=True,
    )

    if not available_modes:
        render_error_screen(assets)
        return

    mode = render_mode_selector(available_modes)
    render_mode_instructions(mode)

    flat_list, video_groups, video_order = build_flat_item_list(mode)
    render_sidebar_stats(mode, flat_list, video_groups, video_order)
    render_main_viewer(mode, flat_list, video_order, assets)

    components.html(BUTTON_STYLE_JS, height=0)


if __name__ == "__main__":
    main()
    # Stop execution here; a legacy duplicated block may exist below.
    # This prevents old entrypoints from running in Streamlit.
    st.stop()
"""
visualize_benchmark.py

Streamlit app for visualizing and verifying the quality of VQA benchmark
questions and answers generated by build_benchmark.py.

Displays the benchmark video alongside its multiple-choice questions.
Click an answer to reveal whether it is correct (green) or wrong (red),
with the correct answer always highlighted in green.

Usage:
    streamlit run visualize_benchmark.py -- --benchmark /path/to/colon-bench

    # Or specify paths individually:
    streamlit run visualize_benchmark.py -- \
        --benchmark-json /path/to/benchmark.json \
        --videos /path/to/videos_dir

Arguments:
    --benchmark         Path to benchmark directory (containing benchmark.json and videos/)
    --benchmark-json    Path to benchmark.json file directly (overrides --benchmark)
    --videos            Path to videos directory directly (overrides --benchmark)
"""

import streamlit as st
import streamlit.components.v1 as components
import json
import os
import sys
import argparse
from typing import Dict, List, Optional


# ========================= ARGUMENT PARSING ===================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark QA Visualization")
    parser.add_argument(
        "--benchmark", default=None,
        help="Path to benchmark directory (containing benchmark.json and videos/)"
    )
    parser.add_argument(
        "--benchmark-json", default=None, dest="benchmark_json",
        help="Path to benchmark.json file directly"
    )
    parser.add_argument(
        "--videos", default=None,
        help="Path to videos directory directly"
    )

    # Filter out streamlit-specific arguments
    known_flags = ['--benchmark', '--benchmark-json', '--videos']
    args_to_parse = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg in known_flags:
            args_to_parse.append(arg)
            if i + 1 < len(sys.argv[1:]):
                args_to_parse.append(sys.argv[i + 2])
                skip_next = True

    if not args_to_parse:
        return None

    try:
        return parser.parse_args(args_to_parse)
    except SystemExit:
        return None


ARGS = parse_args()


# ========================= PAGE CONFIG ========================================

st.set_page_config(
    page_title="Benchmark QA Viewer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS styling adapted from visualize_lesions.py
st.markdown("""
<style>
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header styling */
    .main-title {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        text-align: center;
    }

    /* Description / question box */
    .question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin: 1rem 0 1.5rem 0;
        font-size: 1.15rem;
        line-height: 1.6;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .question-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
        margin-bottom: 0.5rem;
    }

    /* Choice buttons */
    .choice-btn {
        display: block;
        width: 100%;
        padding: 14px 20px;
        margin: 8px 0;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 500;
        text-align: left;
        cursor: pointer;
        border: 2px solid #e2e8f0;
        background: #f8fafc;
        color: #1e293b;
        transition: all 0.2s ease;
        line-height: 1.5;
    }

    .choice-btn:hover {
        border-color: #667eea;
        background: #eef2ff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }

    .choice-btn .choice-key {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        border-radius: 8px;
        background: #e2e8f0;
        color: #475569;
        font-weight: 700;
        margin-right: 12px;
        font-size: 0.95rem;
    }

    /* Correct answer */
    .choice-correct {
        border-color: #16a34a !important;
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%) !important;
        color: #166534 !important;
        box-shadow: 0 4px 16px rgba(22, 163, 74, 0.2) !important;
    }

    .choice-correct .choice-key {
        background: #16a34a !important;
        color: white !important;
    }

    /* Wrong answer */
    .choice-wrong {
        border-color: #dc2626 !important;
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%) !important;
        color: #991b1b !important;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.2) !important;
    }

    .choice-wrong .choice-key {
        background: #dc2626 !important;
        color: white !important;
    }

    /* Dimmed (unselected after reveal) */
    .choice-dimmed {
        opacity: 0.5;
        border-color: #e2e8f0 !important;
        background: #f1f5f9 !important;
    }

    /* Instructions box */
    .instructions-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #475569;
    }

    .instructions-box strong {
        color: #1e293b;
    }

    /* Navigation buttons */
    .stButton > button {
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    /* Default button style */
    .stButton > button {
        background: #9ca3af !important;
        color: white !important;
        border: none !important;
    }

    .stButton > button:hover {
        background: #6b7280 !important;
    }

    .stButton > button:disabled {
        background: #d1d5db !important;
        color: #9ca3af !important;
        cursor: not-allowed;
    }

    /* Progress indicator */
    .progress-container {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 1rem;
    }

    .progress-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    /* Video styling */
    video {
        border-radius: 12px;
    }

    /* Error message styling */
    .error-container {
        text-align: center;
        padding: 60px 40px;
        background: #fff5f5;
        border-radius: 16px;
        border: 2px solid #feb2b2;
        margin: 2rem auto;
        max-width: 600px;
    }

    .error-container h2 {
        color: #c53030;
        margin-bottom: 1rem;
    }

    .error-container code {
        background: #1a1a2e;
        color: #68d391;
        padding: 1rem;
        border-radius: 8px;
        display: block;
        margin: 1rem 0;
        font-size: 0.9rem;
        text-align: left;
        white-space: pre-wrap;
    }

    /* Video info badge */
    .video-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #e2e8f0;
        color: #475569;
        margin-bottom: 0.5rem;
    }

    /* Question counter */
    .q-counter {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# JavaScript for navigation button coloring
BUTTON_STYLE_JS = """
<script>
function styleButtons() {
    const doc = window.parent.document;
    const buttons = doc.querySelectorAll('.stButton button');
    buttons.forEach(btn => {
        const text = btn.textContent.trim();
        if (text.includes('Previous') || text.includes('Next')) {
            btn.style.setProperty('background', 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', 'important');
            btn.style.setProperty('border', 'none', 'important');
        }
    });
}
setTimeout(styleButtons, 100);
const observer = new MutationObserver(() => setTimeout(styleButtons, 50));
observer.observe(window.parent.document.body, { childList: true, subtree: true });
</script>
"""


# ========================= DATA LOADING =======================================

def resolve_paths(args) -> tuple:
    """
    Resolve benchmark JSON and videos directory from arguments.

    Returns:
        (benchmark_json_path, videos_dir) or (None, None) on failure.
    """
    benchmark_json_path = None
    videos_dir = None

    if args is None:
        return None, None

    # Explicit paths take priority
    if args.benchmark_json:
        benchmark_json_path = args.benchmark_json
    if args.videos:
        videos_dir = args.videos

    # Fall back to --benchmark directory
    if args.benchmark:
        bench_dir = args.benchmark
        if benchmark_json_path is None:
            candidate = os.path.join(bench_dir, "benchmark.json")
            if os.path.isfile(candidate):
                benchmark_json_path = candidate
        if videos_dir is None:
            candidate = os.path.join(bench_dir, "videos")
            if os.path.isdir(candidate):
                videos_dir = candidate

    return benchmark_json_path, videos_dir


def load_benchmark(path: str) -> List[Dict]:
    """Load benchmark.json and return list of question dicts."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        st.error(f"Failed to load benchmark JSON: {e}")
        return []


def group_questions_by_video(questions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group questions by their video field, preserving order."""
    groups: Dict[str, List[Dict]] = {}
    order: List[str] = []
    for q in questions:
        vid = q.get("video", "")
        if vid not in groups:
            groups[vid] = []
            order.append(vid)
        groups[vid].append(q)
    # Return ordered dict
    return {v: groups[v] for v in order}


# ========================= SESSION STATE ======================================

def init_session_state():
    """Initialise session state on first run."""
    if "bench_initialized" in st.session_state:
        return

    st.session_state.bench_initialized = True
    st.session_state.bench_questions = []
    st.session_state.bench_videos_dir = ""
    st.session_state.bench_json_path = ""
    st.session_state.bench_video_groups = {}       # video -> [questions]
    st.session_state.bench_video_order = []        # ordered list of video names
    st.session_state.bench_current_video_idx = 0   # which video we are viewing
    st.session_state.bench_current_q_idx = 0       # which question within that video
    st.session_state.bench_revealed = {}            # question_id -> selected choice key (or None)
    st.session_state.bench_flat_idx = 0

    # Load data
    json_path, videos_dir = resolve_paths(ARGS)

    if json_path and os.path.isfile(json_path):
        st.session_state.bench_json_path = json_path
        st.session_state.bench_questions = load_benchmark(json_path)

    if videos_dir and os.path.isdir(videos_dir):
        st.session_state.bench_videos_dir = videos_dir

    # Build groups
    groups = group_questions_by_video(st.session_state.bench_questions)
    st.session_state.bench_video_groups = groups
    st.session_state.bench_video_order = list(groups.keys())


# ========================= NAVIGATION CALLBACKS ================================
# Using on_click callbacks instead of st.rerun() avoids explicit reruns that can
# conflict with Streamlit's internal media-file lifecycle and cause session resets.

def _cb_go_next():
    """Callback: advance to next question."""
    st.session_state.bench_flat_idx += 1

def _cb_go_prev():
    """Callback: go to previous question."""
    st.session_state.bench_flat_idx = max(0, st.session_state.bench_flat_idx - 1)

def _cb_select_choice(question_id: str, choice_key: str):
    """Callback: record the user's answer."""
    st.session_state.bench_revealed[question_id] = choice_key

def _cb_reset_choice(question_id: str):
    """Callback: clear the user's answer."""
    st.session_state.bench_revealed[question_id] = None

def _cb_jump_to_question(total: int):
    """Callback: jump to the question number entered in the number input."""
    target = st.session_state.get("jump_question_number", 1)
    # Clamp to valid range (input is 1-based, flat_idx is 0-based)
    target = max(1, min(target, total))
    st.session_state.bench_flat_idx = target - 1

def _cb_jump_to_video(flat_list: list, video_labels: list):
    """Callback: jump to the first question of the selected video."""
    selected = st.session_state.sidebar_video_select
    new_idx = video_labels.index(selected)
    if new_idx != st.session_state.bench_current_video_idx:
        target_video = st.session_state.bench_video_order[new_idx]
        fi = 0
        for vi, qi, q in flat_list:
            if vi == target_video:
                break
            fi += 1
        st.session_state.bench_flat_idx = fi
        st.session_state.bench_current_video_idx = new_idx
        st.session_state.bench_current_q_idx = 0


# ========================= FLAT NAVIGATION ====================================
# We navigate question-by-question across all videos.
# Build a flat list of (video_name, question_index_within_video, question_dict).

def build_flat_question_list() -> List[tuple]:
    """Return a flat list of (video_name, q_idx, question_dict)."""
    flat = []
    for video_name in st.session_state.bench_video_order:
        for q_idx, q in enumerate(st.session_state.bench_video_groups[video_name]):
            flat.append((video_name, q_idx, q))
    return flat


# ========================= RENDERING =========================================

def render_error_screen():
    """Show usage help when arguments are missing."""
    st.markdown("""
    <div class="error-container">
        <h2>Configuration Required</h2>
        <p>Please provide the path to the benchmark directory:</p>
        <code>streamlit run visualize_benchmark.py -- \\
    --benchmark /path/to/colon-bench</code>
        <p style="margin-top: 1.5rem; color: #666;">
            <strong>Arguments:</strong><br>
            <code>--benchmark</code> Path to benchmark directory with benchmark.json and videos/ (recommended)<br>
            <code>--benchmark-json</code> Direct path to benchmark.json (optional override)<br>
            <code>--videos</code> Direct path to videos directory (optional override)
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(flat_list: List[tuple]):
    """Render sidebar with progress statistics."""
    with st.sidebar:
        st.markdown("### Statistics")

        total_questions = len(flat_list)
        total_videos = len(st.session_state.bench_video_order)
        revealed_count = sum(
            1 for qid in st.session_state.bench_revealed
            if st.session_state.bench_revealed[qid] is not None
        )

        st.metric("Total Questions", total_questions)
        st.metric("Total Videos", total_videos)

        st.markdown("---")
        st.markdown("### Review Progress")

        if total_questions > 0:
            progress = revealed_count / total_questions
            st.progress(progress)
            st.markdown(f"**{revealed_count}** of **{total_questions}** reviewed")

        st.markdown("---")

        # Jump-to-video selector
        if total_videos > 0:
            video_labels = []
            for i, vname in enumerate(st.session_state.bench_video_order):
                n_q = len(st.session_state.bench_video_groups[vname])
                short_name = vname[:16] + "..." if len(vname) > 19 else vname
                video_labels.append(f"Video {i+1}: {short_name} ({n_q}q)")

            st.selectbox(
                "Jump to video",
                video_labels,
                index=st.session_state.bench_current_video_idx
                if st.session_state.bench_current_video_idx < len(video_labels)
                else 0,
                key="sidebar_video_select",
                on_change=_cb_jump_to_video,
                args=(flat_list, video_labels),
            )


def render_main_viewer(flat_list: List[tuple]):
    """Render the main video + question viewer."""
    if not flat_list:
        if not st.session_state.bench_json_path:
            render_error_screen()
        else:
            st.warning("No questions found in the benchmark file.")
        return

    if not st.session_state.bench_videos_dir:
        st.warning("Videos directory not found. Videos will not be displayed.")

    # Current flat index (clamped to valid range)
    flat_idx = max(0, min(st.session_state.bench_flat_idx, len(flat_list) - 1))
    st.session_state.bench_flat_idx = flat_idx

    video_name, q_idx_in_video, question = flat_list[flat_idx]
    question_id = question.get("question_id", str(flat_idx))
    correct_answer = question.get("answer", "")
    choices = question.get("choices", {})
    question_text = question.get("question", "")

    # Sync video/question indices for sidebar
    if video_name in st.session_state.bench_video_order:
        st.session_state.bench_current_video_idx = st.session_state.bench_video_order.index(video_name)
    st.session_state.bench_current_q_idx = q_idx_in_video

    # Header
    st.markdown(
        f"### Question **{flat_idx + 1}** of {len(flat_list)}"
    )

    # Layout: video on left, question + choices on right
    col_video, col_qa = st.columns([1, 1])

    # ---------- Video column ----------
    with col_video:
        st.markdown(
            f'<span class="video-badge">Video: {video_name[:32]}{"..." if len(video_name)>32 else ""}</span>',
            unsafe_allow_html=True,
        )
        video_path = os.path.join(st.session_state.bench_videos_dir, video_name)
        if os.path.isfile(video_path):
            video_bytes = _load_video_bytes(video_path)
            st.video(video_bytes, format="video/mp4", loop=True, autoplay=True)
        else:
            st.warning(f"Video file not found: {video_name}")

        # Pre-warm cache for the next question's video so Streamlit's
        # media handler already has it registered before the user clicks Next.
        if flat_idx + 1 < len(flat_list):
            next_video = flat_list[flat_idx + 1][0]
            if next_video != video_name:
                next_path = os.path.join(st.session_state.bench_videos_dir, next_video)
                if os.path.isfile(next_path):
                    _load_video_bytes(next_path)

    # ---------- Question + Choices column ----------
    with col_qa:
        # Question box
        st.markdown(
            f"""
            <div class="question-box">
                <div class="question-label">Question</div>
                {question_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Determine reveal state
        selected_choice = st.session_state.bench_revealed.get(question_id)

        if selected_choice is None:
            # Not yet answered -- render clickable buttons
            st.markdown("**Select your answer:**")
            for key in sorted(choices.keys()):
                st.button(
                    f"{key}.  {choices[key]}",
                    key=f"choice_{question_id}_{key}",
                    use_container_width=True,
                    on_click=_cb_select_choice,
                    args=(question_id, key),
                )
        else:
            # Already answered -- show colored results
            st.markdown("**Your answer is revealed:**")
            for key in sorted(choices.keys()):
                text = choices[key]
                if key == correct_answer and key == selected_choice:
                    # User picked the correct one
                    css_class = "choice-correct"
                    icon = "&#10004;"  # checkmark
                elif key == correct_answer:
                    # This is the correct answer (user picked something else)
                    css_class = "choice-correct"
                    icon = "&#10004;"
                elif key == selected_choice:
                    # User picked this wrong answer
                    css_class = "choice-wrong"
                    icon = "&#10008;"  # cross
                else:
                    # Other non-selected, non-correct choices
                    css_class = "choice-dimmed"
                    icon = ""

                icon_html = f'<span style="margin-left:auto; font-size:1.2rem;">{icon}</span>' if icon else ""

                st.markdown(
                    f"""
                    <div class="choice-btn {css_class}" style="cursor:default; display:flex; align-items:center;">
                        <span class="choice-key">{key}</span>
                        <span>{text}</span>
                        {icon_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Reset button
            st.button(
                "Reset answer",
                key=f"reset_{question_id}",
                on_click=_cb_reset_choice,
                args=(question_id,),
            )

    # ---------- Navigation ----------
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

    with nav_col1:
        st.button(
            "← Previous",
            key="btn_prev",
            use_container_width=True,
            disabled=(flat_idx <= 0),
            on_click=_cb_go_prev,
        )

    with nav_col2:
        # Jump-to-question input + progress text
        jump_left, jump_right = st.columns([3, 1])
        with jump_left:
            st.number_input(
                "Go to question #",
                min_value=1,
                max_value=len(flat_list),
                value=flat_idx + 1,
                step=1,
                key="jump_question_number",
                label_visibility="collapsed",
                on_change=_cb_jump_to_question,
                args=(len(flat_list),),
            )
        with jump_right:
            st.button(
                "Go",
                key="btn_jump",
                use_container_width=True,
                on_click=_cb_jump_to_question,
                args=(len(flat_list),),
            )
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:0.2rem;">
                <span style="font-size:0.9rem; color:#94a3b8;">
                    {flat_idx + 1} / {len(flat_list)}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col3:
        st.button(
            "Next →",
            key="btn_next",
            use_container_width=True,
            disabled=(flat_idx >= len(flat_list) - 1),
            on_click=_cb_go_next,
        )


# ========================= MAIN =============================================

def main():
    """Main application entry point."""
    init_session_state()

    st.markdown(
        '<h1 class="main-title">Colonoscopy VQA Benchmark Viewer</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="instructions-box">
            <strong>Instructions:</strong> Watch the video clip and read the question.
            Click on the answer you think is correct.
            The selected answer will turn <strong style="color:#16a34a;">green</strong> if correct
            or <strong style="color:#dc2626;">red</strong> if wrong, and the correct answer is
            always highlighted in <strong style="color:#16a34a;">green</strong>.
            Use <strong>Next / Previous</strong> to navigate between questions.
            Use the sidebar to jump to a specific video.
        </div>
        """,
        unsafe_allow_html=True,
    )

    flat_list = build_flat_question_list()

    render_sidebar(flat_list)
    render_main_viewer(flat_list)

    # Inject button styling JS
    components.html(BUTTON_STYLE_JS, height=0)


if __name__ == "__main__":
    main()
