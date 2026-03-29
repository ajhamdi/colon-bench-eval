from __future__ import annotations

import glob
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


def load_json(path: str | os.PathLike[str]) -> Any:
    """Load JSON data from *path*."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_atomic(data: Any, path: str | os.PathLike[str]) -> None:
    """Write JSON atomically via temp file + rename."""
    path_str = str(path)
    parent = os.path.dirname(path_str) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp_path, path_str)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def find_previous_results(output_dir: str | os.PathLike[str], prefix: str) -> Optional[str]:
    """Return the newest result JSON whose basename starts with *prefix*."""
    output_dir_str = str(output_dir)
    if not os.path.isdir(output_dir_str):
        return None
    pattern = os.path.join(output_dir_str, f"{prefix}*.json")
    candidates = [
        candidate
        for candidate in glob.glob(pattern)
        if not candidate.endswith("_progress.json")
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def sanitize_model_name(model: str) -> str:
    """Create a filesystem-safe model slug."""
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def ensure_dir(path: str | os.PathLike[str]) -> str:
    """Create *path* if needed and return it as a string."""
    path_str = str(path)
    os.makedirs(path_str, exist_ok=True)
    return path_str


def project_root_from_script(script_file: str | os.PathLike[str]) -> Path:
    """Resolve the release-repo root from a script path."""
    return Path(script_file).resolve().parents[1]
