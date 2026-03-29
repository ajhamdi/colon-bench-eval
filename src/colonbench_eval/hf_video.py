from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url


DEFAULT_DATASET_REPO_ID = "ajhamdi/colon-bench"
DEFAULT_DATASET_REVISION = "main"


def get_hf_auth_token(explicit_token: str | None = None) -> Union[str, bool]:
    """
    Return an auth token value accepted by `huggingface_hub`.

    If no explicit env token is present, `True` tells the library to use the
    locally logged-in Hugging Face credentials if available.
    """
    token = (
        explicit_token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    return token if token else True


@dataclass
class HFDatasetAssetResolver:
    """
    Resolve and download ColonBench assets from the Hugging Face dataset repo.

    `get_hf_file_metadata(...).location` lets us turn an authenticated
    `/resolve/...` URL into the final downloadable location, which may be a
    temporary CDN URL suitable for provider-side fetching.
    """

    repo_id: str = DEFAULT_DATASET_REPO_ID
    revision: str = DEFAULT_DATASET_REVISION
    token: str | None = None
    remote_url_ttl_sec: int = 20 * 60
    _remote_url_cache: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def auth_token(self) -> Union[str, bool]:
        return get_hf_auth_token(self.token)

    def _repo_path(self, relative_path: str) -> str:
        return relative_path.lstrip("/").replace("\\", "/")

    def resolve_remote_url(self, relative_path: str) -> str:
        repo_path = self._repo_path(relative_path)
        now = time.time()
        with self._lock:
            cached = self._remote_url_cache.get(repo_path)
            if cached and (now - cached[0]) < self.remote_url_ttl_sec:
                return cached[1]

        hub_url = hf_hub_url(
            repo_id=self.repo_id,
            filename=repo_path,
            repo_type="dataset",
            revision=self.revision,
        )
        metadata = get_hf_file_metadata(hub_url, token=self.auth_token)
        resolved = metadata.location or hub_url
        with self._lock:
            self._remote_url_cache[repo_path] = (now, resolved)
        return resolved

    def resolve_video_url(self, video_id: str) -> str:
        return self.resolve_remote_url(f"videos/{video_id}")

    def download_video(self, video_id: str, local_videos_dir: str | None = None) -> str:
        if local_videos_dir:
            candidate = os.path.join(local_videos_dir, video_id)
            if os.path.isfile(candidate):
                return candidate
        return hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=f"videos/{video_id}",
            revision=self.revision,
            token=self.auth_token,
        )

    def download_mask(self, mask_path: str, local_masks_dir: str | None = None) -> str:
        if os.path.isabs(mask_path) and os.path.isfile(mask_path):
            return mask_path

        repo_path = self._repo_path(mask_path)
        if local_masks_dir:
            normalized = repo_path
            if normalized.startswith("masks/"):
                normalized = normalized[len("masks/") :]
            candidate = os.path.join(local_masks_dir, normalized)
            if os.path.isfile(candidate):
                return candidate

        return hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=repo_path,
            revision=self.revision,
            token=self.auth_token,
        )
