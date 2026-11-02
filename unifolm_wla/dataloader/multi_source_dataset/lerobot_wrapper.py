"""Wrapper adding selected_keys support to LeRobotDataset without modifying lerobot source."""

import hashlib
import json
import logging
import os
import shutil
import sys
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import datasets
import torch
from datasets import load_dataset as hf_load_dataset
import datasets.config as hf_datasets_config
import pyarrow.dataset as pa_ds

from .distributed_utils import safe_ddp_context

import lerobot.datasets.video_utils as lerobot_video_utils
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import get_hf_features_from_features, hf_transform_to_torch

_SYSTEM_KEYS = {"index", "episode_index", "frame_index", "timestamp", "task_index", "subtask_index"}
logger = logging.getLogger(__name__)


class BoundedVideoDecoderCache(lerobot_video_utils.VideoDecoderCache):
    """LRU-bounded drop-in for lerobot's VideoDecoderCache.

    lerobot keeps one torchcodec decoder (+ open file) per video path forever, in every dataloader
    worker. With one video per episode and camera (HIW: 2610 files) that grows each worker by GBs
    until the host OOMs; this evicts the least recently used decoder beyond `max_size`.
    """

    def __init__(self, max_size: int):
        super().__init__()
        self._cache = OrderedDict()
        self.max_size = max_size

    def get_decoder(self, video_path: str):
        from torchcodec.decoders import VideoDecoder

        video_path = str(video_path)
        with self._lock:
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
            else:
                file_handle = lerobot_video_utils.fsspec.open(video_path).__enter__()
                self._cache[video_path] = (VideoDecoder(file_handle, seek_mode="approximate"), file_handle)
                while len(self._cache) > self.max_size:
                    _, (_, old_handle) = self._cache.popitem(last=False)
                    old_handle.close()
            return self._cache[video_path][0]


# decode_video_frames_torchcodec reads this module global at call time.
lerobot_video_utils._default_decoder_cache = BoundedVideoDecoderCache(
    int(os.environ.get("UNIFOLM_WLA_VIDEO_DECODER_CACHE", "96"))
)


@contextmanager
def _temporary_hf_datasets_cache(cache_dir: str | None):
    """Route HuggingFace datasets metadata caches to the configured cache_dir."""
    if not cache_dir:
        yield
        return

    cache_path = Path(cache_dir) / "hf_datasets"
    cache_path.mkdir(parents=True, exist_ok=True)

    old_env = os.environ.get("HF_DATASETS_CACHE")
    old_config = hf_datasets_config.HF_DATASETS_CACHE
    os.environ["HF_DATASETS_CACHE"] = str(cache_path)
    hf_datasets_config.HF_DATASETS_CACHE = str(cache_path)
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop("HF_DATASETS_CACHE", None)
        else:
            os.environ["HF_DATASETS_CACHE"] = old_env
        hf_datasets_config.HF_DATASETS_CACHE = old_config


class LeRobotDatasetWithSelectedKeys(LeRobotDataset):
    """LeRobotDataset subclass with selective key loading.

    Adds a `selected_keys` parameter that restricts which features are loaded
    from parquet and video files. System keys (index, timestamp, etc.) are
    always loaded.

    Usage:
        ds = LeRobotDatasetWithSelectedKeys(
            repo_id=..., root=..., delta_timestamps=...,
            selected_keys=["observation.images.cam_high", "action.left_ee_pose"]
        )
    """

    def __init__(self, *args, selected_keys: list[str] | None = None, cache_dir: str | None = None, **kwargs):
        # Store BEFORE super().__init__() because load_hf_dataset()
        # is called eagerly inside parent __init__.
        self._selected_keys = selected_keys
        self._cache_dir = cache_dir
        super().__init__(*args, **kwargs)

    @property
    def features(self) -> dict[str, dict]:
        if self._selected_keys is None:
            return self.meta.features
        allowed = set(self._selected_keys) | _SYSTEM_KEYS
        return {k: v for k, v in self.meta.features.items() if k in allowed}

    @property
    def active_video_keys(self) -> list[str]:
        return [k for k, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def active_camera_keys(self) -> list[str]:
        return [k for k, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    def _check_cached_episodes_sufficient(self) -> bool:
        """Like LeRobotDataset, but only the selected cameras must be on disk.

        A partial mirror drops IR and rectified copies. The parent check treats those
        missing files as a cache miss and then tries to download `repo_id` from the Hub.
        """
        if self._selected_keys is None:
            return super()._check_cached_episodes_sufficient()
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False
        available = {
            ep.item() if hasattr(ep, "item") else ep
            for ep in self.hf_dataset.unique("episode_index")
        }
        requested = set(range(self.meta.total_episodes)) if self.episodes is None else set(self.episodes)
        if not requested.issubset(available):
            return False
        for ep_idx in requested:
            for vid_key in self.active_video_keys:
                if not (self.root / self.meta.get_video_file_path(ep_idx, vid_key)).exists():
                    return False
        return True

    def _get_query_timestamps(self, current_ts: float, query_indices: dict[str, list[int]] | None = None) -> dict[str, list[float]]:
        if self._selected_keys is None:
            return super()._get_query_timestamps(current_ts, query_indices)
        query_timestamps = {}
        for key in self.active_video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _get_arrow_cache_key(self, columns: list[str] | None) -> str | None:
        """Return a deterministic Arrow cache key, or None if cache_dir is not set."""
        if not self._cache_dir:
            return None
        key_data = {
            "root": str(self.root),
            "columns": sorted(columns or []),
            "episodes": sorted(self.episodes) if self.episodes is not None else None,
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _get_arrow_cache_path(self, columns: list[str] | None) -> Path | None:
        cache_key = self._get_arrow_cache_key(columns)
        if cache_key is None:
            return None
        return Path(self._cache_dir) / "arrow_cache" / cache_key

    def _get_arrow_cache_marker_path(self, arrow_cache_path: Path) -> Path:
        return arrow_cache_path.parent / f"{arrow_cache_path.name}.complete"

    def _is_arrow_cache_complete(self, arrow_cache_path: Path | None) -> bool:
        if arrow_cache_path is None:
            return False
        marker_path = self._get_arrow_cache_marker_path(arrow_cache_path)
        return arrow_cache_path.is_dir() and marker_path.is_file()

    def _load_arrow_cache(self, arrow_cache_path: Path) -> datasets.Dataset:
        hf_dataset = datasets.load_from_disk(str(arrow_cache_path))
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _remove_incomplete_arrow_cache(self, arrow_cache_path: Path):
        marker_path = self._get_arrow_cache_marker_path(arrow_cache_path)
        if marker_path.exists() and arrow_cache_path.is_dir():
            return
        if marker_path.exists():
            marker_path.unlink()
        if arrow_cache_path.is_dir():
            logger.info("Removing incomplete Arrow cache: %s", arrow_cache_path)
            shutil.rmtree(arrow_cache_path)
        elif arrow_cache_path.exists():
            arrow_cache_path.unlink()

    def _save_arrow_cache_atomic(self, hf_dataset: datasets.Dataset, arrow_cache_path: Path):
        arrow_cache_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path = self._get_arrow_cache_marker_path(arrow_cache_path)
        tmp_path = arrow_cache_path.parent / f"{arrow_cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"

        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        try:
            hf_dataset.save_to_disk(str(tmp_path))
            os.rename(tmp_path, arrow_cache_path)
            marker_path.write_text("complete\n")
        finally:
            if tmp_path.exists():
                shutil.rmtree(tmp_path)

    def load_hf_dataset(self) -> datasets.Dataset:
        parquet_features = {
            k: v for k, v in self.features.items()
            if v["dtype"] not in ["video", "image"]
        }
        hf_features = get_hf_features_from_features(parquet_features)
        columns = list(parquet_features.keys()) if self._selected_keys is not None else None

        pq_dir = self.root / "data"
        paths = sorted(pq_dir.glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {pq_dir}")

        filters = pa_ds.field("episode_index").isin(self.episodes) if self.episodes is not None else None

        arrow_cache_path = self._get_arrow_cache_path(columns)
        cache_key = self._get_arrow_cache_key(columns)
        hash_id = f"lerobot-hf-cache:{cache_key}" if cache_key is not None else str(self.root)

        # Fallback when no cache_dir is configured: rely on safe_ddp_context's
        # serialization alone (works for small clusters; on CPFS at >16 ranks
        # the read-path FileLock may still race).
        if arrow_cache_path is None:
            with safe_ddp_context(hash_id, use_barrier=True):
                hf_dataset = hf_load_dataset(
                    "parquet",
                    data_files=[str(p) for p in paths],
                    split="train",
                    filters=filters,
                    features=hf_features,
                    columns=columns,
                )
            hf_dataset.set_transform(hf_transform_to_torch)
            return hf_dataset

        # Build once on global master via safe_ddp_context (tier-2/3 ranks see
        # the marker file and skip the build), then every rank reads the Arrow
        # snapshot with load_from_disk. load_from_disk does not use FileLock,
        # which avoids HF datasets' CPFS stale-dentry race on the read path.
        with safe_ddp_context(hash_id, use_barrier=True):
            if not self._is_arrow_cache_complete(arrow_cache_path):
                self._remove_incomplete_arrow_cache(arrow_cache_path)
                hf_dataset = hf_load_dataset(
                    "parquet",
                    data_files=[str(p) for p in paths],
                    split="train",
                    filters=filters,
                    features=hf_features,
                    cache_dir=self._cache_dir,
                    columns=columns,
                )
                self._save_arrow_cache_atomic(hf_dataset, arrow_cache_path)

        return self._load_arrow_cache(arrow_cache_path)

    def __getitem__(self, idx) -> dict:
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        video_keys = self.active_video_keys
        camera_keys = self.active_camera_keys

        if video_keys:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {k: v for k, v in video_frames.items() if k in set(video_keys)} | item

        if self.image_transforms is not None:
            for cam in camera_keys:
                if cam in item:
                    item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name

        if "subtask_index" in self.features and self.meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self.meta.subtasks.iloc[subtask_idx].name

        return item
