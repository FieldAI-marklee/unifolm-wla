"""Per-source dataset wrapper that loads LeRobotDataset(s) and produces unified output."""

import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from .action_mapping import map_action_chunk, map_state
from .config import DatasetSourceConfig, TrainingDataConfig
from .lerobot_wrapper import LeRobotDatasetWithSelectedKeys
from .se3_utils import compute_relative_actions
from .stats_utils import (
    discover_task_dirs,
    get_normalizer,
    load_relative_stats,
    load_stats,
    normalize,
)

logger = logging.getLogger(__name__)

# System keys always loaded from parquet
SYSTEM_KEYS = {"index", "episode_index", "frame_index", "timestamp", "task_index"}

# Maps action field names to unified action slice names.
_ACTION_FIELD_TO_SLICE = {
    "ee_pose": "left_xyz_rotvec",
    "left_ee_pose": "left_xyz_rotvec",
    "right_ee_pose": "right_xyz_rotvec",
    "gripper": "left_gripper",
    "left_gripper": "left_gripper",
    "right_gripper": "right_gripper",
    "torso_joint": "torso_joint",
    "waist_joint": "waist_joint",
    "left_leg": "left_leg_joint",
    "right_leg": "right_leg_joint",
    "left_fig6d": "left_fig6d",
    "right_fig6d": "right_fig6d",
    "base_pose": "base_rotvec",
}

# Maps state field names to unified state slice names.
_STATE_FIELD_TO_SLICE = {
    "ee_pose": "left_xyz_rot6d",
    "left_ee_pose": "left_xyz_rot6d",
    "right_ee_pose": "right_xyz_rot6d",
    "gripper": "left_gripper",
    "left_gripper": "left_gripper",
    "right_gripper": "right_gripper",
    "torso_joint": "torso_joint",
    "waist_joint": "waist_joint",
    "left_leg": "left_leg_joint",
    "right_leg": "right_leg_joint",
    "left_fig6d": "left_fig6d",
    "right_fig6d": "right_fig6d",
    "base_pose": "base_rotvec",
    "base_rot": "base_rotvec",
}


class SingleSourceDataset(Dataset):
    """Wraps one dataset source (possibly multiple task sub-dirs) and produces
    unified 54-dim action and 60-dim state output."""

    def __init__(
        self,
        config: DatasetSourceConfig,
        training_config: TrainingDataConfig,
    ):
        super().__init__()
        self.config = config
        self.training_config = training_config
        self.source_fps = config.source_fps
        self.target_fps = training_config.target_fps
        self.chunk_size = training_config.chunk_size
        self.image_size = config.image_size or training_config.image_size

        # Determine which lerobot keys to load
        self.selected_keys = config.get_all_selected_keys()

        # Build delta_timestamps for loading future action frames (1 second)
        self.delta_timestamps = self._build_delta_timestamps()

        # Discover task dirs and create LeRobotDatasets
        self._datasets = []
        self._cum_lengths = []
        self._task_dirs = []
        self._load_datasets()
        self._image_shapes = self._build_image_shapes()
        self.image_signature = tuple(
            (role, *shape)
            for role, shape in sorted(self._image_shapes.items())
        )

        # Total number of frames
        cum = 0
        for ds in self._datasets:
            cum += len(ds)
            self._cum_lengths.append(cum)
        self._total_frames = cum

        # Load and merge statistics
        self._load_stats()

        # Precompute normalization parameters
        self._precompute_normalizers()
        self._action_norm_offset, self._action_norm_scale = self._build_action_norm_vectors()
        self._state_norm_offset, self._state_norm_scale = self._build_state_norm_vectors()


        logger.info(
            f"SingleSourceDataset '{config.name}': "
            f"{len(self._datasets)} sub-datasets, {self._total_frames} frames, "
            f"FPS {self.source_fps}→{self.target_fps}, image_size={self.image_size or 'original'}"
        )

    def _build_delta_timestamps(self) -> dict[str, list[float]]:
        """Build delta_timestamps to load future action/state data.

        Loads 2 seconds when B-spline resampling is enabled (for extra spline
        context), with t=0 anchor for the current frame. Otherwise 1 second.
        """
        dt = {}
        fps = self.source_fps
        future_ts = [i / fps for i in range(0, fps)]

        ak = self.config.action_keys
        for field_name in vars(ak):
            key = getattr(ak, field_name)
            if key:
                dt[key] = future_ts

        return dt

    def _load_datasets(self):
        """Create LeRobotDataset instances for each task directory."""

        data_path = Path(self.config.data_path)

        if self.config.multi_task:
            task_dirs = discover_task_dirs(data_path)
            if not task_dirs:
                raise FileNotFoundError(f"No task directories found in {data_path}")
            for td in task_dirs:
                self._task_dirs.append(td)
                # Partial mirrors list the episodes whose files are on disk; the rest must not be loaded.
                allow = td / "meta" / "train_episodes.json"
                episodes = json.loads(allow.read_text()) if allow.is_file() else None
                try:
                    ds = LeRobotDatasetWithSelectedKeys(
                        repo_id=td.name,
                        root=td,
                        episodes=episodes,
                        delta_timestamps=self.delta_timestamps,
                        tolerance_s=1.0 / self.source_fps,
                        selected_keys=self.selected_keys,
                        cache_dir=self.training_config.cache_dir,
                    )
                except Exception as e:
                    logger.error(f"Failed to load dataset for task dir {td}: {e}")
                    self._task_dirs.pop()  # remove from task_dirs if loading failed
                    continue
                self._datasets.append(ds)
        else:
            self._task_dirs.append(data_path)
            ds = LeRobotDatasetWithSelectedKeys(
                repo_id=data_path.name,
                root=data_path,
                delta_timestamps=self.delta_timestamps,
                tolerance_s=1.0 / self.source_fps,
                selected_keys=self.selected_keys,
                cache_dir=self.training_config.cache_dir,
            )
            self._datasets.append(ds)

    def _build_image_shapes(self) -> dict[str, tuple[int, int, int]]:
        """Build per-role output image shapes as (C, H, W).

        If a global image_size is configured, every image role uses that H/W.
        Otherwise the source keeps original dataset resolutions from metadata.
        """
        role_shapes: dict[str, tuple[int, int, int]] = {}

        for ik_config in self.config.image_keys:
            metadata_shapes = [
                self._feature_image_shape(ds, ik_config.key)
                for ds in self._datasets
            ]
            first_shape = metadata_shapes[0] if metadata_shapes else None
            if first_shape is None and self.image_size is None:
                raise ValueError(
                    f"Image key {ik_config.key!r} in dataset {self.config.name!r} "
                    "does not have shape metadata. Set top-level image_size or fix meta/info.json."
                )
            if self.image_size is None:
                output_shape = first_shape
                for shape in metadata_shapes[1:]:
                    if shape != output_shape:
                        raise ValueError(
                            f"Image key {ik_config.key!r} in multi-task dataset "
                            f"{self.config.name!r} has inconsistent original shapes: "
                            f"{output_shape} vs {shape}. Set top-level image_size to resize globally."
                        )
            else:
                target_h, target_w = self.image_size
                c = first_shape[0] if first_shape is not None else 3
                output_shape = (c, target_h, target_w)

            existing = role_shapes.get(ik_config.role)
            if existing is not None and existing != output_shape:
                raise ValueError(
                    f"Role {ik_config.role!r} in dataset {self.config.name!r} maps to "
                    f"multiple image shapes: {existing} vs {output_shape}."
                )
            role_shapes[ik_config.role] = output_shape

        return role_shapes

    def _feature_image_shape(self, ds, key: str) -> tuple[int, int, int] | None:
        """Return feature shape in CHW order from LeRobot metadata."""
        feature = ds.features.get(key)
        if feature is None:
            raise KeyError(
                f"Image key {key!r} is not present in dataset {self.config.name!r}. "
                f"Available keys: {sorted(ds.features)}"
            )

        shape = feature.get("shape")
        if shape is not None:
            return self._shape_to_chw(shape)

        info = feature.get("info") or {}
        h = info.get("video.height") or info.get("height")
        w = info.get("video.width") or info.get("width")
        if h is not None and w is not None:
            return (3, int(h), int(w))
        return None

    def _shape_to_chw(self, shape) -> tuple[int, int, int]:
        shape = tuple(int(x) for x in shape)
        if len(shape) == 3:
            if shape[0] in (1, 3):
                c, h, w = shape
            elif shape[-1] in (1, 3):
                h, w, c = shape
            else:
                raise ValueError(
                    f"Cannot infer channel dimension from image shape {shape} "
                    f"in dataset {self.config.name!r}"
                )
            return c, h, w
        if len(shape) == 2:
            h, w = shape
            return 3, h, w
        raise ValueError(
            f"Expected image shape with 2 or 3 dimensions, got {shape} "
            f"in dataset {self.config.name!r}"
        )

    def _load_stats(self):
        """Load pre-merged stats.json / relative_stats.json for this source.

        Only the `precollected_stats_path` fast path is supported now (the
        on-the-fly per-task / cross-variant stat merging lived in the removed
        multi_source_dataset.py). Every dataset config must set
        `precollected_stats_path` to a directory containing both files.
        """
        if not self.config.precollected_stats_path:
            raise ValueError(
                f"Dataset {self.config.name!r} has no precollected_stats_path set. "
                "Only pre-collected stats are supported; compute stats.json and "
                "relative_stats.json ahead of time and point precollected_stats_path at them."
            )
        pc = Path(self.config.precollected_stats_path)
        stats_file = pc / "stats.json"
        rel_file = pc / "relative_stats.json"
        if not stats_file.exists() or not rel_file.exists():
            raise FileNotFoundError(
                f"precollected_stats_path {pc} must contain both stats.json and "
                f"relative_stats.json (got stats={stats_file.exists()}, "
                f"relative_stats={rel_file.exists()})"
            )
        self.data_stats = load_stats(stats_file)
        self.relative_stats = load_relative_stats(rel_file)
        logger.info(
            f"[{self.config.name}] using pre-collected stats from {pc}"
        )

    def _precompute_normalizers(self):
        """Precompute normalization offsets and scales."""
        norm_type = self.training_config.norm_type
        rel_norm_type = self.training_config.rel_norm_type or norm_type
        state_norm_type = self.training_config.state_norm_type or norm_type
        gripper_norm_type = self.training_config.gripper_norm_type or norm_type
        rs_map = self.config.relative_stats_key_map

        self.action_normalizers = {}

        # EE action normalizers (from relative_stats)
        for field_name in ["ee_pose", "left_ee_pose", "right_ee_pose"]:
            rs_key = getattr(rs_map, field_name, None)
            if rs_key and rs_key in self.relative_stats:
                offset, scale = get_normalizer(self.relative_stats, rs_key, rel_norm_type)
                self.action_normalizers[field_name] = (offset, scale)

        # Non-EE action normalizers (from stats.json)
        ak = self.config.action_keys
        for field_name in ["base_command", "waist_joint", "torso_joint", "left_leg", "right_leg"]:
            key = getattr(ak, field_name, None)
            if key and key in self.data_stats:
                offset, scale = get_normalizer(self.data_stats, key, norm_type)
                self.action_normalizers[field_name] = (offset, scale)

        for field_name in ["gripper", "left_gripper", "right_gripper"]:
            key = getattr(ak, field_name, None)
            if key and key in self.data_stats:
                offset, scale = get_normalizer(self.data_stats, key, gripper_norm_type)
                self.action_normalizers[field_name] = (offset, scale)

        # Dex hand: pass-through joints, normalize from data_stats
        for field_name in ["left_fig6d", "right_fig6d"]:
            key = getattr(ak, field_name, None)
            if key and key in self.data_stats:
                self.action_normalizers[field_name] = get_normalizer(self.data_stats, key, gripper_norm_type)

        # Base pose: relative, normalize from relative_stats (same pattern as EE poses)
        rs_key = getattr(rs_map, "base_pose", None)
        if rs_key and rs_key in self.relative_stats:
            self.action_normalizers["base_pose"] = get_normalizer(self.relative_stats, rs_key, rel_norm_type)

        # State normalizers (from stats.json)
        self.state_normalizers = {}
        sk = self.config.state_keys
        for field_name in vars(sk):
            key = getattr(sk, field_name, None)
            if key and key in self.data_stats:
                offset, scale = get_normalizer(self.data_stats, key, state_norm_type)
                self.state_normalizers[field_name] = (offset, scale)

        #(TODO) HACK: remove base_pose from state, reset the normalizer to 0/1, since we don't want to normalize it for now
        if "base_pose" in self.state_normalizers:
            self.state_normalizers["base_pose"] = (np.zeros_like(self.state_normalizers["base_pose"][0]), np.ones_like(self.state_normalizers["base_pose"][1]))

        # base_rot = [gravity_xyz(3), omega_xyz(3)]. Gravity is a unit-vector
        # projection already in [-1, 1] -- skip normalizing it; only omega uses q01/q99.
        if "base_rot" in self.state_normalizers:
            offset, scale = self.state_normalizers["base_rot"]
            offset = np.array(offset, dtype=np.float32).copy()
            scale = np.array(scale, dtype=np.float32).copy()
            offset[:3] = 0.0
            scale[:3] = 1.0
            self.state_normalizers["base_rot"] = (offset, scale)

    def _build_action_norm_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Build 54-dim offset and scale vectors from action_normalizers.

        Used so each dataset item can carry the normalization parameters that
        were applied to it (for downstream denormalization).
        Returns:
            offset: (54,) float32, default 0
            scale: (54,) float32, default 1
        """
        from .action_mapping import SLICES, UNIFIED_DIM

        offset = np.zeros(UNIFIED_DIM, dtype=np.float32)
        scale = np.ones(UNIFIED_DIM, dtype=np.float32)

        for field_name, (o, s) in self.action_normalizers.items():
            if field_name in _ACTION_FIELD_TO_SLICE:
                slice_name = _ACTION_FIELD_TO_SLICE[field_name]
                sl = SLICES[slice_name]
                o_arr = np.atleast_1d(np.asarray(o, dtype=np.float32))
                s_arr = np.atleast_1d(np.asarray(s, dtype=np.float32))
                n = min(sl.stop - sl.start, len(o_arr))
                offset[sl.start:sl.start + n] = o_arr[:n]
                scale[sl.start:sl.start + n] = s_arr[:n]

            elif field_name == "base_command" and self.config.base_command_dims is not None:
                dims = self.config.base_command_dims
                o_arr = np.atleast_1d(np.asarray(o, dtype=np.float32))
                s_arr = np.atleast_1d(np.asarray(s, dtype=np.float32))
                if "vx" in dims and "vy" in dims:
                    sl = SLICES["base_vx_vy"]
                    for dest, src_dim in enumerate([dims["vx"], dims["vy"]]):
                        if src_dim < len(o_arr):
                            offset[sl.start + dest] = o_arr[src_dim]
                            scale[sl.start + dest] = s_arr[src_dim]
                vw_key = "vyaw" if "vyaw" in dims else ("vw" if "vw" in dims else None)
                if vw_key:
                    sl = SLICES["base_vw"]
                    d = dims[vw_key]
                    if d < len(o_arr):
                        offset[sl.start] = o_arr[d]
                        scale[sl.start] = s_arr[d]
                if "height" in dims:
                    sl = SLICES["height"]
                    d = dims["height"]
                    if d < len(o_arr):
                        offset[sl.start] = o_arr[d]
                        scale[sl.start] = s_arr[d]

        return offset, scale

    def _build_state_norm_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Build 60-dim offset and scale vectors from state_normalizers.

        Mirrors _build_action_norm_vectors but for the unified state space.
        EE-pose fields (left_ee_pose, right_ee_pose, etc.) only normalize xyz
        (first 3 dims of each 9-dim rot6d slot) — same logic as
        _normalize_state_unified.
        Returns:
            offset: (60,) float32, default 0
            scale:  (60,) float32, default 1
        """
        from .action_mapping import STATE_DIM, STATE_SLICES

        offset = np.zeros(STATE_DIM, dtype=np.float32)
        scale = np.ones(STATE_DIM, dtype=np.float32)

        for field_name, (o, s) in self.state_normalizers.items():
            slice_name = _STATE_FIELD_TO_SLICE.get(field_name)
            if not slice_name or slice_name not in STATE_SLICES:
                continue
            sl = STATE_SLICES[slice_name]
            o_arr = np.atleast_1d(np.asarray(o, dtype=np.float32))
            s_arr = np.atleast_1d(np.asarray(s, dtype=np.float32))
            if field_name in {"ee_pose", "left_ee_pose", "right_ee_pose", "base_pose"}:
                n = min(3, len(o_arr), sl.stop - sl.start)
            else:
                n = min(sl.stop - sl.start, len(o_arr))
            offset[sl.start:sl.start + n] = o_arr[:n]
            scale[sl.start:sl.start + n] = s_arr[:n]

        return offset, scale

    def _route_index(self, idx: int) -> tuple[int, int]:
        """Map global index to (sub_dataset_index, local_index)."""
        for i, cum_len in enumerate(self._cum_lengths):
            if idx < cum_len:
                local = idx - (self._cum_lengths[i - 1] if i > 0 else 0)
                return i, local
        raise IndexError(f"Index {idx} out of range [0, {self._total_frames})")

    def __len__(self):
        return self._total_frames

    def _preprocess_item(self, item: dict) -> dict:
        """Dataset-specific raw item hook before generic extraction."""
        return item

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_idx = self._route_index(idx)
        ds = self._datasets[ds_idx]
        item = ds[local_idx]
        item = self._preprocess_item(item)

        # 1. Extract current state
        state_dict = self._extract_state(item)
        state_unified, state_mask = map_state(self.config, state_dict)
        state_unified_unnorm = state_unified.copy()
        state_unified = self._normalize_state_unified(state_unified)

        # 2. Extract and compute relative EE actions
        rel_ee_chunk, left_rel_ee_chunk, right_rel_ee_chunk = self._compute_relative_ee(item)

        # 3. Extract non-EE action chunks
        gripper_chunk, left_gripper_chunk, right_gripper_chunk = self._extract_gripper_chunks(item)
        base_command_chunk = self._extract_action_chunk(item, "base_command")
        torso_joint_chunk = self._extract_action_chunk(item, "torso_joint")
        waist_joint_chunk = self._extract_action_chunk(item, "waist_joint")
        left_leg_chunk = self._extract_action_chunk(item, "left_leg")
        right_leg_chunk = self._extract_action_chunk(item, "right_leg")
        left_fig6d_chunk = self._extract_action_chunk(item, "left_fig6d")
        right_fig6d_chunk = self._extract_action_chunk(item, "right_fig6d")
        base_pose_rel_chunk = self._compute_relative_base_pose(item)

        # 4. Normalize action chunks
        rel_ee_chunk = self._normalize_chunk(rel_ee_chunk, "ee_pose")
        left_rel_ee_chunk = self._normalize_chunk(left_rel_ee_chunk, "left_ee_pose")
        right_rel_ee_chunk = self._normalize_chunk(right_rel_ee_chunk, "right_ee_pose")
        gripper_chunk = self._normalize_chunk(gripper_chunk, "gripper")
        left_gripper_chunk = self._normalize_chunk(left_gripper_chunk, "left_gripper")
        right_gripper_chunk = self._normalize_chunk(right_gripper_chunk, "right_gripper")
        base_command_chunk = self._normalize_chunk(base_command_chunk, "base_command")
        torso_joint_chunk = self._normalize_chunk(torso_joint_chunk, "torso_joint")
        waist_joint_chunk = self._normalize_chunk(waist_joint_chunk, "waist_joint")
        left_leg_chunk = self._normalize_chunk(left_leg_chunk, "left_leg")
        right_leg_chunk = self._normalize_chunk(right_leg_chunk, "right_leg")
        left_fig6d_chunk = self._normalize_chunk(left_fig6d_chunk, "left_fig6d")
        right_fig6d_chunk = self._normalize_chunk(right_fig6d_chunk, "right_fig6d")
        base_pose_rel_chunk = self._normalize_chunk(base_pose_rel_chunk, "base_pose")

        # 4b. Binarize gripper in normalized space (if enabled)
        if self.training_config.binarize_gripper:
            gripper_chunk = self._binarize_gripper_chunk(gripper_chunk) if gripper_chunk is not None else None
            left_gripper_chunk = self._binarize_gripper_chunk(left_gripper_chunk) if left_gripper_chunk is not None else None
            right_gripper_chunk = self._binarize_gripper_chunk(right_gripper_chunk) if right_gripper_chunk is not None else None

        # 6. Map to unified representation
        action_unified, action_mask = map_action_chunk(
            self.config,
            rel_ee_chunk, left_rel_ee_chunk, right_rel_ee_chunk,
            gripper_chunk, left_gripper_chunk, right_gripper_chunk,
            base_command_chunk, torso_joint_chunk, waist_joint_chunk,
            left_leg_chunk, right_leg_chunk,
            left_fig6d_chunk=left_fig6d_chunk,
            right_fig6d_chunk=right_fig6d_chunk,
            base_pose_rel_chunk=base_pose_rel_chunk,
        )

        # 7. Process images
        images, image_mask = self._process_images(item)

        # 8. Get task string
        task = random.choice(item.get("task", "").split("@")) if item.get("task") else ""

        return {
            "images": images,
            "image_mask": image_mask,
            "action": torch.from_numpy(action_unified),      # (chunk_size, 54)
            "action_mask": torch.from_numpy(action_mask),     # (54,)
            "state": torch.from_numpy(state_unified),         # (60,)
            "state_unnorm": torch.from_numpy(state_unified_unnorm),  # (60,)
            "state_mask": torch.from_numpy(state_mask),       # (60,)
            "action_norm_offset": torch.from_numpy(self._action_norm_offset),  # (54,)
            "action_norm_scale": torch.from_numpy(self._action_norm_scale),    # (54,)
            "task": task,
            "arm_type": self.config.arm_type,
            "robot_type": self.config.robot_type,
            "embodiment": self.config.embodiment or self.config.robot_type,
            "source": self.config.name,
        }

    def _to_numpy(self, val) -> np.ndarray:
        """Convert torch tensor or scalar to numpy array, ensuring at least 1D."""
        if isinstance(val, torch.Tensor):
            val = val.float().numpy()
        val = np.asarray(val, dtype=np.float32)
        if val.ndim == 0:
            val = val.reshape(1)
        return val

    def _extract_chunk_from_item(self, item: dict, key: str) -> np.ndarray | None:
        """Extract a chunk of values (T timesteps) from a delta_timestamps item.

        For delta_timestamps, lerobot returns shape (T, D) for multi-dim features
        or (T,) for scalar features. We normalize to (T, D).
        """
        if key not in item:
            return None
        val = self._to_numpy(item[key])
        if val.ndim == 1:
            # Scalar feature with T timesteps: (T,) → (T, 1)
            val = val[:, np.newaxis]
        return val

    def _extract_state(self, item: dict) -> dict[str, np.ndarray]:
        """Extract current state values (not future, just current frame)."""
        sk = self.config.state_keys
        state_dict = {}
        for field_name in vars(sk):
            key = getattr(sk, field_name, None)
            if key and key in item:
                val = self._to_numpy(item[key])
                # If key had delta_timestamps on it (shouldn't for state), take first row
                if val.ndim > 1:
                    val = val[0]
                state_dict[field_name] = val
        return state_dict

    def _compute_relative_ee(self, item: dict):
        """Compute relative EE actions from current state and future action poses."""
        ak = self.config.action_keys
        sk = self.config.state_keys
        ee_format = self.config.ee_format

        rel_ee = None
        left_rel_ee = None
        right_rel_ee = None

        if self.config.arm_type == "single_left":
            state_key = sk.ee_pose
            action_key = ak.ee_pose
            if state_key and action_key and state_key in item and action_key in item:
                curr_state = self._to_numpy(item[state_key])
                if curr_state.ndim > 1:
                    curr_state = curr_state[0]
                future_actions = self._extract_chunk_from_item(item, action_key)
                if future_actions is not None:
                    rel_ee = compute_relative_actions(curr_state, future_actions, ee_format)

        elif self.config.arm_type in ("dual", "dual_with_legs"):
            # Left arm
            if sk.left_ee_pose and ak.left_ee_pose:
                if sk.left_ee_pose in item and ak.left_ee_pose in item:
                    curr_left = self._to_numpy(item[sk.left_ee_pose])
                    if curr_left.ndim > 1:
                        curr_left = curr_left[0]
                    future_left = self._extract_chunk_from_item(item, ak.left_ee_pose)
                    if future_left is not None:
                        left_rel_ee = compute_relative_actions(curr_left, future_left, ee_format)

            # Right arm
            if sk.right_ee_pose and ak.right_ee_pose:
                if sk.right_ee_pose in item and ak.right_ee_pose in item:
                    curr_right = self._to_numpy(item[sk.right_ee_pose])
                    if curr_right.ndim > 1:
                        curr_right = curr_right[0]
                    future_right = self._extract_chunk_from_item(item, ak.right_ee_pose)
                    if future_right is not None:
                        right_rel_ee = compute_relative_actions(curr_right, future_right, ee_format)

        return rel_ee, left_rel_ee, right_rel_ee

    def _compute_relative_base_pose(self, item: dict) -> np.ndarray | None:
        """Compute relative base pose action: T_rel = T_curr^-1 @ T_future (xyz_quat format)."""
        ak = self.config.action_keys
        sk = self.config.state_keys
        if not (ak.base_pose and sk.base_pose):
            return None
        if ak.base_pose not in item or sk.base_pose not in item:
            return None
        curr = self._to_numpy(item[sk.base_pose])
        if curr.ndim > 1:
            curr = curr[0]
        future = self._extract_chunk_from_item(item, ak.base_pose)
        if future is None:
            return None
        return compute_relative_actions(curr, future, "xyz_quat")

    def _binarize_gripper_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Convert normalized gripper actions to binary 0/1.

        Applied after normalization so thresholds are consistent across robots
        regardless of their raw gripper range. In normalized space (approx [-1, 1]):
        - values >  0.9 are treated as open  → relabeled 1.0
        - values < -0.9 are treated as closed → relabeled 0.0
        - in-between: relabeled based on next non-in-between state.
        Edge case: trajectory ends on an in-between value → carry the last value.
        """
        vals = chunk[:, 0] if chunk.ndim == 2 else chunk  # (T,)
        open_mask = vals > 0.9
        in_between_mask = ~(open_mask | (vals < -0.9))
        result = np.empty_like(vals)
        carry = 1.0 if vals[-1] > 0.0 else 0.0
        for i in reversed(range(len(vals))):
            if not in_between_mask[i]:
                carry = 1.0 if open_mask[i] else 0.0
            result[i] = carry
        return result.reshape(chunk.shape)

    def _extract_gripper_chunks(self, item: dict):
        """Extract gripper action chunks."""
        ak = self.config.action_keys
        gripper = self._extract_chunk_from_item(item, ak.gripper) if ak.gripper else None
        left_gripper = self._extract_chunk_from_item(item, ak.left_gripper) if ak.left_gripper else None
        right_gripper = self._extract_chunk_from_item(item, ak.right_gripper) if ak.right_gripper else None
        return gripper, left_gripper, right_gripper

    def _extract_action_chunk(self, item: dict, field_name: str) -> np.ndarray | None:
        """Extract a non-EE action chunk by config field name."""
        key = getattr(self.config.action_keys, field_name, None)
        if key:
            return self._extract_chunk_from_item(item, key)
        return None

    def _normalize_chunk(self, chunk: np.ndarray | None, field_name: str) -> np.ndarray | None:
        """Normalize an action chunk using precomputed normalizers."""
        if chunk is None:
            return None
        if field_name in self.action_normalizers:
            offset, scale = self.action_normalizers[field_name]
            return normalize(chunk, offset, scale)
        return chunk

    def _normalize_state_unified(self, state: np.ndarray) -> np.ndarray:
        """Normalize the unified state vector field by field."""
        from .action_mapping import STATE_SLICES

        for field_name, (offset, scale) in self.state_normalizers.items():
            slice_name = _STATE_FIELD_TO_SLICE.get(field_name)
            if slice_name and slice_name in STATE_SLICES:
                s = STATE_SLICES[slice_name]
                o = np.atleast_1d(offset)
                sc = np.atleast_1d(scale)

                if field_name in {"ee_pose", "left_ee_pose", "right_ee_pose", "base_pose"}:
                    # Raw pose stats may be rpy/quat/rvec, while state stores
                    # rot6d for EE and rotvec for base. Normalize xyz only.
                    n = min(3, len(o), s.stop - s.start)
                    state[s.start:s.start + n] = normalize(
                        state[s.start:s.start + n], o[:n], sc[:n]
                    )
                    continue

                n = min(s.stop - s.start, len(o))
                state[s.start:s.start + n] = normalize(
                    state[s.start:s.start + n], o[:n], sc[:n]
                )

        return state

    def _process_images(self, item: dict) -> tuple[dict, dict]:
        """Extract and resize images, mapping to role-based keys."""
        images = {}
        image_mask = {}
        target_size = self.image_size

        for ik_config in self.config.image_keys:
            role = ik_config.role
            key = ik_config.key
            expected_shape = self._image_shapes[role]

            if key in item and item[key] is not None:
                img = item[key]
                if isinstance(img, torch.Tensor):
                    # img shape: (C, H, W) from lerobot
                    if target_size is not None:
                        target_h, target_w = target_size
                        img = TF.resize(img, [target_h, target_w], antialias=True)
                    if tuple(img.shape[-3:]) != expected_shape:
                        raise ValueError(
                            f"Image key {key!r} in dataset {self.config.name!r} "
                            f"returned shape {tuple(img.shape[-3:])}, expected {expected_shape}."
                        )
                    images[role] = img
                    image_mask[role] = True
                else:
                    images[role] = torch.zeros(expected_shape)
                    image_mask[role] = False
            else:
                images[role] = torch.zeros(expected_shape)
                image_mask[role] = False

        return images, image_mask


class UnitreeSingleSourceDataset(SingleSourceDataset):
    """Unitree humanoid dataset source (no morphological-symmetry augmentation port)."""


DATASET_REGISTRY = {
    "unitree": UnitreeSingleSourceDataset,
}


def create_single_source_dataset(
    config: DatasetSourceConfig,
    training_config: TrainingDataConfig,
) -> SingleSourceDataset:
    dataset_cls = DATASET_REGISTRY.get(config.robot_type, SingleSourceDataset)
    return dataset_cls(config, training_config)
