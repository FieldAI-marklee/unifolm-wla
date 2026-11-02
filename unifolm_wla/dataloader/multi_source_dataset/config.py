from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ImageKeyConfig:
    key: str         # lerobot key, e.g. "observation.images.image_0"
    role: str        # semantic role, e.g. "cam_high", "cam_wrist_left"
    # Deprecated compatibility field. Image resizing is controlled only by
    # TrainingDataConfig.image_size; None means use original dataset resolution.
    resize: Optional[list[int]] = None


@dataclass
class ActionKeysConfig:
    """Maps dataset-specific action parquet keys to semantic roles."""
    # EE pose keys (will undergo relative action computation)
    ee_pose: Optional[str] = None                 # single-arm EE pose key
    left_ee_pose: Optional[str] = None            # dual-arm left EE
    right_ee_pose: Optional[str] = None           # dual-arm right EE
    # Gripper keys (absolute, no relative computation)
    gripper: Optional[str] = None                 # single-arm gripper
    left_gripper: Optional[str] = None
    right_gripper: Optional[str] = None
    # Dex hand keys (6-DOF each, pass-through)
    left_fig6d: Optional[str] = None
    right_fig6d: Optional[str] = None
    # Base pose (xyz+quat absolute; relative computed at training time)
    base_pose: Optional[str] = None
    # Other action keys
    base_command: Optional[str] = None            # base velocity
    waist_joint: Optional[str] = None             # waist joint action
    torso_joint: Optional[str] = None             # torso joint action
    left_leg: Optional[str] = None
    right_leg: Optional[str] = None

    def get_all_keys(self) -> list[str]:
        """Return all non-None action keys."""
        return [v for v in vars(self).values() if v is not None]


@dataclass
class StateKeysConfig:
    """Maps dataset-specific state parquet keys to semantic roles."""
    ee_pose: Optional[str] = None
    left_ee_pose: Optional[str] = None
    right_ee_pose: Optional[str] = None
    gripper: Optional[str] = None
    left_gripper: Optional[str] = None
    right_gripper: Optional[str] = None
    base: Optional[str] = None
    waist_joint: Optional[str] = None
    torso_joint: Optional[str] = None
    left_leg: Optional[str] = None
    right_leg: Optional[str] = None
    left_fig6d: Optional[str] = None
    right_fig6d: Optional[str] = None
    base_pose: Optional[str] = None
    base_rot: Optional[str] = None  # gravity(3) + body angular velocity(3); state-only, no action counterpart

    def get_all_keys(self) -> list[str]:
        return [v for v in vars(self).values() if v is not None]


@dataclass
class RelativeStatsKeyMap:
    """Maps relative_stats.json key names to semantic roles."""
    ee_pose: Optional[str] = None                 # single-arm
    left_ee_pose: Optional[str] = None
    right_ee_pose: Optional[str] = None
    torso_joint: Optional[str] = None
    base_pose: Optional[str] = None               # relative base pose stats key


@dataclass
class DatasetSourceConfig:
    name: str
    data_path: str
    robot_type: str
    source_fps: int
    arm_type: str              # "single_left", "dual", "dual_with_legs"
    ee_format: str             # "xyz_rpy", "xyz_quat", or "xyz_rvec"

    enabled: bool = True
    weight: float = 1.0
    multi_task: bool = False
    # Embodiment-embedding type (action_model.embodiment_types) for this source; None -> robot_type.
    # Lets a source keep robot_type's dataset class / key layout but get its own conditioning id,
    # e.g. a different teleop recorder whose labels follow other conventions on the same robot.
    embodiment: Optional[str] = None
    # For robochallenge: list of sub-robot dirs, each with multi-task
    sub_robot_dirs: Optional[list[str]] = None

    image_keys: list[ImageKeyConfig] = field(default_factory=list)
    action_keys: ActionKeysConfig = field(default_factory=ActionKeysConfig)
    state_keys: StateKeysConfig = field(default_factory=StateKeysConfig)
    relative_stats_key_map: RelativeStatsKeyMap = field(default_factory=RelativeStatsKeyMap)

    # Extra data paths to merge stats from (for cross-variant shared normalizers)
    peer_data_paths: Optional[list[str]] = None
    # Merge left and right EE stats into one symmetric normalizer (for dual-arm robots)
    merge_left_right_ee_stats: bool = False

    # Deprecated compatibility field; global TrainingDataConfig.image_size controls resizing.
    image_size: Optional[list[int]] = None

    # base_command dimension info (varies per dataset)
    base_command_dims: Optional[dict] = None  # e.g. {"vx": 0, "vy": 1, "vyaw": 2} or {"vx":0,"vy":1,"vw":2,"height":3}

    # Pre-collected merged stats. When set, points to a directory containing
    # `stats.json` AND `relative_stats.json` at its root (no `meta/` subdir).
    # `_load_stats` short-circuits to load those two files verbatim and skips
    # all per-task / cross-variant merging.
    precollected_stats_path: Optional[str] = None

    def get_all_selected_keys(self) -> list[str]:
        """Get all parquet/video keys to load."""
        keys = []
        keys.extend(ik.key for ik in self.image_keys)
        keys.extend(self.action_keys.get_all_keys())
        keys.extend(self.state_keys.get_all_keys())
        return keys


@dataclass
class TrainingDataConfig:
    target_fps: int = 30
    chunk_size: int = 30
    image_size: Optional[list[int]] = None
    norm_type: str = "minmax_q"
    rel_norm_type: str = "zscore"
    state_norm_type: Optional[str] = None  # Separate norm type for state keys; defaults to norm_type if not set
    gripper_norm_type: Optional[str] = None  # Override norm_type for gripper keys; defaults to norm_type
    binarize_gripper: bool = False  # Convert continuous gripper to binary 0/1
    data_base: str = ""
    cache_dir: Optional[str] = None  # Configured cache root for HF datasets and local Arrow caches
    datasets: list[DatasetSourceConfig] = field(default_factory=list)
    # Optional logical dataset view for quick validation/debug runs. The fraction
    # is applied uniformly per task by default, so every task contributes data.
    sample_fraction: Optional[float] = None
    sample_seed: int = 0
    sample_strategy: str = "per_task"  # "per_task" or "per_source"
    # "proportional": sources sampled by length (weights ignored); "weighted": by per-source weight
    mixture_sampling: str = "proportional"

    # Image roles that the model expects (union of all possible cameras)
    image_roles: list[str] = field(default_factory=lambda: [
        "cam_high", "cam_wrist", "cam_side",
        "cam_wrist_left", "cam_wrist_right",
        "head_left", "head_right",
    ])


def _build_image_keys(raw: list[dict]) -> list[ImageKeyConfig]:
    return [
        ImageKeyConfig(key=item["key"], role=item["role"], resize=item.get("resize"))
        for item in raw
    ]


def _from_dict(cls, raw: dict):
    return cls(**{k: v for k, v in raw.items() if v is not None})


def _build_dataset_config(raw: dict, data_base: str = "") -> DatasetSourceConfig:
    import os

    # Compute full data_path from data_base + suffix
    data_path = os.path.join(data_base, raw["data_path"]) if data_base else raw["data_path"]

    # Compute full peer_data_paths from data_base + suffixes
    peer_suffixes = raw.get("peer_data_paths")
    peer_paths = [os.path.join(data_base, p) for p in peer_suffixes] if (peer_suffixes and data_base) else peer_suffixes

    return DatasetSourceConfig(
        name=raw["name"],
        data_path=data_path,
        robot_type=raw["robot_type"],
        source_fps=raw["source_fps"],
        arm_type=raw["arm_type"],
        ee_format=raw["ee_format"],
        enabled=raw.get("enabled", True),
        weight=raw.get("weight", 1.0),
        multi_task=raw.get("multi_task", False),
        embodiment=raw.get("embodiment"),
        sub_robot_dirs=raw.get("sub_robot_dirs"),
        peer_data_paths=peer_paths,
        merge_left_right_ee_stats=raw.get("merge_left_right_ee_stats", False),
        image_size=raw.get("image_size"),
        image_keys=_build_image_keys(raw.get("image_keys", [])),
        action_keys=_from_dict(ActionKeysConfig, raw.get("action_keys", {})),
        state_keys=_from_dict(StateKeysConfig, raw.get("state_keys", {})),
        relative_stats_key_map=_from_dict(RelativeStatsKeyMap, raw.get("relative_stats_key_map", {})),
        base_command_dims=raw.get("base_command_dims"),
        precollected_stats_path=raw.get("precollected_stats_path"),
    )


def load_config(config_path: str | Path) -> TrainingDataConfig:
    """Load training config from YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    data_base = raw.get("data_base", "")

    datasets = []
    for ds_raw in raw.get("datasets", []):
        datasets.append(_build_dataset_config(ds_raw, data_base))

    return TrainingDataConfig(
        target_fps=raw.get("target_fps", 30),
        chunk_size=raw.get("chunk_size", 30),
        image_size=raw.get("image_size"),
        norm_type=raw.get("norm_type", "minmax_q"),
        rel_norm_type=raw.get("rel_norm_type", "zscore"),
        state_norm_type=raw.get("state_norm_type", raw.get("norm_type", "minmax_q")),
        gripper_norm_type=raw.get("gripper_norm_type"),
        binarize_gripper=raw.get("binarize_gripper", False),
        data_base=data_base,
        cache_dir=raw.get("cache_dir"),
        datasets=datasets,
        sample_fraction=raw.get("sample_fraction"),
        sample_seed=raw.get("sample_seed", 0),
        sample_strategy=raw.get("sample_strategy", "per_task"),
        mixture_sampling=raw.get("mixture_sampling", "proportional"),
        image_roles=raw.get("image_roles", [
            "cam_high", "cam_wrist", "cam_side",
            "cam_wrist_left", "cam_wrist_right",
            "head_left", "head_right",
        ]),
    )
