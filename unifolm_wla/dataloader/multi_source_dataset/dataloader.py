"""DataLoader creation with custom collation for robot datasets."""

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from .config import load_config
from .single_source_dataset import create_single_source_dataset

logger = logging.getLogger(__name__)


class _CpuAcceleratorWorkerDataLoader(DataLoader):
    """DataLoader that forces ``DS_ACCELERATOR=cpu`` while spawning workers.

    Secondary optimization (NOT the 520 MiB fix — that is the spawn-worker guard
    in ``initialize_overwatch``). Each spawned data worker re-imports the training
    entrypoint, which reaches DeepSpeed; DeepSpeed's import-time
    ``get_accelerator()`` auto-detect then runs the async-io op-builder
    (`gcc ... -laio`) compat probe in every worker. Setting ``DS_ACCELERATOR=cpu``
    only around worker spawn makes the worker take DeepSpeed's `cpu: pass` branch,
    skipping those probes. The GPU stays VISIBLE so the worker import does not
    stall (unlike blanking CUDA_VISIBLE_DEVICES, which deadlocks at first iter()).

    The parent (trainer) is unaffected: it already cached ``CUDA_Accelerator`` in
    DeepSpeed's module global at startup, which short-circuits a later env change;
    we also restore the parent env right after spawn.
    """

    def _get_iterator(self):
        prev = os.environ.get("DS_ACCELERATOR")
        os.environ["DS_ACCELERATOR"] = "cpu"
        try:
            return super()._get_iterator()
        finally:
            if prev is None:
                os.environ.pop("DS_ACCELERATOR", None)
            else:
                os.environ["DS_ACCELERATOR"] = prev


class WeightedMixtureSampler(DistributedSampler):
    """Draw sources with probability proportional to their `weight`, regardless of their length.

    Each sample of source s gets weight ``w_s / len(s)`` and indices are drawn with replacement,
    so a small source is revisited while a large one is subsampled. The draw is seeded by
    ``seed + epoch`` (``set_epoch`` reshuffles) and is identical on every rank; each rank keeps
    its ``rank::num_replicas`` slice, as DistributedSampler does.
    """

    def __init__(self, dataset, weights: list[float], lengths: list[int], num_replicas: int = 1,
                 rank: int = 0, seed: int = 0, num_samples: int | None = None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed)
        if any(n == 0 for n in lengths):
            raise ValueError(f"weighted mixture sampling needs every enabled source non-empty, got lengths {lengths}")
        w = torch.cat([torch.full((n,), wt / n, dtype=torch.float64) for wt, n in zip(weights, lengths)])
        self.sample_weights = w / w.sum()
        self.total_size = num_samples or len(dataset)
        self.total_size -= self.total_size % num_replicas
        self.num_samples = self.total_size // num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.sample_weights, self.total_size, replacement=True, generator=g)
        return iter(idx[self.rank:self.total_size:self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for mixed robot data.

    Handles:
    - Fixed-size tensors: stack normally
    - Image dicts: stack per-role, fill missing cameras with same-shape zeros
    - Image masks: collect per-role
    - Task strings: collect as list
    """
    result = {}

    # Stack fixed-size tensors
    result["action"] = torch.stack([b["action"] for b in batch])           # (B, T, 54)
    result["action_mask"] = torch.stack([b["action_mask"] for b in batch]) # (B, 54)
    result["state"] = torch.stack([b["state"] for b in batch])             # (B, 60)
    result["state_mask"] = torch.stack([b["state_mask"] for b in batch])   # (B, 60)
    result["action_norm_offset"] = torch.stack([b["action_norm_offset"] for b in batch])
    result["action_norm_scale"] = torch.stack([b["action_norm_scale"] for b in batch])

    # Collect image roles in first-appearance (insertion) order, NOT sorted.
    # Each sample's images dict is built in config.image_keys order by
    # _process_images, so first-appearance order is deterministic and matches
    # the role-tag order in the prompt (the pixel order fed to the VLM processor
    # must match).
    all_roles = []
    seen = set()
    for b in batch:
        for r in b["images"].keys():
            if r not in seen:
                seen.add(r)
                all_roles.append(r)

    # Stack images per role. All enabled sources must produce the same shape for
    # a shared role; silently padding mixed-resolution images would waste compute
    # and introduce black-border artifacts into the vision encoder.
    images = {}
    image_mask = {}
    for role in all_roles:
        role_entries = []
        role_masks = []
        role_template = None
        expected_shape = None
        for b in batch:
            img = b["images"].get(role)
            valid = bool(b["image_mask"].get(role, False)) if img is not None else False
            role_entries.append(img)
            role_masks.append(valid)
            if img is not None:
                role_template = img if role_template is None else role_template
                shape = tuple(img.shape)
                if expected_shape is None:
                    expected_shape = shape
                elif shape != expected_shape:
                    raise ValueError(
                        f"Mixed image shapes for role {role!r}: "
                        f"expected {expected_shape}, got {shape}. "
                        "Set a top-level image_size to resize to a common resolution."
                    )

        if role_template is None:
            continue

        role_imgs = []
        for img in role_entries:
            if img is None:
                img = torch.zeros_like(role_template)
            role_imgs.append(img)

        images[role] = torch.stack(role_imgs)             # (B, C, H, W)
        image_mask[role] = torch.tensor(role_masks)       # (B,)

    result["images"] = images
    result["image_mask"] = image_mask

    # Collect strings
    result["task"] = [b["task"] for b in batch]
    result["arm_type"] = [b["arm_type"] for b in batch]
    result["robot_type"] = [b["robot_type"] for b in batch]
    result["embodiment"] = [b.get("embodiment") or b["robot_type"] for b in batch]
    result["source"] = [b.get("source", "") for b in batch]

    return result


def create_training_dataloader(
    config_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    shuffle: bool = True,
    distributed: bool = False,
    num_replicas: int | None = None,
    rank: int | None = None,
):
    """Create a training DataLoader for one or more dataset sources.

    Every enabled source is built independently and concatenated into one
    map-style dataset. With the default ``mixture_sampling: proportional``,
    sampling is proportional to source length and per-source ``weight`` values
    are not applied; ``mixture_sampling: weighted`` draws each source with
    probability proportional to its ``weight`` (WeightedMixtureSampler).

    Args:
        config_path: path to YAML config
        batch_size: batch size
        num_workers: number of data loading workers
        pin_memory: pin memory for GPU transfer
        prefetch_factor: number of batches to prefetch per worker
        shuffle: shuffle samples every epoch (ignored if `distributed=True`,
                 where `DistributedSampler(shuffle=...)` controls it instead)
        distributed: if True, wrap with `DistributedSampler`
        num_replicas: passed to `DistributedSampler` when `distributed=True`
        rank: passed to `DistributedSampler` when `distributed=True`

    Returns:
        (dataloader, dataset) tuple
    """
    config = load_config(config_path)
    enabled = [d for d in config.datasets if d.enabled]
    if not enabled:
        raise ValueError(
            f"create_training_dataloader requires at least one enabled dataset "
            f"source, got 0 in {config_path}."
        )

    source_datasets = [
        create_single_source_dataset(source_config, config)
        for source_config in enabled
    ]
    dataset = (
        source_datasets[0]
        if len(source_datasets) == 1
        else ConcatDataset(source_datasets)
    )
    if len(source_datasets) > 1:
        source_summary = ", ".join(
            f"{source.config.name}={len(source)}" for source in source_datasets
        )
        logger.info(
            "Concatenated %d enabled dataset sources (%s), total=%d",
            len(source_datasets),
            source_summary,
            len(dataset),
        )

    sampler = None
    if config.mixture_sampling == "weighted":
        weights = [s.config.weight for s in source_datasets]
        lengths = [len(s) for s in source_datasets]
        sampler = WeightedMixtureSampler(
            dataset, weights, lengths,
            num_replicas=num_replicas if distributed else 1,
            rank=rank if distributed else 0,
            seed=config.sample_seed,
        )
        logger.info(
            "Weighted mixture sampling: %s",
            ", ".join(f"{s.config.name}: weight={w / sum(weights):.3f} len={n}"
                      for s, w, n in zip(source_datasets, weights, lengths)),
        )
    elif config.mixture_sampling != "proportional":
        raise ValueError(f"mixture_sampling must be 'proportional' or 'weighted', got {config.mixture_sampling!r}")
    elif distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
        )

    dataloader = _CpuAcceleratorWorkerDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        # `spawn` (not the default `fork`): a forked worker would inherit the
        # trainer's already-initialized CUDA context. The data path is CPU-only.
        #
        # persistent_workers=True: spawn the workers once and reuse them across
        # epochs. The trainer recreates the iterator each epoch via
        # `_reset_dataloader` -> iter(dataloader); with persistent workers that
        # reuses the existing pool instead of spawning a new worker generation
        # per epoch.
        #
        # The ~520 MiB/worker CUDA context is NOT created here — it was caused by
        # spawn workers re-running the training entrypoint (`runpy.run_path`),
        # whose import of `tools.py` built a `DistributedOverwatch` ->
        # `accelerate.PartialState()` -> `torch.cuda.set_device()` in every worker.
        # Fixed in `initialize_overwatch` (returns the CUDA-free PureOverwatch when
        # `--multiprocessing-fork` is in argv). DS_ACCELERATOR=cpu above only
        # suppresses DeepSpeed's op-builder gcc probes in workers.
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return dataloader, dataset

def _worker_init(worker_id):
    import torch, os
    torch.set_num_threads(1)        # ATen
    try:
        import cv2; cv2.setNumThreads(0)   # if you use opencv
    except ImportError:
        pass

    # NOTE: OMP_NUM_THREADS is inherited from the parent process; if you want
    # workers to have a different OMP budget, you have to fork BEFORE setting it,
    # or use os.sched_setaffinity. Inheritance from bash is usually fine.
