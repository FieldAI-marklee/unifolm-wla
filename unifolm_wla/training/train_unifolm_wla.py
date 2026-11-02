import argparse
import collections
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from unifolm_wla.dataloader import build_dataloader
from unifolm_wla.model.framework.base_framework import build_framework
from unifolm_wla.model.framework.share_tools import apply_config_compat
from unifolm_wla.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from unifolm_wla.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Accelerator is initialized in main() after config is parsed so that
# gradient_accumulation_steps from CLI args is correctly forwarded to Accelerate/DeepSpeed.
accelerator = None

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    data_id = getattr(cfg.datasets.vla_data, "data_mix", None) or getattr(cfg.datasets.vla_data, "data_config_path", "")
    logger.info(f"Creating VLA Dataset: {data_id}")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    # When the dataloader has a custom distributed batch_sampler (batch_size=None),
    # it already handles distribution internally. Don't pass it to accelerator.prepare()
    # — that would cause accelerate to try to re-wrap the sampler, triggering both
    # the even_batches check and the train_micro_batch_size_per_gpu "auto" failure.
    # We still need to set train_micro_batch_size_per_gpu for DeepSpeed's gradient
    # accumulation math, since it can't auto-detect from a None batch_size.
    if vla_train_dataloader.batch_size is None:
        from accelerate.state import AcceleratorState
        AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            cfg.datasets.vla_data.per_device_batch_size
        )

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


_SAFETENSORS_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
    torch.uint8: "U8", torch.bool: "BOOL",
}


def save_safetensors_streaming(state_dict, path):
    """Write a safetensors file one tensor at a time.

    safetensors.save_file materializes the whole file in host RAM (~12 GB here), which on top of the
    CPU-offloaded optimizer gets the process (and the IDE) OOM-killed.
    """
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for k, t in state_dict.items():
        n = t.numel() * t.element_size()
        header[k] = {"dtype": _SAFETENSORS_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + n]}
        offset += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for t in state_dict.values():
            f.write(t.detach().contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes())
    os.replace(tmp, path)


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    adam_kwargs = dict(
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )
    from accelerate.state import AcceleratorState

    ds_plugin = AcceleratorState().deepspeed_plugin
    zero_cfg = ds_plugin.deepspeed_config.get("zero_optimization", {}) if ds_plugin else {}
    if zero_cfg.get("offload_optimizer", {}).get("device") == "cpu":
        # ZeRO-Offload rejects torch optimizers; DeepSpeedCPUAdam is the AdamW (adamw_mode) it expects.
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        optimizer = DeepSpeedCPUAdam(param_groups, adamw_mode=True, **adam_kwargs)
    else:
        optimizer = torch.optim.AdamW(param_groups, fused=True, **adam_kwargs)

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


def _is_lora_active(cfg) -> bool:
    """True if `cfg.trainer.lora` is enabled and at least one backbone sub-block is enabled."""
    lora_cfg = getattr(cfg.trainer, "lora", None)
    if not lora_cfg or not lora_cfg.get("enabled", False):
        return False
    return any(
        lora_cfg.get(name, {}).get("enabled", False)
        for name in ("qwen_vl_interface", "action_model")
    )


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader=None, optimizer=None, lr_scheduler=None, accelerator=None):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.tb_writer = None
        self._window = self._new_window()
        self._val_sets = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (DeepSpeed init / dataloader build) crashes, the produced run dir is
        # still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        if self.vla_train_dataloader.batch_size is None:
            # DataLoader already has a distributed batch_sampler (e.g. multi_source_dataset's
            # DistributedResolutionSourceBucketBatchSampler). Don't pass it to accelerator.prepare()
            # — accelerate would try to re-wrap the sampler, triggering even_batches and
            # train_micro_batch_size_per_gpu failures. Prepare only model + optimizer.
            self.model, self.optimizer = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )

        self._init_trackers()
        self._build_val_sets()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_trackers(self):
        """Initialize TensorBoard (<output_dir>/tensorboard) and Weights & Biases.

        The wandb mode is $WANDB_MODE if set, else the `wandb_mode` config key (default "offline":
        runs land in <output_dir>/wandb and can be uploaded later with `wandb sync`).
        """
        if self.accelerator.is_main_process:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.config.output_dir, "tensorboard"))
            mode = os.environ.get("WANDB_MODE") or getattr(self.config, "wandb_mode", None) or "offline"
            os.makedirs(os.path.join(self.config.output_dir, "wandb"), exist_ok=True)
            run_config = OmegaConf.to_container(self.config, resolve=True)
            sources = getattr(getattr(self.vla_train_dataloader, "dataset", None), "datasets", [])
            run_config["data_sources"] = {
                s.config.name: {"weight": s.config.weight, "frames": len(s),
                                "embodiment": s.config.embodiment or s.config.robot_type}
                for s in sources if hasattr(s, "config")
            }
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
                mode=mode,
                config=run_config,
            )
            logger.info(f"Trackers: TensorBoard at {self.tb_writer.log_dir}, wandb mode={mode}")

    def _build_val_sets(self):
        """Fixed per-source validation samples from `trainer.val_data_config_path` (optional).

        Each enabled source of that data config contributes `trainer.val_samples_per_source`
        frames, drawn once with a fixed seed, so every validation pass sees the same inputs.
        """
        path = getattr(self.config.trainer, "val_data_config_path", None)
        if not path or not self.accelerator.is_main_process:
            return
        from unifolm_wla.dataloader.multi_source_dataset.config import load_config
        from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset

        n = int(getattr(self.config.trainer, "val_samples_per_source", 64))
        vcfg = load_config(path)
        rng = np.random.default_rng(0)
        self._val_sets = {}
        for sc in vcfg.datasets:
            if not sc.enabled:
                continue
            ds = create_single_source_dataset(sc, vcfg)
            idx = np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False)).tolist()
            self._val_sets[sc.name] = (ds, idx)
            logger.info(f"Validation source {sc.name}: {len(idx)} of {len(ds)} frames")

    @torch.no_grad()
    def validate(self) -> dict:
        """Per-source open-loop validation on the fixed samples of `_build_val_sets`.

        val/<source>/masked_mse   normalized-space MSE of predict_action over the action_mask dims
        val/<source>/ee_pos_mm    mean relative-EE xyz error (both arms), in mm, de-normalized
        The diffusion noise is seeded, so the passes are comparable step to step.
        """
        if not self._val_sets:
            return {}
        from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES
        from unifolm_wla.dataloader.multi_source_dataset.dataloader import collate_fn

        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        bs = int(getattr(self.config.trainer, "val_batch_size", 8))
        horizon = model.action_horizon
        ee = np.r_[np.arange(54)[SLICES["left_xyz_rotvec"]][:3], np.arange(54)[SLICES["right_xyz_rotvec"]][:3]]
        out = {}
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(0)
            for name, (ds, idx) in self._val_sets.items():
                se, cnt, ee_err = 0.0, 0, []
                for i in range(0, len(idx), bs):
                    batch = collate_fn([ds[j] for j in idx[i:i + bs]])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = model.predict_action(examples=batch)["normalized_actions"].astype(np.float64)
                    gt = batch["action"].numpy()[:, -horizon:, :].astype(np.float64)
                    m = batch["action_mask"].numpy()[:, None, :].astype(bool)
                    se += float((((pred - gt) ** 2) * m).sum())
                    cnt += int(m.sum()) * gt.shape[1]
                    scale = batch["action_norm_scale"].numpy()[:, None, :]
                    d = (pred - gt) * scale
                    ee_err.append(np.linalg.norm(d[..., ee].reshape(*d.shape[:2], 2, 3), axis=-1).mean(axis=(1, 2)))
                out[f"val/{name}/masked_mse"] = se / max(cnt, 1)
                out[f"val/{name}/ee_pos_mm"] = float(np.concatenate(ee_err).mean() * 1e3)
        if was_training:
            model.train()
        logger.info(f"Step {self.completed_steps} validation: {out}")
        return out

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def init_checkpoint_and_lora(self):
        """Initialize checkpoint directory, handle checkpoint loading, and inject
        LoRA adapters (if configured). Must run before the optimizer is built
        (`setup_optimizer_and_scheduler`/`build_param_lr_groups` needs to see the
        final, post-LoRA parameter tree) — so this is called explicitly from
        `main()` rather than from `prepare_training()`.

        LoRA injection renames wrapped linears (e.g. `to_q` -> `to_q.base_layer`),
        so ordering relative to checkpoint loading matters:
          - resume: this run's own prior checkpoint already has renamed keys (if
            LoRA was active) -> inject LoRA first, then load.
          - fresh pretrained_checkpoint: external checkpoint has vanilla keys ->
            load first, then inject LoRA on top.
        """
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        lora_cfg = getattr(self.config.trainer, "lora", None)
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.apply_lora_adapters(self.model, lora_cfg)
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            remove_modules = getattr(self.config.trainer, "remove_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules, remove_modules=remove_modules)
            self.model = self.apply_lora_adapters(self.model, lora_cfg)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            self.model = self.apply_lora_adapters(self.model, lora_cfg)
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _full_state_dict(self):
        ds_plugin = self.accelerator.state.deepspeed_plugin
        if ds_plugin is not None and ds_plugin.zero_stage == 3:
            return self.accelerator.get_state_dict(self.model)
        # ZeRO <= 2 keeps full params on every rank; get_state_dict would first clone them all to host RAM.
        return self.accelerator.unwrap_model(self.model).state_dict()

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self._full_state_dict()
            if save_format == "safetensors":
                save_safetensors_streaming(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            if _is_lora_active(self.config):
                self._save_adapter_only(state_dict, checkpoint_path + "_adapter", save_format)

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    @staticmethod
    def _new_window() -> dict:
        return {"loss": collections.defaultdict(list), "data_s": 0.0, "model_s": 0.0, "samples": 0,
                "sources": collections.Counter(), "grad_norm": [], "t0": time.perf_counter()}

    def _accumulate(self, step_metrics: dict, batch, t_data: float, t_model: float):
        """Fold one micro-step into the current logging window."""
        w = self._window
        for k, v in step_metrics.items():
            w["loss"][k].append(v)
        w["data_s"] += t_data
        w["model_s"] += t_model
        if isinstance(batch, dict) and "action" in batch:
            w["samples"] += int(batch["action"].shape[0])
            w["sources"].update(batch.get("source", []))

    def _log_metrics(self, extra: dict | None = None):
        """Log the window since the last call, once per optimizer step (every `logging_frequency`).

        Losses are averaged over all micro-steps of the window (i.e. over the accumulation window),
        timings are per optimizer step, `data/frac/<source>` is the realized mixture.
        """
        if not self.accelerator.is_main_process:
            self._window = self._new_window()
            return
        w, steps = self._window, max(1, self.config.trainer.logging_frequency)
        wall = time.perf_counter() - w["t0"]
        metrics = {f"train/{k}": float(np.mean(v)) for k, v in w["loss"].items() if v}
        if w["grad_norm"]:
            metrics["train/grad_norm"] = float(np.mean(w["grad_norm"]))
        last_lrs = self.lr_scheduler.get_last_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            group_name = group.get("name", str(i))
            metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
        metrics["timing/data"] = w["data_s"] / steps
        metrics["timing/model"] = w["model_s"] / steps
        metrics["timing/step"] = wall / steps
        metrics["perf/samples_per_s"] = w["samples"] / max(wall, 1e-9)
        if torch.cuda.is_available():
            metrics["perf/gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        n = sum(w["sources"].values())
        for src, c in w["sources"].items():
            metrics[f"data/frac/{src}"] = c / n
        metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 4)
        metrics.update(extra or {})
        self._emit(metrics)
        self._window = self._new_window()

    def _emit(self, metrics: dict):
        if not self.accelerator.is_main_process:
            return
        wandb.log(metrics, step=self.completed_steps)
        if self.tb_writer is not None:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, self.completed_steps)
            self.tb_writer.flush()
        logger.info(f"Step {self.completed_steps}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # Validation before the first update, so the val/* curves start at the loaded checkpoint
        # (the base model on a fresh run).
        if getattr(self.config.trainer, "val_at_start", True) and self._val_sets:
            self._emit(self.validate())
        if dist.is_initialized():
            dist.barrier()

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # completed_steps only advances on the micro-step that syncs gradients; gate eval/save on
            # it too, or every micro-step of an accumulation window re-runs them (up to
            # gradient_accumulation_steps saves of the full ~12.5 GB model per save point).
            synced = self.accelerator.sync_gradients
            if synced:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            self._accumulate(step_metrics, batch_vla, t_end_data - t_start_data, t_end_model - t_start_model)
            if synced:
                gn = self._grad_norm()
                if gn is not None:
                    self._window["grad_norm"].append(gn)

            extra = {}
            if synced and self.completed_steps % self.config.trainer.eval_interval == 0:
                extra = self.eval_action_model({})
                extra.update(self.validate())
                if dist.is_initialized():
                    dist.barrier()

            if synced and (self.completed_steps % self.config.trainer.logging_frequency == 0 or extra):
                self._log_metrics(extra)

            if synced and self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        if isinstance(examples, dict):
            actions = examples["action"].cpu().numpy()[:, -self.accelerator.unwrap_model(self.model).action_horizon:, :]
        else:
            model = self.accelerator.unwrap_model(self.model)
            actions = np.array([e["action"][-model.action_horizon:] for e in examples])
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(examples=examples)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["train/mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _grad_norm(self):
        """Global grad norm of the last optimizer step (DeepSpeed computes it; accelerate's
        clip_grad_norm_ returns None under DeepSpeed)."""
        fn = getattr(self.model, "get_global_grad_norm", None)
        if fn is None:
            return None
        gn = fn()
        if gn is None:
            return None
        gn = float(gn.item() if torch.is_tensor(gn) else gn)
        return gn if np.isfinite(gn) else None

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()

        return {
            "action_dit_loss": action_loss.item(),
            **{
                k: output_dict[k].item()
                for k in ("flow_action_loss", "lm_ce_loss")
                if k in output_dict
            },
        }

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self._full_state_dict()
            if save_format == "safetensors":
                save_safetensors_streaming(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            if _is_lora_active(self.config):
                self._save_adapter_only(state_dict, os.path.join(final_checkpoint, "adapter"), save_format)

            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()
            if self.tb_writer is not None:
                self.tb_writer.close()

        self.accelerator.wait_for_everyone()

    def _save_adapter_only(self, full_state_dict, path_prefix, save_format):
        """Filter an already-gathered full `state_dict` (DeepSpeed-safe, via
        `accelerator.get_state_dict`) down to LoRA-adapter keys only, and save
        it separately as a small artifact alongside the full checkpoint.
        """
        adapter_state_dict = {k: v for k, v in full_state_dict.items() if "lora_" in k}
        if not adapter_state_dict:
            logger.warning(
                "LoRA is enabled but no `lora_*` keys were found in the state_dict — "
                "check that apply_lora_adapters actually ran."
            )
            return

        if save_format == "safetensors":
            from safetensors.torch import save_file

            save_file(adapter_state_dict, path_prefix + ".safetensors")
        elif save_format == "pt":
            torch.save(adapter_state_dict, path_prefix + ".pt")
        else:
            raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

        num_params = sum(v.numel() for v in adapter_state_dict.values())
        self.accelerator.print(
            f"✅ Adapter-only checkpoint saved at {path_prefix} "
            f"({len(adapter_state_dict)} tensors, {num_params / 1e6:.3f}M params)"
        )


def main(cfg) -> None:
    global accelerator
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    gradient_accumulation_steps = getattr(cfg.trainer, "gradient_accumulation_steps", 1)
    deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
    )
    accelerator.print(accelerator.state)

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)

    trainer = VLATrainer(cfg=cfg, model=vla, accelerator=accelerator)
    trainer.init_checkpoint_and_lora()

    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=trainer.model, cfg=cfg)

    trainer.vla_train_dataloader = vla_train_dataloader
    trainer.optimizer = optimizer
    trainer.lr_scheduler = lr_scheduler

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/unifolm_wla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
