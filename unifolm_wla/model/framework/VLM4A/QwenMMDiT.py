"""
Qwen-MMDiT Framework
Qwen-VL + MMDiT (double-stream joint-attention) flow-matching head to predict
continuous actions. Mirrors `QwenGR00T` but swaps the action head for the MMDiT
backbone (`DiT_modules/mmdit.py`).
"""

import os
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from unifolm_wla.model.modules.utils.image_tools import to_pil_preserve
from unifolm_wla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from unifolm_wla.model.framework.base_framework import baseframework
from unifolm_wla.model.framework.share_tools import merge_framework_config


from unifolm_wla.model.modules.action_model.MMDiT_ActionHeader import (
    MMDiTFlowmatchingActionHead,
    get_action_model,
)
from unifolm_wla.model.modules.action_model.MMDiT_ActionHeader import build_embodiment_map
from unifolm_wla.model.modules.vlm import get_vlm_model
from unifolm_wla.model.tools import FRAMEWORK_REGISTRY
from unifolm_wla.training.trainer_utils.trainer_tools import resize_images


def _adapt_multi_source_batch(batch: dict) -> list:
    """Convert collated multi_source_dataset batch → List[dict] for forward()."""
    import torchvision.transforms.functional as TF
    B = batch["action"].shape[0]
    examples = []
    for i in range(B):
        # Preserve the collate role order (= config.image_keys order) so the
        # pixel order matches the role-tag order in the prompt. Do NOT sort.
        roles = [r for r in batch["images"].keys() if batch["image_mask"][r][i].item()]
        images = [TF.to_pil_image(batch["images"][r][i]) for r in roles]
        examples.append({
            "image": images,
            "image_roles": roles,
            "lang": batch["task"][i],
            "action": batch["action"][i].cpu().numpy(),
            "action_mask": batch["action_mask"][i].cpu().numpy(),
            "state": batch["state"][i].cpu().numpy(),
            "state_mask": batch["state_mask"][i].cpu().numpy(),
            "arm_type": batch["arm_type"][i],
            "robot_type": batch["robot_type"][i],
            "embodiment": batch["embodiment"][i] if "embodiment" in batch else batch["robot_type"][i],
        })
    return examples


def _build_state_tensor(examples: list, device, dtype, training: bool = True) -> Optional[torch.Tensor]:
    """Build state tensor (B, 1, state_dim) with optional mask concatenation.

    Training-only data augmentation (per-example):
    - 10% prob: zero out state and state_mask
    - 30% prob: add Gaussian noise (std=0.1) to valid state dims
    - 60% prob: keep original
    """
    if "state" not in examples[0]:
        return None

    state = torch.tensor(np.array([e["state"] for e in examples]), device=device, dtype=dtype)
    has_mask = "state_mask" in examples[0]

    if has_mask:
        state_mask = torch.tensor(np.array([e["state_mask"] for e in examples]), device=device, dtype=dtype)

    if training:
        batch_size = state.shape[0]
        rand_vals = torch.rand(batch_size, 1, device=device)
        is_zero = rand_vals < 0.1
        is_noise = (rand_vals >= 0.1) & (rand_vals < 0.4)

        if is_noise.any():
            noise = torch.randn_like(state) * 0.1
            if has_mask:
                noise = noise * (state_mask > 0).to(dtype)
            state = torch.where(is_noise, state + noise, state)

        if is_zero.any():
            state = torch.where(is_zero, torch.zeros_like(state), state)
            if has_mask:
                state_mask = torch.where(is_zero, torch.zeros_like(state_mask), state_mask)

    if has_mask:
        state = torch.cat([state, state_mask], dim=1)

    return state.unsqueeze(1)


@dataclass
class QwenMMDiTDefaultConfig():
    """QwenMMDiT defaults: same as QwenGR00T but with an MMDiT action head.

    The MMDiT `DiT` backbone is parameterised by (num_attention_heads,
    attention_head_dim, num_layers, output_dim, cross_attention_dim).
    """

    name: str = "QwenMMDiT"
    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )
    # === Robot-state projector (state → VLM `<|robot_state|>` embedding) ===
    # When `enabled`, a RobotStateProjector (mirroring ms-swift pretraining) maps
    # the per-sample robot state into VLM hidden space and scatters it onto the
    # `<|robot_state|>` token. Weights load from `load_from` (default: base_vlm)
    # under `robot_state_projector.*` keys and co-train with the VLM.
    robot_state_projector: dict = field(
        default_factory=lambda: {
            "enabled": False,
            # Projector input dim = raw state_dim (60) * 2 (state ⊕ state_mask).
            # Must equal the pretrained projector's net.0 in-features.
            "input_dim": 120,
            # Intermediate MLP width. None → VLM hidden_size (matches pretrain).
            "projector_hidden_size": None,
            # Checkpoint dir holding robot_state_projector.* weights. None → base_vlm.
            "load_from": None,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 54,
            "state_dim": 60,
            "action_horizon": 8,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            # === MMDiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # cross_attention_dim is aligned to the VLM hidden_size at runtime
                "cross_attention_dim": 2048,
                "num_layers": 16,
                "num_attention_heads": 12,
                "attention_head_dim": 64,
                "output_dim": 1024,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenMMDiT")
class Qwen_MMDiT(baseframework):
    """
    Multimodal vision-language-action model (MMDiT variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - MMDiT double-stream joint-attention flow-matching head for continuous actions
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenMMDiTDefaultConfig, config)

        # When the robot-state projector injects state into the VLM, drop the
        # DiT head's own state token (state_dim=0 skips state_encoder).
        self._use_state_projector = bool(
            self.config.framework.get("robot_state_projector", {}).get("enabled", False)
        )
        if self._use_state_projector:
            self.config.framework.action_model.state_dim = 0

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # Align the MMDiT text-stream input dim to the VLM hidden size.
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: MMDiTFlowmatchingActionHead = get_action_model(config=self.config)

        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # Optional robot-state projector: inject state into the VLM token embedding.
        rsp_cfg = self.config.framework.get("robot_state_projector", {})
        if self._use_state_projector:
            self.qwen_vl_interface.attach_robot_state_projector(
                input_dim=int(rsp_cfg.get("input_dim", 120)),
                projector_hidden_size=rsp_cfg.get("projector_hidden_size"),
                load_from=rsp_cfg.get("load_from") or self.config.framework.qwenvl.get("base_vlm"),
            )
            logger.info(
                f"[QwenMMDiT] robot_state_projector enabled "
                f"(input_dim={int(rsp_cfg.get('input_dim', 120))}, co-trained)"
            )

    def _build_projector_state(self, examples: list) -> Optional[torch.Tensor]:
        """Clean (no-augmentation) state ⊕ mask for the VLM `<|robot_state|>` token."""
        if not self._use_state_projector:
            return None
        device = self.qwen_vl_interface.model.device
        state = _build_state_tensor(examples, device, torch.float32, training=self.training)
        return state.squeeze(1) if state is not None else None

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        if isinstance(examples, dict):
            examples = _adapt_multi_source_batch(examples)
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        solutions = None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=solutions,
            image_roles=[e.get("image_roles") for e in examples],
            arm_types=[e.get("arm_type") for e in examples],
        )
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        assistant_mask = qwen_inputs.pop("assistant_mask", None)

        robot_states = self._build_projector_state(examples)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                robot_states=robot_states,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Mask the assistant turn out of the DiT joint-attention keys so the
        # action tokens cannot attend to their own (leaked) discretised targets.
        keys_attn_mask = (
            backbone_attention_mask.to(dtype=torch.bool)
            if backbone_attention_mask is not None else None
        )
        if assistant_mask is not None:
            keys_attn_mask = (~assistant_mask) & keys_attn_mask if keys_attn_mask is not None else ~assistant_mask

        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            action_mask = None
            if "action_mask" in examples[0]:
                action_mask = torch.tensor(
                    np.array([e["action_mask"] for e in examples]),
                    device=last_hidden.device, dtype=last_hidden.dtype,
                )
                actions_target = torch.cat(
                    [actions_target, action_mask.unsqueeze(1).expand_as(actions_target)], dim=2
                )

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            if keys_attn_mask is not None:
                keys_attn_mask_repeated = keys_attn_mask.repeat(repeated_diffusion_steps, 1)
            else:
                keys_attn_mask_repeated = None

            state = (
                None if self._use_state_projector
                else _build_state_tensor(examples, last_hidden.device, last_hidden.dtype, training=self.training)
            )
            state_repeated = state.repeat(repeated_diffusion_steps, 1, 1) if state is not None else None

            body_type_ids = self._build_body_type_ids(examples, last_hidden.device)
            body_type_ids_repeated = (
                body_type_ids.repeat(repeated_diffusion_steps) if body_type_ids is not None else None
            )

            flow_action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=keys_attn_mask_repeated,
                body_type_ids=body_type_ids_repeated,
            )

        return {"flow_action_loss": flow_action_loss, "action_loss": flow_action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if isinstance(examples, dict):
            examples = _adapt_multi_source_batch(examples)
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions,
            image_roles=[e.get("image_roles") for e in examples],
            arm_types=[e.get("arm_type") for e in examples],
            add_generation_prompt=False,
        )
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        robot_states = self._build_projector_state(examples)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                robot_states=robot_states,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state = (
            None if self._use_state_projector
            else _build_state_tensor(examples, last_hidden.device, last_hidden.dtype, training=self.training)
        )

        action_mask = None
        if "action_mask" in examples[0]:
            action_mask = torch.tensor(
                np.array([e["action_mask"] for e in examples]),
                device=last_hidden.device, dtype=last_hidden.dtype,
            )

        body_type_ids = self._build_body_type_ids(examples, last_hidden.device)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state, action_mask=action_mask,
                encoder_attention_mask=backbone_attention_mask, body_type_ids=body_type_ids,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _build_body_type_ids(self, examples, device):
        """Map each example's embodiment (default: robot_type) -> embodiment embedding id (0 = unknown).

        An embodiment this model was not trained with falls back to its robot_type, so a checkpoint
        without the extra type (e.g. the base model) still scores such a source as its robot.
        """
        if not examples or "robot_type" not in examples[0]:
            return None
        emb_map = build_embodiment_map(self.action_model.embodiment_types)
        ids = [emb_map.get(e.get("embodiment"), emb_map.get(e.get("robot_type"), 0)) for e in examples]
        return torch.tensor(ids, dtype=torch.long, device=device)
