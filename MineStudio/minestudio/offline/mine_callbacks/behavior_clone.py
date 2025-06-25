"""
Date: 2024-11-12 13:59:08
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-09 15:51:34
FilePath: /MineStudio/minestudio/train/mine_callbacks/behavior_clone.py
"""

from typing import Any, Dict

import torch

from minestudio.models import MinePolicy
from minestudio.offline.mine_callbacks.callback import ObjectiveCallback


class BehaviorCloneCallback(ObjectiveCallback):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        step_name: str,
        latents: Dict[str, torch.Tensor],
        mine_policy: MinePolicy,
    ) -> Dict[str, torch.Tensor]:
        assert "agent_action" in batch, "key `agent_action` is required for behavior cloning."
        agent_action = batch["agent_action"]  # [bs,window_len,1]
        pi_logits = latents["pi_logits"]  # [bs, window_len,1,8641/121]
        log_prob = mine_policy.pi_head.logprob(agent_action, pi_logits, return_dict=True)
        entropy = mine_policy.pi_head.entropy(pi_logits, return_dict=True)
        camera_mask = (agent_action["camera"] != 60).float().squeeze(-1)
        global_mask = batch.get("mask", torch.ones_like(camera_mask))
        logp_camera = (log_prob["camera"] * global_mask * camera_mask).mean(-1)
        logp_buttons = (log_prob["buttons"] * global_mask).mean(-1)
        entropy_camera = (entropy["camera"] * global_mask * camera_mask).mean(-1)
        entropy_buttons = (entropy["buttons"] * global_mask).mean(-1)
        camera_loss, button_loss = -logp_camera, -logp_buttons
        bc_loss = camera_loss + button_loss
        entropy = entropy_camera + entropy_buttons
        result = {
            "loss": bc_loss.mean() * self.weight,
            "camera_loss": camera_loss.mean(),
            "button_loss": button_loss.mean(),
            "entropy": entropy.mean(),
            "bc_weight": self.weight,
        }
        return result
