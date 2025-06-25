"""
Date: 2024-12-12 13:10:58
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-12 13:26:55
FilePath: /MineStudio/minestudio/train/mine_callbacks/kl_divergence.py
"""

import torch
from typing import Dict, Any
from minestudio.models import MinePolicy
from minestudio.offline.mine_callbacks.callback import ObjectiveCallback


class KLDivergenceCallback(ObjectiveCallback):
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
        posterior_dist = latents["posterior_dist"]
        prior_dist = latents["prior_dist"]

        q_mu, q_log_var = posterior_dist["mu"], posterior_dist["log_var"]
        p_mu, p_log_var = prior_dist["mu"], prior_dist["log_var"]

        kl_div = self.kl_divergence(q_mu, q_log_var, p_mu, p_log_var)
        result = {
            "loss": kl_div.mean() * self.weight,
            "kl_div": kl_div.mean(),
            "kl_weight": self.weight,
        }
        return result

    def kl_divergence(self, q_mu, q_log_var, p_mu, p_log_var):
        # shape: (B, D)
        KL = -0.5 * torch.sum(
            1 + (q_log_var - p_log_var) - (q_log_var - p_log_var).exp() - (q_mu - p_mu).pow(2) / p_log_var.exp(),
            dim=-1,
        )  # shape: (B)
        return KL
