"""
Date: 2024-11-12 10:57:29
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-12 11:16:09
FilePath: /MineStudio/minestudio/train/callbacks/callback.py
"""

import torch
from typing import Dict, Any
from minestudio.models import MinePolicy


class ObjectiveCallback:
    def __init__(self): ...

    def __call__(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        step_name: str,
        latents: Dict[str, torch.Tensor],
        mine_policy: MinePolicy,
    ) -> Dict[str, torch.Tensor]:
        return {}
