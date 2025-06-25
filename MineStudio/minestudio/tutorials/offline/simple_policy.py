"""
Date: 2024-11-12 14:00:50
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-12 12:26:39
FilePath: /MineStudio/minestudio/tutorials/train/simple_policy.py
"""

import torch
import torch.nn as nn
import lightning as L
from einops import rearrange
from typing import Dict, Any, Tuple

from minestudio.data import MineDataModule
from minestudio.offline import MineLightning
from minestudio.models import MinePolicy
from minestudio.offline.callbacks import BehaviorCloneCallback


class SimplePolicy(MinePolicy):
    def __init__(self, hiddim: int):
        super().__init__(hiddim=hiddim)
        self.net = nn.Sequential(
            nn.Linear(128 * 128 * 3, hiddim),
            nn.ReLU(),
            nn.Linear(hiddim, hiddim),
        )

    def forward(self, input: Dict[str, Any], state_in: Any) -> Tuple[Dict[str, torch.Tensor], Any]:
        x = rearrange(input["image"], "b t h w c -> b t (h w c)")
        x = self.net(x.float())
        result = {
            "pi_logits": self.pi_head(x),
            "vpred": self.value_head(x),
        }
        return result, state_in

    def initial_state(self, batch_size):
        return None


mine_lightning = MineLightning(
    mine_policy=SimplePolicy(hiddim=1024),
    log_freq=20,
    learning_rate=1e-4,
    warmup_steps=1000,
    weight_decay=0.01,
    callbacks=[
        BehaviorCloneCallback(weight=1.0),
    ],
)

mine_data = MineDataModule(
    data_params=dict(
        mode="raw",
        dataset_dirs=[
            "/nfs-shared-2/data/contractors/dataset_10xx",
        ],
        frame_width=128,
        frame_height=128,
        win_len=128,
    ),
    batch_size=8,
    num_workers=4,
    prefetch_factor=2,
)
trainer = L.Trainer(
    max_epochs=1, devices=2, precision=16, strategy="ddp_find_unused_parameters_true", use_distributed_sampler=False
)
trainer.fit(mine_lightning, datamodule=mine_data)
