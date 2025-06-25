"""
Date: 2024-11-24 08:23:02
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-13 14:16:24
FilePath: /MineStudio/minestudio/tutorials/train/3_pretrain_groots/main.py
"""

import hydra
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from einops import rearrange
from typing import Dict, Any, Tuple

from minestudio.data import MineDataModule
from minestudio.offline import MineLightning
from minestudio.models import GrootPolicy
from minestudio.offline.utils import convert_to_normal
from minestudio.offline.mine_callbacks import BehaviorCloneCallback, KLDivergenceCallback
from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback, EMA


logger = WandbLogger(project="minestudio")


# logger = None
@hydra.main(config_path=".", config_name="groot_config")
def main(args):
    groot_policy = GrootPolicy(**convert_to_normal(args.model))
    mine_lightning = MineLightning(
        mine_policy=groot_policy,
        log_freq=20,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=[
            BehaviorCloneCallback(weight=args.loss_scale * args.bc_weight),
            KLDivergenceCallback(weight=args.loss_scale * args.kl_div_weight),
        ],
        hyperparameters=convert_to_normal(args),
    )

    mine_data = MineDataModule(
        data_params=dict(
            mode="raw",
            dataset_dirs=args.dataset_dirs,
            frame_width=224,
            frame_height=224,
            win_len=128,
            enable_segment=True,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_ratio=args.split_ratio,
        shuffle_episodes=args.shuffle_episodes,
        episode_continuous_batch=args.episode_continuous_batch,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        SpeedMonitorCallback(),
        SmartCheckpointCallback(
            dirpath="./weights",
            filename="weight-{epoch}-{step}",
            save_top_k=-1,
            every_n_train_steps=args.save_freq,
            save_weights_only=True,
        ),
        SmartCheckpointCallback(
            dirpath="./checkpoints",
            filename="ckpt-{epoch}-{step}",
            save_top_k=1,
            every_n_train_steps=args.save_freq + 1,
            save_weights_only=False,
        ),
        EMA(
            decay=args.ema.decay,
            validate_original_weights=args.ema.validate_original_weights,
            every_n_steps=args.ema.every_n_steps,
            cpu_offload=args.ema.cpu_offload,
        ),
    ]

    L.Trainer(
        logger=logger,
        devices=args.devices,
        precision="bf16",
        strategy="ddp_find_unused_parameters_true",
        use_distributed_sampler=not args.episode_continuous_batch,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    ).fit(
        model=mine_lightning,
        datamodule=mine_data,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    main()
