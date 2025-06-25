"""
Date: 2024-11-12 14:00:50
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-15 12:18:56
FilePath: /MineStudio/minestudio/tutorials/offline/1_finetune_vpts/main.py
"""

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
from minestudio.data import MineDataModule
from minestudio.models import load_vpt_policy
from minestudio.offline import MineLightning
from minestudio.offline.lightning_callbacks import SmartCheckpointCallback, SpeedMonitorCallback
from minestudio.offline.mine_callbacks import BehaviorCloneCallback
from minestudio.offline.utils import convert_to_normal
from swanlab.integration.pytorch_lightning import SwanLabLogger


@hydra.main(config_path=".", config_name="vpt_config")
def main(args):
    mine_lightning = MineLightning(
        mine_policy=load_vpt_policy(
            model_path=args.model_path,
            weights_path=args.weights_path,
        ),
        log_freq=20,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        callbacks=[
            BehaviorCloneCallback(weight=args.objective_weight),
        ],
        hyperparameters=convert_to_normal(args),
    )

    mine_data = MineDataModule(
        data_params=convert_to_normal(args.data),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_ratio=args.split_ratio,
        shuffle_episodes=args.shuffle_episodes,
        episode_continuous_batch=args.episode_continuous_batch,
    )
    logger = SwanLabLogger(project="MinecraftOptimus", experiment_name=args.experiment_name)

    L.Trainer(
        logger=logger,
        devices=args.devices,
        precision=args.precision,
        strategy="ddp_find_unused_parameters_true",
        use_distributed_sampler=not args.episode_continuous_batch,
        gradient_clip_val=1.0,
        callbacks=[
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
        ],
    ).fit(model=mine_lightning, datamodule=mine_data)


if __name__ == "__main__":
    main()
