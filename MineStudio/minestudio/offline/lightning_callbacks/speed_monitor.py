"""
Date: 2024-11-28 15:35:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-28 15:37:52
FilePath: /MineStudio/minestudio/train/lightning_callbacks/speed_monitor.py
"""

import time

import lightning.pytorch as pl


class SpeedMonitorCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        INTERVAL = 16
        if trainer.global_rank != 0 or batch_idx % INTERVAL != 0:
            return
        now = time.time()

        if hasattr(self, "time_start"):
            time_cost = now - self.time_start
            trainer.logger.log_metrics({"train/speed(batch/s)": INTERVAL / time_cost}, step=trainer.global_step)
            self.time_start = now
        else:
            self.time_start = now
