"""
Date: 2024-11-28 15:37:18
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-15 12:17:37
FilePath: /MineStudio/minestudio/offline/lightning_callbacks/smart_checkpoint.py
"""

from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint

from minestudio.offline.lightning_callbacks.ema import EMA


class SmartCheckpointCallback(ModelCheckpoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _ema_callback(self, trainer: "pytorch_lightning.Trainer") -> Optional[EMA]:
        ema_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, EMA):
                ema_callback = callback
        return ema_callback

    def _ema_format_filepath(self, filepath: str) -> str:
        return filepath.replace(self.FILE_EXTENSION, f"-EMA{self.FILE_EXTENSION}")

    def _save_checkpoint(self, trainer: "pytorch_lightning.Trainer", filepath: str) -> None:
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # with ema_callback.save_original_optimizer_state(trainer):
            super()._save_checkpoint(trainer, filepath)

            # save EMA copy of the model as well.
            with ema_callback.save_ema_model(trainer):
                filepath = self._ema_format_filepath(filepath)
                if self.verbose:
                    rank_zero_info(f"Saving EMA weights to separate checkpoint {filepath}")
                super()._save_checkpoint(trainer, filepath)
        else:
            super()._save_checkpoint(trainer, filepath)

    def _remove_checkpoint(self, trainer: "pytorch_lightning.Trainer", filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        ema_callback = self._ema_callback(trainer)
        if ema_callback is not None:
            # remove EMA copy of the state dict as well.
            filepath = self._ema_format_filepath(filepath)
            super()._remove_checkpoint(trainer, filepath)
