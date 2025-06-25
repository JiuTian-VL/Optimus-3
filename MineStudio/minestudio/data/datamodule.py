"""
Date: 2024-11-10 12:31:33
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-25 15:56:34
FilePath: /MineStudio/minestudio/data/datamodule.py
"""

from typing import Dict, Optional

import lightning.pytorch as pl
from rich import print
from torch.utils.data import DataLoader

from minestudio.data.minecraft.dataset import MinecraftDataset
from minestudio.data.minecraft.optimus2_dataset import MinecraftDataset as Optimus2MinecraftDataset
from minestudio.data.minecraft.utils import MineDistributedBatchSampler, batchify


class MineDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_params: Dict,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle_episodes: bool = False,
        prefetch_factor: Optional[int] = None,
        episode_continuous_batch: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_params = data_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_episodes = shuffle_episodes
        self.prefetch_factor = prefetch_factor
        self.episode_continuous_batch = episode_continuous_batch
        self.kwargs = kwargs
        self.is_optimus2_pretrain = kwargs.get("is_optimus2_pretrain", False)

    def setup(self, stage: Optional[str] = None):
        if self.is_optimus2_pretrain:
            self.train_dataset = Optimus2MinecraftDataset(
                split="train", shuffle=self.shuffle_episodes, **self.data_params, **self.kwargs
            )
        else:
            self.train_dataset = MinecraftDataset(
                split="train", shuffle=self.shuffle_episodes, **self.data_params, **self.kwargs
            )
            self.val_dataset = MinecraftDataset(
                split="val", shuffle=self.shuffle_episodes, **self.data_params, **self.kwargs
            )

    def train_dataloader(self):
        if self.episode_continuous_batch:
            assert self.data_params["mode"] == "raw", "episode_continuous_batch only support raw mode"
            # using MineDistributedBatchSampler for loading continuous video frames
            batch_sampler = MineDistributedBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
            )
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
                drop_last=True,
            )
        return train_loader

    def val_dataloader(self):
        if self.episode_continuous_batch:
            assert self.data_params["mode"] == "raw", "episode_continuous_batch only support raw mode"
            # using MineDistributedBatchSampler for loading continuous video frames
            batch_sampler = MineDistributedBatchSampler(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
            )
            val_loader = DataLoader(
                dataset=self.val_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
            )
        else:
            val_loader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=batchify,
                prefetch_factor=self.prefetch_factor,
                pin_memory=True,
                drop_last=True,
            )
        return val_loader


if __name__ == "__main__":
    import lightning as L
    from tqdm import tqdm

    fabric = L.Fabric(accelerator="cuda", devices=1, strategy="ddp")
    fabric.launch()
    data_module = MineDataModule(
        data_params=dict(
            mode="raw",
            dataset_dirs=[
                "/nfs-shared-2/data/contractors/dataset_6xx",
                "/nfs-shared-2/data/contractors/dataset_7xx",
                "/nfs-shared-2/data/contractors/dataset_8xx",
                "/nfs-shared-2/data/contractors/dataset_9xx",
                "/nfs-shared-2/data/contractors/dataset_10xx",
            ],
            enable_contractor_info=False,
            enable_segment=True,
            frame_width=224,
            frame_height=224,
            win_len=128,
            skip_frame=1,
            split_ratio=0.8,
        ),
        batch_size=1,
        num_workers=2,
        shuffle_episodes=True,
        prefetch_factor=4,
        episode_continuous_batch=False,
    )
    data_module.setup()
    # for i in range(5):
    #     print(i, data_module.train_dataset[i]['image'].float().mean())
    # exit()
    train_loader = data_module.train_dataloader()
    train_loader = fabric.setup_dataloaders(train_loader, use_distributed_sampler=False)
    rank = fabric.local_rank
    for idx, batch in enumerate(tqdm(train_loader, disable=True)):
        # print(idx, batch["image"].float().mean())
        # if idx > 5:
        #     break
        print(f"{rank = } \t" + "\t".join([f"{a[-20:]} {b}" for a, b in zip(batch["episode"], batch["progress"])]))
