"""
Date: 2024-11-10 11:01:51
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-12 10:55:34
FilePath: /MineStudio/minestudio/data/minecraft/demo.py
"""

import hydra
from rich.console import Console
from torch.utils.data import DataLoader

from minestudio.data.minecraft.part_event import EventDataset
from minestudio.data.minecraft.part_raw import RawDataset
from minestudio.data.minecraft.utils import MineDistributedBatchSampler, batchify, visualize_dataloader


def visualize_raw_dataset(args):
    raw_dataset = RawDataset(
        dataset_dirs=args.dataset_dirs,
        enable_video=args.enable_video,
        enable_action=args.enable_action,
        enable_contractor_info=args.enable_contractor_info,
        enable_segment=args.enable_segment,
        win_len=args.win_len,
        skip_frame=args.skip_frame,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        enable_augmentation=args.enable_augmentation,
        enable_resize=args.enable_resize,
        shuffle=args.shuffle,
    )
    Console().log(f"num-workers: {args.num_workers}")
    batch_sampler = MineDistributedBatchSampler(
        dataset=raw_dataset,
        batch_size=args.batch_size,
        num_replicas=1,
        rank=0,
    )
    dataloader = DataLoader(
        dataset=raw_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=batchify,
    )

    visualize_dataloader(
        dataloader,
        num_samples=args.num_samples,
        resolution=(args.frame_width, args.frame_height),
        legend=args.legend,
        save_fps=args.save_fps,
    )


def visualize_event_dataset(args):
    event_dataset = EventDataset(
        dataset_dirs=args.dataset_dirs,
        enable_video=args.enable_video,
        enable_action=args.enable_action,
        enable_contractor_info=args.enable_contractor_info,
        enable_segment=args.enable_segment,
        win_len=args.win_len,
        skip_frame=args.skip_frame,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        enable_resize=args.enable_resize,
        enable_augmentation=args.enable_augmentation,
        event_regex=args.event_regex,
        min_nearby=args.min_nearby,
        max_within=args.max_within,
        bias=args.bias,
    )
    Console().log(f"num-workers: {args.num_workers}")
    dataloader = DataLoader(
        dataset=event_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        collate_fn=batchify,
    )

    # dump_trajectories(
    visualize_dataloader(
        dataloader,
        num_samples=args.num_samples,
        resolution=(args.frame_width, args.frame_height),
        legend=args.legend,
        save_fps=args.save_fps,
    )


@hydra.main(config_path="demo_configs", config_name="type-raw")
def main(args):
    if args.dataset_type == "event":
        visualize_event_dataset(args)
    elif args.dataset_type == "raw":
        visualize_raw_dataset(args)
    else:
        raise ValueError(f"Unknown dataset_type: {args.dataset_type}")


if __name__ == "__main__":
    main()
