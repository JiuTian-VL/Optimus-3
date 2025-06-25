"""
Date: 2024-11-10 10:26:32
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-12-17 06:14:29
FilePath: /MineStudio/minestudio/data/minecraft/part_raw.py
"""

import json
import math
import random
from typing import Any, Literal, Mapping, Tuple

import numpy as np
import torch

from minestudio.data.minecraft.core import BaseDataset


class RawDataset(BaseDataset):
    """Raw dataset for training and testing."""

    def __init__(
        self,
        win_len: int = 1,
        skip_frame: int = 1,
        split: Literal["train", "val"] = "train",
        split_ratio: float = 0.8,
        verbose: bool = True,
        shuffle: bool = False,
        task_description_file: str | None = None,
        filter_label: str | None = None,
        **kernel_kwargs,
    ) -> Any:
        super(RawDataset, self).__init__(verbose=verbose, **kernel_kwargs)
        self.win_len = win_len
        self.skip_frame = skip_frame
        self.split = split
        self.split_ratio = split_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.task_description_file = task_description_file
        self.episode_task_map = {}
        if self.task_description_file is not None:
            self.task_description_file = self.task_description_file.split(",")
        if filter_label is not None:
            self.filter_label = filter_label.split(",")
        else:
            self.filter_label = None
        self.build_items()

    def build_items(self) -> None:
        if self.task_description_file is not None:
            for task_file in self.task_description_file:
                with open(task_file, "r") as f:
                    data = json.load(f)
                for sample in data:
                    end = sample.get("end", -1)
                    if self.filter_label is not None and sample["label"] in self.filter_label:
                        continue
                    if end == -1:
                        self.episode_task_map[sample["id"]] = {
                            "task": sample["task"],
                            "label": sample["label"],
                        }
                    else:
                        if sample["id"] not in self.episode_task_map:
                            self.episode_task_map[sample["id"]] = []
                        self.episode_task_map[sample["id"]].append({
                            "task": sample["task"],
                            "label": sample["label"],
                            "start": sample["start"],
                            "end": sample["end"],
                        })
        for k in self.episode_task_map.keys():
            if isinstance(self.episode_task_map[k], list):
                self.episode_task_map[k] = sorted(self.episode_task_map[k], key=lambda x: x["start"], reverse=False)

        self.episodes_with_length = self.kernel.get_episodes_with_length()
        _episodes_with_length = list(self.episodes_with_length.items())
        if self.task_description_file is not None:
            _episodes_with_length = [(k, v) for k, v in _episodes_with_length if k in self.episode_task_map]

        if self.shuffle:
            seed = 0
            print(f"[Raw Dataset] Shuffling episodes with seed {seed}. ")
            random.seed(seed)  # ensure the same shuffle order for all workers
            random.shuffle(_episodes_with_length)

        divider = int(len(_episodes_with_length) * self.split_ratio)
        if self.split == "train":
            _episodes_with_length = _episodes_with_length[:divider]
        else:
            _episodes_with_length = _episodes_with_length[divider:]

        self.items = []
        self.num_items = 0
        self.episodes_with_items = []
        for episode, length in _episodes_with_length:
            num_episode_items = (length + self.win_len - 1) // self.win_len
            self.episodes_with_items.append((episode, num_episode_items, self.num_items))
            self.num_items += num_episode_items
            self.items.append((self.num_items, episode))

    def locate_item(self, idx: int) -> Tuple[str, int]:
        """Find the first episode that idx > acc[episode]"""
        left, right = 0, len(self.items)
        while left < right:
            mid = (left + right) // 2
            if self.items[mid][0] <= idx:
                left = mid + 1
            else:
                right = mid
        if left == 0:
            relative_idx = idx
        else:
            relative_idx = idx - self.items[left - 1][0]
        episode = self.items[left][1]
        return episode, relative_idx

    def __len__(self) -> int:
        return self.num_items

    def _query_episode_info(self, episode_info: list[dict], start: int, end: int) -> dict:
        min_distance = int(1e9)
        min_window_info = None
        for window_info in episode_info:
            window_start = window_info["start"]
            window_end = window_info["end"]
            if start >= window_start and end <= window_end + 1:
                return window_info
            distance = min(abs(start - window_start), abs(end - window_end))
            if distance < min_distance:
                min_distance = distance
                min_window_info = window_info
        return min_window_info

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        assert idx < len(self), f"Index <{idx}> out of range <{len(self)}>"
        episode, relative_idx = self.locate_item(idx)
        start = max(1, relative_idx * self.win_len)  # start > 0 is the prequest for previous action
        item = self.kernel.read(episode, start, self.win_len, self.skip_frame)
        episode_info = self.episode_task_map.get(episode, {"task": "raw", "label": "<null>"})
        if isinstance(episode_info, list):
            window_info = self._query_episode_info(episode_info, start + 1, start + 1 + self.win_len)
            item["task"] = window_info["task"]
            item["label"] = window_info["label"]
        else:
            item["task"] = episode_info["task"]
            item["label"] = episode_info["label"]
        assert item["task"] != "raw", f"the task should not be raw! {episode}"
        item["timestamp"] = np.arange(start, start + self.win_len, self.skip_frame)
        item["episode"] = episode
        episode_samples = math.ceil(self.episodes_with_length[episode] / self.win_len)
        item["progress"] = f"{relative_idx}/{episode_samples}"
        item = self.postprocess(item)
        return item


if __name__ == "__main__":
    kernel_kwargs = dict(
        dataset_dirs=[
            "/nfs-shared-2/data/contractors/dataset_6xx",
            "/nfs-shared-2/data/contractors/dataset_7xx",
            "/nfs-shared-2/data/contractors/dataset_8xx",
            "/nfs-shared-2/data/contractors/dataset_9xx",
            "/nfs-shared-2/data/contractors/dataset_10xx",
        ],
        enable_contractor_info=False,
        enable_segment=True,
    )

    dataset = RawDataset(
        frame_width=224,
        frame_height=224,
        win_len=128,
        skip_frame=1,
        split="train",
        split_ratio=0.9,
        verbose=True,
        **kernel_kwargs,
    )

    from minestudio.data.minecraft.utils import MineDistributedBatchSampler

    sampler = MineDistributedBatchSampler(dataset, batch_size=4, num_replicas=1, rank=0, shuffle=False, drop_last=True)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    for idx, batch in enumerate(loader):
        print("\t".join([f"{a} {b}" for a, b in zip(batch["episode"], batch["progress"])]))
        if idx > 50:
            break
