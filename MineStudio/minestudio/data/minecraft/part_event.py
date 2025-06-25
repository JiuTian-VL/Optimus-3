"""
Date: 2024-11-10 10:26:52
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-12-30 20:55:41
FilePath: /MineStudio/minestudio/data/minecraft/part_event.py
"""

import pickle
import re
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple, Union

import lmdb
import torch
from rich.console import Console

from minestudio.data.minecraft.core import BaseDataset


class EventLMDBDriver:
    def __init__(
        self,
        event_path: Union[str, Path],
        event_regex: str,
        min_nearby: Optional[int] = None,
        max_within: Optional[int] = None,
    ) -> None:
        if isinstance(event_path, str):
            event_path = Path(event_path)
        assert event_path.is_dir(), f"Event lmdb file {event_path} does not exist. "
        self.lmdb_stream = lmdb.open(str(event_path), max_readers=128, readonly=True, lock=False)

        with self.lmdb_stream.begin(write=False) as txn:
            __event_info__ = pickle.loads(txn.get(b"__event_info__"))
            # check if codebook exists
            __codebook_bytes__ = txn.get(b"__codebook__", None)
            if __codebook_bytes__ is None:
                self.__codebook__ = None
            else:
                self.__codebook__ = {v: k for k, v in pickle.loads(__codebook_bytes__).items()}
            self.event_info = {}
            for event, value in __event_info__.items():
                if re.match(event_regex, event):
                    self.event_info[event] = value

        self.event_list = sorted(list(self.event_info.keys()))
        # if min_nearby is not None or max_within is not None:
        self.filter_out(min_nearby, max_within)

    def filter_out(self, min_nearby: Optional[int] = None, max_within: Optional[int] = None):
        episode_event_last = {}
        remaining_events = {}
        for event in self.event_list:
            num_events = self.get_event_size(event)
            remaining_events[event] = []
            for i in range(num_events):
                episode, event_time, value = self.get_event_item(event, i)
                if event_time < 128:  # remove dirty events
                    continue
                episode_event_key = f"{episode}:{event}"
                if episode_event_key not in episode_event_last:
                    episode_event_last[episode_event_key] = -100000

                if min_nearby is not None and event_time - episode_event_last[episode_event_key] <= min_nearby:
                    continue

                if max_within is not None and len(remaining_events[event]) >= max_within:
                    break

                episode_event_last[episode_event_key] = event_time
                remaining_events[event].append(i)
            self.event_info[event]["__num_items__"] = len(remaining_events[event])
        self.remaining_events = remaining_events

    def get_event_list(self) -> List[str]:
        return self.event_list

    def get_event_size(self, event: str) -> int:
        if event not in self.event_info:
            return 0
        return self.event_info[event]["__num_items__"]

    def get_event_item(self, event: str, item_idx: int) -> Tuple[str, int, int]:
        assert item_idx < self.get_event_size(event), f"Item index {item_idx} out of range. "
        if hasattr(self, "remaining_events"):
            item_idx = self.remaining_events[event][item_idx]  # remap the index
        key = str((event, item_idx))
        with self.lmdb_stream.begin(write=False) as txn:
            item = pickle.loads(txn.get(key.encode()))
        episode, event_time, value = item
        if self.__codebook__ is not None:
            episode = self.__codebook__[episode]
        return episode, event_time, value


class EventKernel:
    def __init__(self, event_path: List[Union[str, Path]], event_regex: str, verbose: bool = True, **kwargs) -> None:
        self.verbose = verbose
        self.event_drivers = [EventLMDBDriver(event, event_regex, **kwargs) for event in event_path]
        event_set = set()
        for driver in self.event_drivers:
            event_set.update(driver.get_event_list())
        self.event_list = sorted(list(event_set))
        if verbose:
            Console().log(f"[Event Kernel] Number of loaded events: {len(self.event_list)}")

    def get_event_list(self) -> List[str]:
        return self.event_list

    def get_event_size(self, event: str) -> int:
        if event not in self.event_list:
            return 0
        return sum([driver.get_event_size(event) for driver in self.event_drivers])

    def get_event_item(self, event: str, item_idx: int) -> Tuple[str, int, int]:
        for driver in self.event_drivers:
            size = driver.get_event_size(event)
            if item_idx < size:
                return driver.get_event_item(event, item_idx)
            item_idx -= size
        raise ValueError(f"Item index {item_idx} out of range. ")


class EventDataset(BaseDataset):
    def __init__(
        self,
        win_len: int = 1,
        skip_frame: int = 1,
        split: Literal["train", "val"] = "train",
        split_ratio: float = 0.8,
        verbose: bool = True,
        # below are event dataset specific parameters
        bias: int = 0,
        event_regex: str = "",
        min_nearby: Optional[int] = None,  # minimal avaliable distance between two selected events
        max_within: Optional[int] = None,  # maximum number of samples within each event
        **kernel_kwargs,
    ) -> Any:
        super(EventDataset, self).__init__(verbose=verbose, **kernel_kwargs)
        self.win_len = win_len
        self.skip_frame = skip_frame
        self.split = split
        self.split_ratio = split_ratio
        self.verbose = verbose
        self.bias = bias
        self.event_regex = event_regex
        self.event_kernel = EventKernel(
            event_path=[Path(x) / "event" for x in kernel_kwargs["dataset_dirs"]],
            event_regex=event_regex,
            verbose=verbose,
            min_nearby=min_nearby,
            max_within=max_within,
        )

        self.build_items()

    def build_items(self) -> None:
        self.event_list = self.event_kernel.get_event_list()

        self.num_items = 0
        event_with_items = []
        for event in self.event_list:
            num_event_items = self.event_kernel.get_event_size(event)
            if self.split == "train":
                bias = 0
                num_event_items = int(num_event_items * self.split_ratio)
            elif self.split == "val":
                bias = int(num_event_items * self.split_ratio)
                num_event_items = num_event_items - bias
            else:
                raise ValueError(f"Split type <{self.split}> not supported. ")
            self.num_items += num_event_items
            event_with_items.append((self.num_items, event, bias))
        self.items = event_with_items

        if self.verbose:
            Console().log(
                f"[Event Dataset] Regex: {self.event_regex}, Number of events: {len(self.event_list)}, number of items: {self.num_items}"
            )

    def locate_item(self, idx: int) -> Tuple[str, int]:
        """Find the first event that idx > acc[event]"""
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
        event = self.items[left][1]
        bias = self.items[left][2]
        return event, relative_idx + bias

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        assert idx < len(self), f"Index <{idx}> out of range <{len(self)}>"
        event, relative_idx = self.locate_item(idx)
        episode, event_time, value = self.event_kernel.get_event_item(event, relative_idx)
        start = max(event_time - self.win_len + self.bias, 0)
        item = self.kernel.read(episode, start=start, win_len=self.win_len, skip_frame=self.skip_frame)
        item["text"] = event.replace("minecraft.", "")
        return self.postprocess(item)


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

    event_dataset = EventDataset(
        win_len=128,
        skip_frame=1,
        split="train",
        split_ratio=0.8,
        verbose=True,
        event_regex="minecraft.kill_entity:.*",
        **kernel_kwargs,
    )

    item = event_dataset[0]

    import ipdb

    ipdb.set_trace()
