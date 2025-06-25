from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Union
import torch
import logging
from minestudio.online.utils import auto_stack, auto_to_torch
from typing import Optional

logger = logging.getLogger("ray")


@dataclass(frozen=True)
class FragmentIndex:
    worker_uuid: str
    fid_in_worker: int


@dataclass
class SampleFragment:
    obs: Union[Dict[str, Any], torch.Tensor]
    action: Union[Dict[str, Any], torch.Tensor]
    next_done: np.ndarray
    reward: np.ndarray
    first: np.ndarray
    in_state: List[np.ndarray]
    worker_uuid: str
    fid_in_worker: int
    next_obs: Dict[str, Any]
    episode_uuids: List[str]

    @property
    def index(self) -> FragmentIndex:
        return FragmentIndex(worker_uuid=self.worker_uuid, fid_in_worker=self.fid_in_worker)

    def print(self) -> None:
        logger.info(f"FragmentIndex: {self.index}, obs: {self.obs}")
        logger.info(f"FragmentIndex: {self.index}, action: {self.action}")
        logger.info(f"FragmentIndex: {self.index}, next_done: {self.next_done}")
        logger.info(f"FragmentIndex: {self.index}, reward: {self.reward}")
        logger.info(f"FragmentIndex: {self.index}, first: {self.first}")
        logger.info(f"FragmentIndex: {self.index}, in_state: {self.in_state}")
        logger.info(f"FragmentIndex: {self.index}, worker_uuid: {self.worker_uuid}")
        logger.info(f"FragmentIndex: {self.index}, fid_in_worker: {self.fid_in_worker}")
        logger.info(f"FragmentIndex: {self.index}, next_obs: {self.next_obs}")
        logger.info(f"FragmentIndex: {self.index}, episode_uuids: {self.episode_uuids}")


class FragmentDataDict(Dict[FragmentIndex, Any]):
    def format_batch(self, fragments: List[SampleFragment], device: torch.device):
        return auto_to_torch(auto_stack([self[f.index] for f in fragments]), device=device)


@dataclass(frozen=True)
class FragmentMetadata:
    model_version: int
    session_id: str
    worker_uuid: str
    fid_in_worker: int


@dataclass
class StepRecord:
    worker_uuid: str
    obs: Dict[str, Any]
    state: Optional[List[np.ndarray]]
    action: Dict[str, Any]
    last_reward: float
    last_terminated: bool
    last_truncated: bool
    model_version: int
    episode_uuid: str
    session_id: str
