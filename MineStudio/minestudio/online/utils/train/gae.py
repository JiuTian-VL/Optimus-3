import ray
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from minestudio.online.utils.rollout.datatypes import FragmentDataDict, FragmentIndex
import minestudio.online.utils.train.wandb_logger as wandb_logger
from collections import defaultdict
import logging


def get_last_fragment_indexes(fragment_indexs: List[FragmentIndex]) -> List[FragmentIndex]:
    fragment_indexs = fragment_indexs.copy()
    fragment_indexs.sort(key=lambda x: x.fid_in_worker)

    last_fragment_indexs = []
    _last_idx = {}
    for index in reversed(fragment_indexs):
        if not (index.worker_uuid in _last_idx) or _last_idx[index.worker_uuid] != index.fid_in_worker + 1:
            last_fragment_indexs.append(index)
        _last_idx[index.worker_uuid] = index.fid_in_worker

    return last_fragment_indexs


@ray.remote
class GAEWorker:
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
    ):
        self._gae_lengths = []
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.gae_infos: Dict[FragmentIndex, Dict[str, Any]] = {}

    def update_gae_infos(self, gae_infos: Dict[FragmentIndex, Dict[str, Any]]):
        self.gae_infos.update(gae_infos)

    def calculate_target(self):
        fragment_indexs = list(self.gae_infos.keys())
        fragment_indexs.sort(key=lambda x: x.fid_in_worker)

        self.td_targets, self.advantages = FragmentDataDict(), FragmentDataDict()

        _last_idx = {}
        _gae_length = {}
        fragment_count = defaultdict(int)
        last_advantage = defaultdict(float)
        last_next_vpred = defaultdict(float)

        for index in reversed(fragment_indexs):
            if not (index.worker_uuid in _last_idx) or _last_idx[index.worker_uuid] != index.fid_in_worker + 1:
                last_advantage[index.worker_uuid] = 0
                if "next_vpred" not in self.gae_infos[index]:
                    ray.util.pdb.set_trace()
                last_next_vpred[index.worker_uuid] = self.gae_infos[index]["next_vpred"]
                if index.worker_uuid in self._gae_lengths:
                    self._gae_lengths.append(_gae_length[index.worker_uuid])
                _gae_length[index.worker_uuid] = 0

            _last_idx[index.worker_uuid] = index.fid_in_worker

            vpred: np.ndarray = self.gae_infos[index]["vpred"]
            assert len(vpred.shape) == 1

            reward = self.gae_infos[index]["reward"]
            next_done = self.gae_infos[index]["next_done"]

            next_vpred = last_next_vpred[index.worker_uuid]
            last_gae_adv = last_advantage[index.worker_uuid]
            self.advantages[index] = np.zeros_like(vpred)
            self.td_targets[index] = np.zeros_like(vpred)

            _gae_length[index.worker_uuid] += len(next_done)

            fragment_count[index.worker_uuid] += 1
            for t in range(len(next_done) - 1, -1, -1):
                next_nonterminal = 1.0 - next_done[t]
                delta = reward[t] + self.discount * next_vpred * next_nonterminal - vpred[t]
                last_gae_adv = delta + self.discount * self.gae_lambda * next_nonterminal * last_gae_adv
                self.advantages[index][t] = last_gae_adv
                self.td_targets[index][t] = last_gae_adv + vpred[t]
                next_vpred = vpred[t]
                if np.isnan(self.td_targets[index][t]) or np.isnan(next_vpred):
                    ray.util.pdb.set_trace()
            last_advantage[index.worker_uuid] = last_gae_adv
            last_next_vpred[index.worker_uuid] = next_vpred
        self._gae_lengths += list(_gae_length.values())

        if len(self._gae_lengths) > 0:
            wandb_logger.log(
                {
                    "GAEWorker/average_gae_length": np.mean(self._gae_lengths),
                }
            )
            self._gae_lengths = []
        # self.print_episodes() # for debug

    def get_target(self, indexs: List[FragmentIndex]) -> Tuple[FragmentDataDict, FragmentDataDict]:
        td_targets, advantages = FragmentDataDict(), FragmentDataDict()
        for index in indexs:
            td_targets[index] = self.td_targets[index]
            advantages[index] = self.advantages[index]
        return td_targets, advantages
