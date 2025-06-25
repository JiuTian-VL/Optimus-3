from collections import deque
from copy import deepcopy
import random

from minestudio.online.rollout.replay_buffer import ReplayBufferInterface
from minestudio.online.utils import auto_stack, auto_to_torch
from minestudio.online.utils.rollout.datatypes import FragmentIndex, SampleFragment
from typing import Any, Dict, List, Optional, Tuple
import torch
import ray
from ray.util.actor_pool import ActorPool


@ray.remote
class FragmentLoader:
    def __init__(self):
        self.replay_buffer = ReplayBufferInterface()

    def load(self, record: Tuple[FragmentIndex, str]) -> Dict[str, Any]:
        index, fragment_uuid = record
        fragment = self.replay_buffer.load_fragment(fragment_uuid)
        return {"index": index, "fragment": fragment}


def create_loader_pool(num_readers: int, num_cpus_per_reader: int):
    actors = [
        FragmentLoader.options(  # type: ignore
            placement_group=None, num_cpus=num_cpus_per_reader, resources={"database": 0.0001}
        ).remote()
        for _ in range(num_readers)
    ]  # type: ignore
    return ActorPool(actors)


def data_iter(
    loader_pool: ActorPool, records: List[Tuple[FragmentIndex, str]], batch_size: int, prefetch_batches: int
):
    records = records.copy()
    random.shuffle(records)

    accum = []
    num_received = 0

    records_on_the_fly = (prefetch_batches + 1) * batch_size
    records_to_submit = deque(records)
    for _ in range(records_on_the_fly):
        if len(records_to_submit) == 0:
            break
        loader_pool.submit(lambda actor, record: actor.load.remote(record), records_to_submit.popleft())

    while num_received < len(records):
        accum.append(
            deepcopy(loader_pool.get_next_unordered())
            # It seems that ray will not release the object from its plasma store (accounted in SHR column of htop) until all references to its memory are gone.
            # The following code may keep some metadata of the fragment (e.g. fragment.next_done). While these data are quite small, ray will keep the whole fragment in plasma store, if we don't deepcopy it.
        )
        num_received += 1
        if len(records_to_submit) > 0:
            loader_pool.submit(lambda actor, record: actor.load.remote(record), records_to_submit.popleft())
        if len(accum) >= batch_size:
            yield auto_stack(accum[:batch_size])
            accum = accum[batch_size:]
    if len(accum) > 0:
        yield auto_stack(accum)


def prepare_batch(model, batch_fragments: List[SampleFragment]):
    _obs, _state, _action, _first = [], [], [], []
    device = model.device
    for f in batch_fragments:
        _obs.append(f.obs)
        _state.append(f.in_state)
        _action.append(f.action)
        _first.append(f.first)
    obs = auto_to_torch(auto_stack(_obs), device=device)
    state = model.merge_state(auto_to_torch(_state, device=device))
    action = auto_to_torch(auto_stack(_action), device=device)
    first = auto_to_torch(auto_stack(_first), device=device)
    return {
        "obs": obs,
        "state": state,
        "action": action,
        "first": first,
    }


def batchify_next_obs(next_obs: Dict[str, Any], device: torch.device):
    _obs = auto_stack([auto_stack([next_obs])])
    obs = auto_to_torch(_obs, device=device)
    return obs
