from queue import Queue
import ray
import numpy as np
from ray.actor import ActorHandle
from omegaconf import DictConfig
from minestudio.online.rollout.rollout_worker import RolloutWorker
from minestudio.online.utils import auto_stack, auto_to_numpy
from minestudio.online.utils.rollout.datatypes import SampleFragment
from minestudio.online.rollout.replay_buffer import ReplayBufferInterface
from minestudio.online.rollout.episode_statistics import EpisodeStatistics
from minestudio.online.utils.rollout.datatypes import FragmentMetadata, StepRecord
from typing import Dict, Union, Any, List, Optional, Callable
import torch
from dataclasses import dataclass
from collections import defaultdict
from minestudio.models import MinePolicy
from minestudio.simulator import MinecraftSim

WRAPPER_CONCURRENCY = 3


@ray.remote(max_concurrency=WRAPPER_CONCURRENCY)  # type: ignore
class RolloutWorkerWrapper:
    def __init__(
        self,
        fragment_length: int,
        policy_generator: Callable[[], MinePolicy],
        env_generator: Callable[[], MinecraftSim],
        worker_config: DictConfig,
        episode_statistics: ActorHandle,
        to_send_queue_size: int,
        use_normalized_vf: bool,
        resume: Optional[str],
        max_staleness: int,
        rollout_worker_id: int,
    ):
        self.max_staleness = max_staleness
        self.fragment_length = fragment_length
        self.use_normalized_vf = use_normalized_vf
        self.next_model_version = -1
        self.rollout_worker = RolloutWorker(
            model_device="cuda",
            progress_handler=self.progress_handler,
            episode_statistics=episode_statistics,
            use_normalized_vf=self.use_normalized_vf,
            resume=resume,
            next_model_version=self.next_model_version,
            rollout_worker_id=rollout_worker_id,
            policy_generator=policy_generator,
            env_generator=env_generator,
            **worker_config,
        )
        self.buffer: Dict[str, List[StepRecord]] = defaultdict(list)
        self.current_model_version = -1
        self.current_session_id = ""
        self.next_session_id = ""
        self.next_model_version = -1
        self.next_state_dict = None
        self.num_fragments = defaultdict(int)
        self.to_send_queue = Queue(to_send_queue_size)
        self.replay_buffer = ReplayBufferInterface()

    def update_model(self, session_id: str, model_version: int, packed_state_dict_ref: List[ray.ObjectRef]):
        assert len(packed_state_dict_ref) == 1
        self.next_state_dict = ray.get(packed_state_dict_ref[0])
        self.next_model_version = model_version
        self.next_session_id = session_id
        self.rollout_worker.update_model_version(self.next_model_version)

    def progress_handler(
        self,
        *,
        worker_uuid: str,
        obs: Dict[str, Any],
        state: List[torch.Tensor],
        action: Dict[str, Any],
        last_reward: float,
        last_terminated: bool,
        last_truncated: bool,
        episode_uuid: str,
    ) -> None:
        if len(self.buffer[worker_uuid]) > 0 and (
            self.current_model_version - self.buffer[worker_uuid][-1].model_version > self.max_staleness
            or self.current_session_id != self.buffer[worker_uuid][-1].session_id
        ):
            self.buffer[worker_uuid] = []

        self.buffer[worker_uuid].append(
            StepRecord(
                worker_uuid=worker_uuid,
                obs=obs,
                state=None,
                action=action,
                last_reward=last_reward,
                last_terminated=last_terminated,
                last_truncated=last_truncated,
                model_version=self.current_model_version,
                episode_uuid=episode_uuid,
                session_id=self.current_session_id,
            )
        )

        steps_to_send = None
        if len(self.buffer[worker_uuid]) > self.fragment_length:
            assert len(self.buffer[worker_uuid]) == self.fragment_length + 1
            steps_to_send = self.buffer[worker_uuid][: self.fragment_length]
            self.buffer[worker_uuid] = self.buffer[worker_uuid][self.fragment_length :]
            self.to_send_queue.put((steps_to_send, self.buffer[worker_uuid][0]))

        if len(self.buffer[worker_uuid]) == 1:
            self.buffer[worker_uuid][0].state = auto_to_numpy(state)  # [s.cpu().numpy() for s in state]

    def _send(self, steps: List[StepRecord], next_step: StepRecord):
        last_next_done = next_step.last_terminated or next_step.last_truncated
        rewards = [step.last_reward for step in steps][1:] + [next_step.last_reward]
        next_done = [step.last_terminated or step.last_truncated for step in steps][1:] + [last_next_done]
        assert steps[0].state is not None

        fragment = SampleFragment(
            obs=auto_stack([step.obs for step in steps]),
            action=auto_stack([step.action for step in steps]),
            next_done=np.array(next_done, dtype=np.bool_),
            reward=np.array(rewards, dtype=np.float32),
            first=np.array([step.last_terminated or step.last_truncated for step in steps], dtype=np.bool_),
            episode_uuids=[step.episode_uuid for step in steps],
            in_state=steps[0].state,
            worker_uuid=steps[0].worker_uuid,
            fid_in_worker=self.num_fragments[steps[0].worker_uuid],
            next_obs=next_step.obs,
        )

        self.num_fragments[fragment.worker_uuid] += 1

        assert fragment.reward.shape[0] == self.fragment_length

        model_version = steps[0].model_version
        session_id = steps[0].session_id

        self.replay_buffer.add_fragment(
            fragment=fragment,
            metadata=FragmentMetadata(
                session_id=session_id,
                model_version=model_version,
                worker_uuid=fragment.worker_uuid,
                fid_in_worker=fragment.fid_in_worker,
            ),
        )

    def rollout_thread(self):
        while True:
            if self.next_state_dict is not None:
                self.rollout_worker.load_weights(self.next_state_dict)
                self.current_model_version = self.next_model_version
                self.current_session_id = self.next_session_id
                self.next_state_dict = None
            self.rollout_worker.loop()

    def sender_thread(self):
        while True:
            steps_to_send, next_step = self.to_send_queue.get()
            self._send(steps_to_send, next_step)


class _RolloutManager:
    def __init__(
        self,
        policy_generator: Callable[[], MinePolicy],
        env_generator: Callable[[], MinecraftSim],
        use_normalized_vf: bool,
        discount: float,
        num_rollout_workers: int,
        num_cpus_per_worker: int,
        num_gpus_per_worker: int,
        to_send_queue_size: int,
        fragment_length: int,
        resume: Optional[str],
        replay_buffer_config: DictConfig,
        worker_config: DictConfig,
        episode_statistics_config: DictConfig,
    ):
        """
        This class is responsible for creating and managing a group of rollout workers.

        Args:
            num_rollout_workers (int): number of rollout workers to create
            num_cpus_per_worker (int): number of cpus to allocate for each rollout worker
            fragment_length (int): number of steps in each fragment (this should be equal to the context length of the model)
            replay_buffer_config (DictConfig): configuration for the replay buffer
            worker_config (DictConfig): configuration for rollout workers
            episode_statistics_config (DictConfig): configuration for episode statistics
        """
        self.num_rollout_workers = num_rollout_workers
        self.num_cpus_per_worker = num_cpus_per_worker
        self.fragment_length = fragment_length
        self.replay_buffer_config = replay_buffer_config
        self.episode_statistics_config = episode_statistics_config
        self.worker_config = worker_config
        self.policy_generator = policy_generator
        self.env_generator = env_generator
        self.num_gpus_per_worker = num_gpus_per_worker
        self.to_send_queue_size = to_send_queue_size
        self.use_normalized_vf = use_normalized_vf

        self.replay_buffer = ReplayBufferInterface(self.replay_buffer_config)

        self.episode_statistics = EpisodeStatistics.remote(
            discount=discount,  # type: ignore
            **self.episode_statistics_config,
        )

        import pickle

        try:
            sim = env_generator()
            pickle.dumps(sim)
            print("env_generator is pickleable")
            del sim
        except Exception as e:
            ray.util.pdb.set_trace

        self.rollout_workers = [
            RolloutWorkerWrapper.options(  # type: ignore
                num_cpus=self.num_cpus_per_worker,
                num_gpus=self.num_gpus_per_worker,
                max_concurrency=WRAPPER_CONCURRENCY,
            ).remote(
                fragment_length=self.fragment_length,
                policy_generator=self.policy_generator,
                env_generator=self.env_generator,
                resume=resume,
                worker_config=self.worker_config,
                episode_statistics=self.episode_statistics,
                max_staleness=self.replay_buffer_config.max_staleness,
                use_normalized_vf=self.use_normalized_vf,
                to_send_queue_size=self.to_send_queue_size,
                rollout_worker_id=rollout_worker_id,
            )
            for rollout_worker_id in range(self.num_rollout_workers)
        ]

    def start(self):
        for worker in self.rollout_workers:
            worker.rollout_thread.remote()
            worker.sender_thread.remote()

    def update_model(self, session_id: str, model_version: int, packed_state_dict_ref: List[ray.ObjectRef]):
        assert len(packed_state_dict_ref) == 1
        remotes = [
            worker.update_model.remote(
                session_id=session_id, model_version=model_version, packed_state_dict_ref=packed_state_dict_ref
            )
            for worker in self.rollout_workers
        ]
        ray.get(remotes)

        self.replay_buffer.update_model_version(session_id=session_id, model_version=model_version)

    def log_statistics(self, step: int, record_next_episode: bool):
        print("received_request in menager:" + str(step) + str(record_next_episode))
        ray.get(self.episode_statistics.log_statistics.remote(step, record_next_episode))

    def get_replay_buffer_config(self):
        return self.replay_buffer_config

    def update_training_session(self):
        ray.get(self.episode_statistics.update_training_session.remote())
        self.replay_buffer.update_training_session()


@ray.remote
class RolloutManager(_RolloutManager):
    def __init__(self, **kwargs):
        self.saved_config = kwargs
        super().__init__(**kwargs)

    def get_saved_config(self):
        return self.saved_config
