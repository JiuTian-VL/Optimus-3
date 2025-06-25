from collections import defaultdict
import os
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import ray
import ray.data
from ray.actor import ActorHandle
from omegaconf import DictConfig
import torch
import torch.distributed
import random
from minestudio.online.rollout.replay_buffer import ReplayBufferInterface
from minestudio.online.utils.train import get_current_session_id

from minestudio.online.utils import auto_getitem, auto_to_numpy, auto_slice, auto_cat
from minestudio.online.utils.rollout.datatypes import FragmentDataDict, FragmentIndex, SampleFragment

# import minestudio.online.utils.registry as registry
import uuid
from minestudio.models import MinePolicy
from minestudio.simulator import MinecraftSim
from minestudio.online.utils.train.gae import GAEWorker, get_last_fragment_indexes
from minestudio.online.utils.train.data import batchify_next_obs, prepare_batch, data_iter, create_loader_pool
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from ray.experimental import tqdm_ray
from minestudio.online.utils import auto_stack


class BaseTrainer:
    def __init__(
        self,
        rollout_manager: ActorHandle,
        policy_generator: Callable[[], MinePolicy],
        env_generator: Callable[[], MinecraftSim],
        num_workers: int,
        num_readers: int,
        num_cpus_per_reader: int,
        num_gpus_per_worker: int,
        prefetch_batches: int,
        discount: float,
        gae_lambda: float,
        context_length: int,
        use_normalized_vf: bool,
        inference_batch_size_per_gpu: int,
        resume: Optional[str],
        resume_optimizer: bool,
        **kwargs,
    ):
        self.rollout_manager = rollout_manager
        self.policy_generator = policy_generator
        self.env_generator = env_generator
        self.num_workers = num_workers
        self.num_readers = num_readers
        self.prefetch_batches = prefetch_batches
        self.num_cpus_per_reader = num_cpus_per_reader
        self.num_gpus_per_worker = num_gpus_per_worker
        self.inference_batch_size_per_gpu = inference_batch_size_per_gpu
        self.use_normalized_vf = use_normalized_vf
        self.context_length = context_length

        print("Warning: resume is not implemented")
        self.resume = resume
        self.resume_optimizer = resume_optimizer
        self.num_updates = 0
        self.num_optimized = 0

        self.gae_actor: ActorHandle = GAEWorker.remote(  # type: ignore
            discount=discount,  # type: ignore
            gae_lambda=gae_lambda,
        )

    def broadcast_model_to_rollout_workers(self, new_version):
        if self.rank == 0:
            if new_version:
                self.model_version += 1
            state_dict_ref = ray.put({key: value.cpu() for key, value in self.inner_model.state_dict().items()})
            ray.get(
                self.rollout_manager.update_model.remote(
                    session_id=get_current_session_id(),
                    model_version=self.model_version,
                    packed_state_dict_ref=[state_dict_ref],
                )
            )

    def fetch_fragments_and_estimate_advantages(self, *, num_fragments: int) -> Dict[str, Any]:
        torch.distributed.barrier()

        if self.rank == 0:
            ray.get(self.gae_actor.reset.remote())
            start_get = time.time()
            self.replay_buffer.fetch_fragments(num_fragments)
            end_get = time.time()

            logging.getLogger("ray").info(f"Prepared {num_fragments} fragments in {end_get - start_get} seconds.")

        torch.distributed.barrier()

        _all_records = self.replay_buffer.prepared_fragments()
        records = _all_records[self.rank :: self.num_workers]  # type: ignore

        last_fragment_indexs = set(get_last_fragment_indexes([r[0] for r in _all_records]))

        old_logps, old_pi_logits, vpreds, next_obs_vpreds = (
            FragmentDataDict(),
            FragmentDataDict(),
            FragmentDataDict(),
            FragmentDataDict(),
        )
        rewards = FragmentDataDict()

        gae_infos: Dict[FragmentIndex, Dict[str, Any]] = defaultdict(dict)

        # condition = self.model.load_condition(self.model_spec, )
        with torch.inference_mode():
            it = data_iter(
                loader_pool=self.loader_pool,
                records=records,
                batch_size=self.inference_batch_size_per_gpu,
                prefetch_batches=self.prefetch_batches,
            )
            if self.rank == 0:
                it = tqdm_ray.tqdm(
                    it, desc="Inference before GAE", total=len(records) // self.inference_batch_size_per_gpu
                )
            for _batch in it:
                add_idd = str(self.num_updates) + "for_base_trainer" + str(random.randint(0, 1000000))

                fragments: List[SampleFragment] = _batch["fragment"]  # type: ignore
                indexs: List[FragmentIndex] = _batch["index"]  # type: ignore

                batch = prepare_batch(
                    self.inner_model,
                    fragments,
                )
                # TODO: reduce VRAM usage

                B, T = batch["first"].shape

                new_state = batch["state"]

                # chunked forward
                _forward_results = []
                for start in range(0, T, self.context_length):
                    end = min(start + self.context_length, T)
                    chunk_obs = auto_slice(batch["obs"], start, end, dim=1, type_list=1)
                    chunk_first = auto_slice(batch["first"], start, end, dim=1, type_list=1)
                    # hack: change dimension of list to make obs better in: (batch, videos)
                    try:
                        forward_result, new_state = self.model(
                            input=chunk_obs, state_in=new_state, context={"first": chunk_first}
                        )
                    except:
                        ray.util.pdb.set_trace()

                    if torch.isnan(forward_result["pi_logits"]["buttons"]).any():
                        ray.util.pdb.set_trace()

                    _forward_results.append(forward_result)
                forward_result = auto_cat(_forward_results, dim=1)

                with torch.no_grad():
                    logp = self.inner_model.pi_head.logprob(batch["action"], forward_result["pi_logits"])
                    # pi_logits should be [1,128,1,8641]?
                    # ray.util.pdb.set_trace()
                    logging.getLogger("ray").info(f"logp's shape: {logp.shape}")
                pi_logits: Union[Dict[str, torch.Tensor], torch.Tensor] = forward_result["pi_logits"]
                vpred = forward_result["vpred"].reshape(B, T)
                if torch.isnan(vpred).any():
                    ray.util.pdb.set_trace()
                for i, index, fragment in zip(range(len(indexs)), indexs, fragments):
                    gae_infos[index]["reward"] = fragment.reward
                    gae_infos[index]["next_done"] = fragment.next_done

                    rewards[index] = fragment.reward.sum()
                    vpreds[index] = vpred[i].cpu().numpy()

                    if self.use_normalized_vf:
                        denormalized_vpred = self.inner_model.value_head.denormalize(vpred).reshape(B, T)  # type: ignore
                        gae_infos[index]["vpred"] = denormalized_vpred[i].cpu().numpy()  # type: ignore
                    else:
                        gae_infos[index]["vpred"] = vpreds[index]
                    old_logps[index] = logp[i].cpu().numpy()
                    old_pi_logits[index] = auto_to_numpy(auto_getitem(pi_logits, i))

                # process next_obs
                for i, index, fragment in zip(range(len(indexs)), indexs, fragments):
                    if index in last_fragment_indexs:
                        next_obs = batchify_next_obs(fragment.next_obs, self.inner_model.device)
                        next_in_state = new_state  # [s[i].unsqueeze(0) for s in new_state] # TODO: clone?
                        next_first = torch.tensor(
                            [[fragment.next_done[-1]]], device=self.inner_model.device, dtype=torch.bool
                        )
                        (
                            next_forward_result,
                            _,
                        ) = self.model(input=next_obs, state_in=next_in_state, context={"first": next_first})

                        # next_pi_latent, next_vf_latent = next_latents['pi_latent'], next_latents['vf_latent']
                        # with torch.no_grad():
                        #     next_pi_logits = self.inner_model.pi_head()(next_pi_latent)
                        #     next_vpred = self.inner_model.value_head(next_vf_latent)xw
                        # next_forward_result["pi_logits"] = next_pi_logits
                        # next_forward_result["vpred"] = next_vpredxwxw

                        next_obs_vpreds[index] = next_forward_result["vpred"][0][0].cpu().numpy().item()
                        if self.use_normalized_vf:
                            gae_infos[index]["next_vpred"] = (
                                self.inner_model.value_head.denormalize(  # type: ignore
                                    next_forward_result["vpred"]
                                )[0][0]
                                .cpu()
                                .numpy()
                                .item()  # type: ignore
                            )
                        else:
                            gae_infos[index]["next_vpred"] = next_obs_vpreds[index]
        ray.get(self.gae_actor.update_gae_infos.remote(gae_infos))  # type: ignore

        torch.distributed.barrier()

        if self.rank == 0:
            ray.get(self.gae_actor.calculate_target.remote())

        torch.distributed.barrier()
        indexs = [r[0] for r in records]
        td_targets, advantages = ray.get(self.gae_actor.get_target.remote(indexs))
        return {
            "records": records,
            "rewards": rewards,
            "td_targets": td_targets,
            "advantages": advantages,
            "old_logps": old_logps,
            "old_pi_logits": old_pi_logits,
            "old_vpreds": vpreds,
        }

    def setup_model_and_optimizer(self) -> Tuple[MinecraftSim, torch.optim.Optimizer]:
        raise NotImplementedError

    def _train_func(self, config):
        self.rank = ray.train.get_context().get_world_rank()

        self.replay_buffer = ReplayBufferInterface()
        self.loader_pool = create_loader_pool(self.num_readers, self.num_cpus_per_reader)

        self.inner_model, self.optimizer = self.setup_model_and_optimizer(self.policy_generator)
        assert not isinstance(self.inner_model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
        assert self.inner_model.training

        if self.resume:
            logging.getLogger("ray").info(f"Resuming from {self.resume}")
            state_dict_model = torch.load(
                os.path.join(self.resume, "model.ckpt"), map_location=self.inner_model.device
            )
            self.inner_model.load_state_dict(state_dict_model, strict=False)
            del state_dict_model
            # optimizer: see below
        self.model: torch.nn.Module = ray.train.torch.prepare_model(
            self.inner_model, parallel_strategy="ddp", parallel_strategy_kwargs={"find_unused_parameters": True}
        )
        assert self.model is self.inner_model or self.model.module is self.inner_model
        if self.resume and self.resume_optimizer:
            self.optimizer.load_state_dict(
                torch.load(os.path.join(self.resume, "optimizer.ckpt"), map_location=self.inner_model.device)
            )

        self.model_version = 0
        self.broadcast_model_to_rollout_workers(new_version=True)
        self.train()

    def train(self):
        raise NotImplementedError

    def fit(self):
        print("Entering fit")
        trainer = TorchTrainer(
            self._train_func,
            scaling_config=ScalingConfig(
                num_workers=self.num_workers, resources_per_worker={"GPU": self.num_gpus_per_worker}, use_gpu=True
            ),
        )
        logging.getLogger("ray").info("Start training...")
        trainer.fit()
