import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import ModelHubMixin, constants, hf_hub_download
from minestudio.utils.mineclip_lib.mineclip.mineclip import MineCLIP
from safetensors.torch import load_file, save_file

from .config import MINECLIP_CONFIG, PRIOR_INFO
from .vae import TranslatorVAE
from .MineRLConditionalAgent import MineRLConditionalAgent
from .utils.embed_utils import get_prior_embed


FPS = 20


class Optimus2ActionAgent(ModelHubMixin):
    def __init__(
        self,
        text_cond_scale: float = 6.0,
        visual_cond_scale: float = 7.0,
        prior_config=PRIOR_INFO,
        mineclip_config=MINECLIP_CONFIG,
        agent_pi_head_config=None,
        agent_policy_config=None,
    ) -> None:
        self.text_cond_scale = text_cond_scale
        self.visual_cond_scale = visual_cond_scale
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.mineclip = load_mineclip_wconfig(self.device)
        self.mineclip = MineCLIP(**mineclip_config)
        self.mineclip.to(self.device)
        mineclip_dim = prior_config["mineclip_dim"]
        latent_dim = prior_config["latent_dim"]
        hidden_dim = prior_config["hidden_dim"]
        # self.prior = load_vae_model(PRIOR_INFO, self.device)
        self.prior = TranslatorVAE(input_dim=mineclip_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.prior.to(self.device)
        # self.prior.to(self.device)

        self.agent = MineRLConditionalAgent(
            device=self.device,
            policy_kwargs=agent_policy_config,
            pi_head_kwargs=agent_pi_head_config,
        )
        # self.agent.load_weights(in_weights)
        self.agent.reset(text_cond_scale)

        self.mllm_embed_linear = torch.nn.Sequential(
            torch.nn.Linear(2048, 2048 * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(2048 * 4, 512),
        )
        self.mllm_embed_linear.to(self.device)

        # ckpt = torch.load(linear_ckpt_path, map_location="cpu")["state_dict"]
        # state_dict = {k.replace("mllm_embed_linear.", ""): v for k, v in ckpt.items() if "mllm_embed_linear." in k}
        # self.mllm_embed_linear.load_state_dict(state_dict, strict=True)

    def to(self, device):
        """Move the model to the specified device."""
        self.device = device
        self.mineclip.to(device)
        self.prior.to(device)
        self.agent.to(device)
        self.mllm_embed_linear.to(device)

    def _get_prompt_embed(self, prompt: str) -> Any:
        prompt_embed = get_prior_embed(prompt, self.mineclip, self.prior, self.device)
        self.prompt_embed = {prompt: prompt_embed}
        return self.prompt_embed

    def action(self, prompt: str, obs):
        minerl_obs = {"pov": obs}
        prompt_embed = self._get_prompt_embed(prompt)

        for _, embed in prompt_embed.items():
            minerl_action = self.agent.get_action(minerl_obs, embed)

        for k, v in minerl_action.items():
            minerl_action[k] = np.array(v.tolist()[0])
        minerl_action["ESC"] = np.array(0)
        return minerl_action

    def optimus_action(self, embed, obs, task: str):
        embed = self.mllm_embed_linear(embed)  # [bs,1,512]
        minerl_obs = {"pov": obs}

        _prompt_embed = get_prior_embed(task, self.mineclip, self.prior, self.device)
        embed = embed.reshape(*_prompt_embed.shape)
        embed = self.prior(embed, deterministic=False)

        # calculate the cosine similarity between the prompt and the prior
        # cosine = torch.nn.functional.cosine_similarity(
        #     embed, torch.from_numpy(_prompt_embed).to(embed.device), dim=-1
        # ).mean()
        # print(f"cosine: {cosine.item()}")s

        # if cosine < 0.99:
        #     print(f"cosine: {cosine.item()}")

        minerl_action, agent_action = self.agent.get_action(minerl_obs, embed)

        for k, v in minerl_action.items():
            minerl_action[k] = np.array(v.tolist()[0])
        minerl_action["ESC"] = np.array(0)
        return minerl_action, agent_action

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        all_state_dict = {}
        for k, v in self.mllm_embed_linear.state_dict().items():
            all_state_dict[f"mllm_embed_linear.{k}"] = v.cpu().clone()
        for k, v in self.agent.policy.state_dict().items():
            all_state_dict[f"agent.policy.{k}"] = v.cpu().clone()
        for k, v in self.prior.state_dict().items():
            all_state_dict[f"prior.{k}"] = v.cpu().clone()
        for k, v in self.mineclip.state_dict().items():
            all_state_dict[f"mineclip.{k}"] = v.cpu().clone()

        save_file(all_state_dict, save_directory / constants.SAFETENSORS_SINGLE_FILE)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs,
    ):
        def _load_weight(model, model_file: str, map_location: str, strict: bool):
            state_dict = load_file(model_file, map_location)
            mineclip_state_dict = {k.replace("mineclip.", ""): v for k, v in state_dict.items() if "mineclip." in k}
            model.mineclip.load_state_dict(mineclip_state_dict, strict=strict)
            prior_state_dict = {k.replace("prior.", ""): v for k, v in state_dict.items() if "prior." in k}
            model.prior.load_state_dict(prior_state_dict, strict=strict)
            mllm_embed_linear_state_dict = {
                k.replace("mllm_embed_linear.", ""): v for k, v in state_dict.items() if "mllm_embed_linear." in k
            }
            model.mllm_embed_linear.load_state_dict(mllm_embed_linear_state_dict, strict=strict)
            agent_policy_state_dict = {
                k.replace("agent.policy.", ""): v for k, v in state_dict.items() if "agent.policy." in k
            }
            model.agent.policy.load_state_dict(agent_policy_state_dict, strict=strict)
            return model

        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=constants.SAFETENSORS_SINGLE_FILE,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        return _load_weight(model, model_file, map_location, strict)


class Optimus3ActionAgent(ModelHubMixin):
    def __init__(
        self,
        text_cond_scale: float = 6.0,
        visual_cond_scale: float = 7.0,
        prior_config=PRIOR_INFO,
        mineclip_config=MINECLIP_CONFIG,
        agent_pi_head_config=None,
        agent_policy_config=None,
    ) -> None:
        self.text_cond_scale = text_cond_scale
        self.visual_cond_scale = visual_cond_scale
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # self.mineclip = load_mineclip_wconfig(self.device)
        self.mineclip = MineCLIP(**mineclip_config)
        self.mineclip.to(self.device)
        mineclip_dim = prior_config["mineclip_dim"]
        latent_dim = prior_config["latent_dim"]
        hidden_dim = prior_config["hidden_dim"]
        # self.prior = load_vae_model(PRIOR_INFO, self.device)
        self.prior = TranslatorVAE(input_dim=mineclip_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.prior.to(self.device)
        # self.prior.to(self.device)

        self.agent = MineRLConditionalAgent(
            device=self.device,
            policy_kwargs=agent_policy_config,
            pi_head_kwargs=agent_pi_head_config,
        )
        # self.agent.load_weights(in_weights)
        self.agent.reset(text_cond_scale)

        self.mllm_embed_linear = torch.nn.Sequential(
            torch.nn.Linear(3584, 3584 * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(3584 * 2, 512),
        )
        self.mllm_embed_linear.to(self.device)

        # ckpt = torch.load(linear_ckpt_path, map_location="cpu")["state_dict"]
        # state_dict = {k.replace("mllm_embed_linear.", ""): v for k, v in ckpt.items() if "mllm_embed_linear." in k}
        # self.mllm_embed_linear.load_state_dict(state_dict, strict=True)

    def to(self, device):
        """Move the model to the specified device."""
        self.device = device
        self.mineclip.to(device)
        self.prior.to(device)
        self.agent.to(device)
        self.mllm_embed_linear.to(device)

    def _get_prompt_embed(self, prompt: str) -> Any:
        prompt_embed = get_prior_embed(prompt, self.mineclip, self.prior, self.device)
        self.prompt_embed = {prompt: prompt_embed}
        return self.prompt_embed

    def action(self, prompt: str, obs):
        minerl_obs = {"pov": obs}
        prompt_embed = self._get_prompt_embed(prompt)

        for _, embed in prompt_embed.items():
            minerl_action = self.agent.get_action(minerl_obs, embed)

        for k, v in minerl_action.items():
            minerl_action[k] = np.array(v.tolist()[0])
        minerl_action["ESC"] = np.array(0)
        return minerl_action

    def optimus3_action(self, embed, obs, task: str):
        embed = self.mllm_embed_linear(embed)  # [bs,1,512]
        minerl_obs = {"pov": obs}

        _prompt_embed = get_prior_embed(task, self.mineclip, self.prior, self.device)
        embed = embed.reshape(*_prompt_embed.shape)
        embed = self.prior(embed, deterministic=False)

        # calculate the cosine similarity between the prompt and the prior
        # cosine = torch.nn.functional.cosine_similarity(
        #     embed, torch.from_numpy(_prompt_embed).to(embed.device), dim=-1
        # ).mean()
        # print(f"cosine: {cosine.item()}")s

        # if cosine < 0.99:
        #     print(f"cosine: {cosine.item()}")

        minerl_action, agent_action = self.agent.get_action(minerl_obs, embed)

        for k, v in minerl_action.items():
            minerl_action[k] = np.array(v.tolist()[0])
        minerl_action["ESC"] = np.array(0)
        return minerl_action, agent_action

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        all_state_dict = {}
        for k, v in self.mllm_embed_linear.state_dict().items():
            all_state_dict[f"mllm_embed_linear.{k}"] = v.cpu().clone()
        for k, v in self.agent.policy.state_dict().items():
            all_state_dict[f"agent.policy.{k}"] = v.cpu().clone()
        for k, v in self.prior.state_dict().items():
            all_state_dict[f"prior.{k}"] = v.cpu().clone()
        for k, v in self.mineclip.state_dict().items():
            all_state_dict[f"mineclip.{k}"] = v.cpu().clone()

        save_file(all_state_dict, save_directory / constants.SAFETENSORS_SINGLE_FILE)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        map_location: str = "cpu",
        strict: bool = True,
        **model_kwargs,
    ):
        def _load_weight(model, model_file: str, map_location: str, strict: bool):
            state_dict = load_file(model_file, map_location)
            mineclip_state_dict = {k.replace("mineclip.", ""): v for k, v in state_dict.items() if "mineclip." in k}
            model.mineclip.load_state_dict(mineclip_state_dict, strict=strict)
            prior_state_dict = {k.replace("prior.", ""): v for k, v in state_dict.items() if "prior." in k}
            model.prior.load_state_dict(prior_state_dict, strict=strict)
            mllm_embed_linear_state_dict = {
                k.replace("mllm_embed_linear.", ""): v for k, v in state_dict.items() if "mllm_embed_linear." in k
            }
            model.mllm_embed_linear.load_state_dict(mllm_embed_linear_state_dict, strict=strict)
            agent_policy_state_dict = {
                k.replace("agent.policy.", ""): v for k, v in state_dict.items() if "agent.policy." in k
            }
            model.agent.policy.load_state_dict(agent_policy_state_dict, strict=strict)
            return model

        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=constants.SAFETENSORS_SINGLE_FILE,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        return _load_weight(model, model_file, map_location, strict)
