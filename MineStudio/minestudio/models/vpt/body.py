"""
Date: 2024-11-11 20:54:15
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-04 13:32:36
FilePath: /MineStudio/minestudio/models/vpt/body.py
"""

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch as th
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F

from minestudio.models.base_policy import MinePolicy
from minestudio.online.utils import auto_stack, auto_to_torch
from minestudio.utils.register import Registers
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.tree_util import tree_map
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks


class ImgPreprocessing(nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = nn.Parameter(th.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = nn.Parameter(th.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        x = img.to(dtype=th.float32)
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x


class ImgObsProcess(nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        return self.linear(self.cnn(img))


class MinecraftPolicy(nn.Module):
    """
    :param recurrence_type:
        None                - No recurrence, adds no extra layers
        lstm                - (Depreciated). Singular LSTM
        multi_layer_lstm    - Multi-layer LSTM. Uses n_recurrence_layers to determine number of consecututive LSTMs
            Does NOT support ragged batching
        multi_masked_lstm   - Multi-layer LSTM that supports ragged batching via the first vector. This model is slower
            Uses n_recurrence_layers to determine number of consecututive LSTMs
        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(
        self,
        recurrence_type="lstm",
        impala_width=1,
        impala_chans=(16, 32, 32),
        obs_processing_width=256,
        hidsize=512,
        single_output=False,  # True if we don't need separate outputs for action/value outputs
        img_shape=None,
        scale_input_img=True,
        only_img_input=False,
        init_norm_kwargs={},
        impala_kwargs={},
        # Unused argument assumed by forc.
        input_shape=None,  # pylint: disable=unused-argument
        active_reward_monitors=None,
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False,
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,  # Not needed for transformer
        **unused_kwargs,
    ):
        super().__init__()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        active_reward_monitors = active_reward_monitors or {}

        self.single_output = single_output

        chans = tuple(int(impala_width * c) for c in impala_chans)
        self.hidsize = hidsize

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # Setup inputs
        self.img_preprocess = ImgPreprocessing(img_statistics=img_statistics, scale_img=scale_input_img)
        self.img_process = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=chans,
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs,
        )

        self.pre_lstm_ln = nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None

        self.recurrence_type = recurrence_type
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        )

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = th.nn.LayerNorm(hidsize)

    def output_latent_size(self):
        return self.hidsize

    def forward(self, ob, state_in, context):
        first = context["first"]
        x = self.img_preprocess(ob["image"])
        x = self.img_process(x)

        if self.diff_obs_process:
            processed_obs = self.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        else:
            state_out = state_in

        x = F.relu(x, inplace=False)

        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, state_out
        return (pi_latent, vf_latent), state_out

    def initial_state(self, batchsize):
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None


@Registers.model.register
class VPTPolicy(MinePolicy, PyTorchModelHubMixin):
    def __init__(self, policy_kwargs, action_space=None, **kwargs):
        super().__init__(hiddim=policy_kwargs["hidsize"], action_space=action_space, **kwargs)
        self.net = MinecraftPolicy(**policy_kwargs)
        self.cached_init_states = dict()

    def initial_state(self, batch_size: int = None):
        if batch_size is None:
            return [t.squeeze(0).to(self.device) for t in self.net.initial_state(1)]
        else:
            if batch_size not in self.cached_init_states:
                self.cached_init_states[batch_size] = [t.to(self.device) for t in self.net.initial_state(batch_size)]
            return self.cached_init_states[batch_size]


    @th.no_grad()
    def act(
        self,
        obs,
        first,
        state_in,
        stochastic: bool = True,
        taken_action=None,
        return_pd=False,
        cond_scale=None,
        **kwargs,
    ):
        if state_in is None:
            if cond_scale is None:
                state_in = self.initial_state(1)
            else:
                state_in = self.initial_state(2)
        if cond_scale is not None:
            bsz = obs["image"].shape[0]
            assert bsz == 1, "cond_scale only works for batch size 1"
            # Change the batch size to 2, and duplicate the first element.
            obs = tree_map(lambda x: th.cat([x, x], dim=0), obs)
            # Set the embedding on the second element to zeros.
            obs["mllm_embed"][1] = th.zeros_like(obs["mllm_embed"][1])
            obs["mllm_embed"] = obs["mllm_embed"].reshape(2, 1, -1)
            first = th.cat([first, first], dim=0)

        # We need to add a fictitious time dimension everywhere
        # obs = tree_map(lambda x: x.unsqueeze(1), obs)
        # first = first.unsqueeze(1)

        (pd, v_h), state_out = self.net(obs, state_in, context={"first": first})

        # Compute entropy of the action distribution (buttons only)
        buttons = pd["buttons"][0, 0, 0, :]
        softmax_buttons = th.nn.functional.softmax(buttons)
        self.entropy_last = -th.sum(softmax_buttons * th.log(softmax_buttons))

        if cond_scale is not None:
            # Combine the pytree elements using a weighted sum across the batch.
            # x[0]: conditional
            # x[1]: unconditional
            # cond_scale = 0: regular conditional policy
            # cond_scale > 0: subtract some of the unconditional policy
            pd = tree_map(
                lambda x: (((1 + cond_scale) * x[0]) - (cond_scale * x[1])).unsqueeze(0),
                pd,
            )

        if taken_action is None:
            ac = self.pi_head.sample(pd, deterministic=not stochastic)
        else:
            ac = tree_map(lambda x: x.unsqueeze(1), taken_action)
        log_prob = self.pi_head.logprob(ac, pd)
        assert not th.isnan(log_prob).any()

        # After unsqueezing, squeeze back to remove fictitious time dimension
        result = {
            "log_prob": log_prob[:, 0],
            "vpred": self.value_head.denormalize(v_h)[:, 0],
        }
        if return_pd:
            result["pd"] = tree_map(lambda x: x[:, 0], pd)
        ac = tree_map(lambda x: x[:, 0], ac)

        return ac, state_out


    def forward(self, input, state_in, **kwargs):
        B, T = input["image"].shape[:2]
        first = torch.tensor([[False]], device=self.device).repeat(B, T)
        state_in = self.initial_state(B) if state_in is None else state_in

        # input: 1, 128, 128, 128, 3
        # first: 1, 128
        # state_in[0]: 1, 1, 1, 128
        # state_in[1]: 1, 1, 128, 128
        try:
            (pi_h, v_h), state_out = self.net(input, state_in, context={"first": first})
        except Exception:
            import ray

            ray.util.pdb.set_trace()
        pi_logits = self.pi_head(pi_h)
        vpred = self.value_head(v_h)
        latents = {"pi_logits": pi_logits, "vpred": vpred}
        return latents, state_out

    def merge_input(self, inputs) -> torch.tensor:
        inputs = auto_to_torch(inputs, device=self.device)
        if inputs[0]["image"].dim() == 3:
            in_inputs = [{"image": input["image"]} for input in inputs]
            out_inputs = auto_to_torch(auto_stack([auto_stack([input]) for input in in_inputs]), device=self.device)
            return out_inputs
        elif inputs[0]["image"].dim() == 4:
            out_inputs = auto_to_torch(auto_stack([input["image"] for input in inputs]), device=self.device)
            return out_inputs

    def merge_state(self, states) -> Optional[List[torch.Tensor]]:
        result_states = []
        for i in range(len(states[0])):
            result_states.append(auto_to_torch(torch.cat([state[i] for state in states], 0), device=self.device))
        return result_states

    def split_state(self, states, split_num) -> Optional[List[List[torch.Tensor]]]:
        result_states = [[states[j][i : i + 1] for j in range(len(states))] for i in range(split_num)]
        return result_states


@Registers.model_loader.register
def load_vpt_policy(model_path: str, weights_path: Optional[str] = None):
    if model_path is None:
        if weights_path is None:
            repo_id = "CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x"
        return VPTPolicy.from_pretrained(f"{repo_id}")
    model = pickle.load(Path(model_path).open("rb"))
    policy_kwargs = model["model"]["args"]["net"]["args"]
    vpt_policy = VPTPolicy(
        policy_kwargs=policy_kwargs,
        temperature=model["model"]["args"]["pi_head_opts"]["temperature"],
    )
    if weights_path is None:
        return vpt_policy
    weights = torch.load(weights_path, map_location="cpu")
    if "state_dict" in weights:
        weights = {k.replace("mine_policy.", ""): v for k, v in weights["state_dict"].items()}
    weights = {k: v for k, v in weights.items() if k in vpt_policy.state_dict()}
    vpt_policy.load_state_dict(weights, strict=True)
    return vpt_policy


if __name__ == "__main__":
    # model = load_vpt_policy(
    #     model_path="/nfs-shared/jarvisbase/pretrained/foundation-model-2x.model",
    #     weights_path="/nfs-shared-2/hekaichen/minestudio_checkpoint/gate.ckpt"
    # ).to("cuda")
    # model.push_to_hub("CraftJarvis/MineStudio_VPT.rl_for_build_portal_2x")
    model = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_for_shoot_animals_2x").to("cuda")
    model.eval()
    dummy_input = {
        "image": torch.zeros(1, 1, 128, 128, 3).to("cuda"),
    }
    output, memory = model(dummy_input, None)
    print(output)
