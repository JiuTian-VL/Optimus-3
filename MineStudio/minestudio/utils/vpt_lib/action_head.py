from typing import Any, Dict, Optional, Tuple

import gym
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from gym3.types import ValType


LOG0 = -100


def fan_in_linear(module: nn.Module, scale=1.0, bias=True):
    """Fan-in init"""
    module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

    if bias:
        module.bias.data *= 0


class ActionHead(nn.Module):
    """Abstract base class for action heads compatible with forc"""

    def forward(self, input_data, **kwargs) -> Any:
        """
        Just a forward pass through this head
        :returns pd_params - parameters describing the probability distribution
        """
        raise NotImplementedError

    def logprob(self, action_sample, pd_params, **kwargs):
        """Logartithm of probability of sampling `action_sample` from a probability described by `pd_params`"""
        raise NotImplementedError

    def entropy(self, pd_params):
        """Entropy of this distribution"""
        raise NotImplementedError

    def sample(self, pd_params, deterministic: bool = False) -> Any:
        """
        Draw a sample from probability distribution given by those params

        :param pd_params Parameters of a probability distribution
        :param deterministic Whether to return a stochastic sample or deterministic mode of a distribution
        """
        raise NotImplementedError

    def kl_divergence(self, params_q, params_p):
        """KL divergence between two distribution described by these two params"""
        raise NotImplementedError


class DiagGaussianActionHead(ActionHead):
    """
    Action head where actions are normally distributed uncorrelated variables with specific means and variances.

    Means are calculated directly from the network while standard deviations are a parameter of this module
    """

    LOG2PI = np.log(2.0 * np.pi)

    def __init__(self, input_dim: int, num_dimensions: int):
        super().__init__()

        self.input_dim = input_dim
        self.num_dimensions = num_dimensions

        self.linear_layer = nn.Linear(input_dim, num_dimensions)
        self.log_std = nn.Parameter(torch.zeros(num_dimensions), requires_grad=True)

    def reset_parameters(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        assert not mask, "Can not use a mask in a gaussian action head"
        means = self.linear_layer(input_data)
        # Unsqueeze many times to get to the same shape
        logstd = self.log_std[(None,) * (len(means.shape) - 1)]

        mean_view, logstd = torch.broadcast_tensors(means, logstd)

        return torch.stack([mean_view, logstd], dim=-1)

    def logprob(self, action_sample: torch.Tensor, pd_params: torch.Tensor) -> torch.Tensor:
        """Log-likelihood"""
        means = pd_params[..., 0]
        log_std = pd_params[..., 1]

        std = torch.exp(log_std)

        z_score = (action_sample - means) / std

        return -(0.5 * ((z_score**2 + self.LOG2PI).sum(dim=-1)) + log_std.sum(dim=-1))

    def entropy(self, pd_params: torch.Tensor) -> torch.Tensor:
        """
        Categorical distribution entropy calculation - sum probs * log(probs).
        In case of diagonal gaussian distribution - 1/2 log(2 pi e sigma^2)
        """
        log_std = pd_params[..., 1]
        return (log_std + 0.5 * (self.LOG2PI + 1)).sum(dim=-1)

    def sample(self, pd_params: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        means = pd_params[..., 0]
        log_std = pd_params[..., 1]

        if deterministic:
            return means
        else:
            return torch.randn_like(means) * torch.exp(log_std) + means

    def kl_divergence(self, params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)

        Formula is:
        log(sigma_p) - log(sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2))/(2 * sigma_p^2)
        """
        means_q = params_q[..., 0]
        log_std_q = params_q[..., 1]

        means_p = params_p[..., 0]
        log_std_p = params_p[..., 1]

        std_q = torch.exp(log_std_q)
        std_p = torch.exp(log_std_p)

        kl_div = log_std_p - log_std_q + (std_q**2 + (means_q - means_p) ** 2) / (2.0 * std_p**2) - 0.5

        return kl_div.sum(dim=-1, keepdim=True)


class CategoricalActionHead(ActionHead):
    """Action head with categorical actions"""

    def __init__(
        self,
        input_dim: int,
        shape: Tuple[int],
        num_actions: int,
        builtin_linear_layer: bool = True,
        temperature: float = 1.0,
        nucleus_prob: Optional[float] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.output_shape = shape + (num_actions,)
        self.temperature = temperature
        self.nucleus_prob = nucleus_prob

        if builtin_linear_layer:
            self.linear_layer = nn.Linear(input_dim, np.prod(self.output_shape))
        else:
            assert input_dim == num_actions, (
                f"If input_dim ({input_dim}) != num_actions ({num_actions}), you need a linear layer to convert them."
            )
            self.linear_layer = None

    def reset_parameters(self):
        if self.linear_layer is not None:
            init.orthogonal_(self.linear_layer.weight, gain=0.01)
            init.constant_(self.linear_layer.bias, 0.0)
            finit.fan_in_linear(self.linear_layer, scale=0.01)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> Any:
        if self.linear_layer is not None:
            flat_out = self.linear_layer(input_data)
        else:
            flat_out = input_data
        shaped_out = flat_out.reshape(flat_out.shape[:-1] + self.output_shape)
        shaped_out /= self.temperature
        if mask is not None:
            shaped_out[~mask] = LOG0

        # Convert to float32 to avoid RuntimeError: "log_softmax_lastdim_kernel_impl" not implemented for 'Half'
        return F.log_softmax(shaped_out.float(), dim=-1)

    def logprob(self, actions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        value = actions.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        result = log_pmf.gather(-1, value).squeeze(-1)
        # result is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            result = result.sum(dim=-1)
        return result

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Categorical distribution entropy calculation - sum probs * log(probs)"""
        probs = torch.exp(logits)
        entropy = -torch.sum(probs * logits, dim=-1)
        # entropy is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            entropy = entropy.sum(dim=-1)
        return entropy

    # minecraft domain should directly use this sample function
    def vanilla_sample(self, logits: torch.Tensor, deterministic: bool = False, **kwargs) -> Any:
        """The original sample function from VPT library."""
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            # logits = torch.nn.functional.log_softmax(logits, dim=-1)
            # Gumbel-Softmax trick.
            u = torch.rand_like(logits)
            # In float16, if you have around 2^{float_mantissa_bits} logits, sometimes you'll sample 1.0
            # Then the log(-log(1.0)) will give -inf when it should give +inf
            # This is a silly hack to get around that.
            # This hack does not skew the probability distribution, because this event can't possibly win the argmax.
            u[u == 1.0] = 0.999

            return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)

    def nucleus_sample(self, logits: torch.Tensor, deterministic: bool = False, p: float = 0.85, **kwargs) -> Any:
        """The nucleus sample function."""
        if deterministic:
            return self.vanilla_sample(logits, deterministic)
        probs = torch.exp(logits)
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
        nucleus = cum_sum_probs < p
        nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[~nucleus] = float("-inf")
        sampled_sorted_indexes = self.vanilla_sample(sorted_log_probs, deterministic=False)
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
        return res.squeeze(-1)

    def sample(self, logits: torch.Tensor, deterministic: bool = False, **kwargs) -> Any:
        if self.nucleus_prob is None:
            return self.vanilla_sample(logits, deterministic, **kwargs)
        else:
            return self.nucleus_sample(logits, deterministic, p=self.nucleus_prob, **kwargs)

    def kl_divergence(self, logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)
        When talking about logits this is:
        sum exp(Q_i) * (Q_i - P_i)
        """
        kl = (torch.exp(logits_q) * (logits_q - logits_p)).sum(-1, keepdim=True)
        # kl is per-entry, still of size self.output_shape; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            kl = kl.sum(dim=-2)  # dim=-2 because we use keepdim=True above.
        return kl


class MSEActionHead(ActionHead):
    def __init__(self, input_dim: int, num_dimensions: int):
        super().__init__()

        self.input_dim = input_dim
        self.num_dimensions = num_dimensions

        self.linear_layer = nn.Linear(input_dim, num_dimensions)

    def reset_parameters(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        assert not mask, "Can not use a mask in a mse action head"
        means = self.linear_layer(input_data)

        return means

    def logprob(self, action_sample: torch.Tensor, pd_params: torch.Tensor) -> torch.Tensor:
        return -((action_sample - pd_params).pow(2)).sum(dim=-1)

    def entropy(self, pd_params: torch.Tensor) -> torch.Tensor:
        # raise NotImplementedError("Entropy is not defined for MSE action head")
        return torch.zeros_like(pd_params).sum(dim=-1)

    def sample(self, pd_params: torch.Tensor, deterministic: bool = False, **kwargs) -> torch.Tensor:
        return pd_params

    def kl_divergence(self, params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("KL divergence is not defined for MSE action head")


class TupleActionHead(nn.ModuleList, ActionHead):
    """Action head with multiple sub-actions"""

    def reset_parameters(self):
        for subhead in self:
            subhead.reset_parameters()

    def forward(self, input_data: torch.Tensor, **kwargs) -> Any:
        return tuple([subhead(input_data) for subhead in self])

    def logprob(self, actions: Tuple[torch.Tensor], logits: Tuple[torch.Tensor]) -> torch.Tensor:
        return tuple([subhead.logprob(actions[k], logits[k]) for k, subhead in enumerate(self)])

    def sample(self, logits: Tuple[torch.Tensor], deterministic: bool = False) -> Any:
        return tuple([subhead.sample(logits[k], deterministic) for k, subhead in enumerate(self)])

    def entropy(self, logits: Tuple[torch.Tensor]) -> torch.Tensor:
        return tuple([subhead.entropy(logits[k]) for k, subhead in enumerate(self)])

    def kl_divergence(self, logits_q: Tuple[torch.Tensor], logits_p: Tuple[torch.Tensor]) -> torch.Tensor:
        return sum(subhead.kl_divergence(logits_q[k], logits_p[k]) for k, subhead in enumerate(self))


class DictActionHead(nn.ModuleDict, ActionHead):
    """Action head with multiple sub-actions"""

    def reset_parameters(self):
        for subhead in self.values():
            subhead.reset_parameters()

    def forward(self, input_data: torch.Tensor, **kwargs) -> Any:
        """
        :param kwargs: each kwarg should be a dict with keys corresponding to self.keys()
                e.g. if this ModuleDict has submodules keyed by 'A', 'B', and 'C', we could call:
                    forward(input_data, foo={'A': True, 'C': False}, bar={'A': 7}}
                Then children will be called with:
                    A: forward(input_data, foo=True, bar=7)
                    B: forward(input_data)
                    C: forward(input_Data, foo=False)
        """
        result = {}
        for head_name, subhead in self.items():
            head_kwargs = {
                kwarg_name: kwarg[head_name]
                for kwarg_name, kwarg in kwargs.items()
                if kwarg is not None and head_name in kwarg
            }
            result[head_name] = subhead(input_data, **head_kwargs)
        return result

    def logprob(
        self, actions: Dict[str, torch.Tensor], logits: Dict[str, torch.Tensor], return_dict=False
    ) -> torch.Tensor:
        if return_dict:
            return {k: subhead.logprob(actions[k], logits[k]) for k, subhead in self.items()}
        else:
            return sum(subhead.logprob(actions[k], logits[k]) for k, subhead in self.items())

    def sample(self, logits: Dict[str, torch.Tensor], deterministic: bool = False) -> Any:
        return {k: subhead.sample(logits[k], deterministic) for k, subhead in self.items()}

    def entropy(self, logits: Dict[str, torch.Tensor], return_dict=False) -> torch.Tensor:
        if return_dict:
            return {k: subhead.entropy(logits[k]) for k, subhead in self.items()}
        else:
            return sum(subhead.entropy(logits[k]) for k, subhead in self.items())

    def kl_divergence(self, logits_q: Dict[str, torch.Tensor], logits_p: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(subhead.kl_divergence(logits_q[k], logits_p[k]) for k, subhead in self.items())


def make_action_head(ac_space: ValType, pi_out_size: int, temperature: float = 1.0, **kwargs):
    """Helper function to create an action head corresponding to the environment action space"""
    if isinstance(ac_space, gymnasium.spaces.MultiDiscrete):
        return CategoricalActionHead(
            pi_out_size, ac_space.shape, ac_space.nvec[0].item(), temperature=temperature, **kwargs
        )
    elif isinstance(ac_space, gymnasium.spaces.Dict):
        return DictActionHead({
            k: make_action_head(v, pi_out_size, temperature, **kwargs) for k, v in ac_space.items()
        })
    elif isinstance(ac_space, gymnasium.spaces.Tuple):
        return TupleActionHead([make_action_head(v, pi_out_size, temperature, **kwargs) for v in ac_space])
    elif isinstance(ac_space, gym.spaces.Discrete):
        return CategoricalActionHead(pi_out_size, ac_space.shape, ac_space.n, temperature=temperature, **kwargs)
    elif isinstance(ac_space, gym.spaces.Box) or isinstance(ac_space, gymnasium.spaces.Box):
        assert len(ac_space.shape) == 1, "Nontrivial shapes not yet implemented."
        return MSEActionHead(pi_out_size, ac_space.shape[0], **kwargs)
    raise NotImplementedError(f"Action space of type {type(ac_space)} is not supported")


# def make_action_head(ac_space: ValType, pi_out_size: int, temperature: float = 1.0):
#     """Helper function to create an action head corresponding to the environment action space"""
#     if isinstance(ac_space, TensorType):
#         if isinstance(ac_space.eltype, Discrete):
#             return CategoricalActionHead(pi_out_size, ac_space.shape, ac_space.eltype.n, temperature=temperature)
#         elif isinstance(ac_space.eltype, Real):
#             if temperature != 1.0:
#                 logging.warning("Non-1 temperature not implemented for DiagGaussianActionHead.")
#             assert len(ac_space.shape) == 1, "Nontrivial shapes not yet implemented."
#             return DiagGaussianActionHead(pi_out_size, ac_space.shape[0])
#     elif isinstance(ac_space, DictType):
#         return DictActionHead({k: make_action_head(v, pi_out_size, temperature) for k, v in ac_space.items()})

#     raise NotImplementedError(f"Action space of type {type(ac_space)} is not supported")
