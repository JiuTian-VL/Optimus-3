import pickle

import gym

from ..config import MINECLIP_CONFIG
from ..mineclip_code.load_mineclip import load
from ..MineRLConditionalAgent import MineRLConditionalAgent
from ..VPT.agent import ENV_KWARGS


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def load_mineclip_wconfig(device):
    # print("Loading MineClip...")
    return load(MINECLIP_CONFIG, device=device)


def make_env(seed):
    from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival

    print("Loading MineRL...")
    env = HumanSurvival(**ENV_KWARGS).make()
    print("Starting new env...")
    env.reset()
    if seed is not None:
        print(f"Setting seed to {seed}...")
        env.seed(seed)
    return env


def make_agent(in_model, in_weights, cond_scale):
    print(f"Loading agent with cond_scale {cond_scale}...")
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    env = gym.make("MineRLBasaltFindCave-v0")
    # Make conditional agent
    agent = MineRLConditionalAgent(
        env,
        device="cuda",
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(in_weights)
    agent.reset(cond_scale=cond_scale)
    env.close()
    return agent


def load_mineclip_agent_env(in_model, in_weights, seed, cond_scale):
    mineclip = load_mineclip_wconfig()
    agent = make_agent(in_model, in_weights, cond_scale=cond_scale)
    env = make_env(seed)
    return agent, mineclip, env
