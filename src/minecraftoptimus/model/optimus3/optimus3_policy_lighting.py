import os
from typing import List

import lightning as L
import torch
import torch.nn.functional as F
from minestudio.models import MinePolicy
from minestudio.offline.mine_callbacks import ObjectiveCallback
from rich import print
from transformers import AutoModelForVision2Seq, AutoProcessor

import minecraftoptimus.model  # noqa
from minecraftoptimus.model.steve1.config import PRIOR_INFO
from minecraftoptimus.model.steve1.data.text_alignment.vae import load_vae_model
from minecraftoptimus.model.steve1.utils.mineclip_agent_env_utils import load_mineclip_wconfig
from minecraftoptimus.utils import TASK2LABEL
from minecraftoptimus.utils.shape import resize_tensor_batch


IMPORTANT_VARIABLES = [
    "MINESTUDIO_SAVE_DIR",
    "MINESTUDIO_DATABASE_DIR",
]

for var in IMPORTANT_VARIABLES:
    val = os.environ.get(var, "not found")
    print(f"[Env Variable]  {var}: {val}")


def merge_dict_tensors(dict_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    将包含tensor的字典数组合并成一个字典，每个tensor在第一个维度上叠加
    对于input_ids和attention_mask会padding到最大长度

    参数:
        dict_list: 包含字典的列表，每个字典的值都是torch.Tensor

    返回:
        合并后的字典，每个键对应的值是所有输入字典中对应tensor在第一个维度上叠加的结果
    """
    if not dict_list:
        return {}

    # 确保所有字典有相同的键
    keys = dict_list[0].keys()
    if any(d.keys() != keys for d in dict_list):
        raise ValueError("所有字典必须包含相同的键")

    result = {}
    for k in keys:
        if k in ["labels", "input_ids", "attention_mask", "image_grid_thw"]:
            # 获取最大长度
            max_len = max(d[k].shape[-1] for d in dict_list)
            # 对每个tensor进行padding
            padded_tensors = []
            for d in dict_list:
                tensor = d[k]
                pad_size = max_len - tensor.shape[-1]
                if pad_size > 0:
                    if k == "input_ids":
                        # 使用pad_token_id填充input_ids
                        pad_token_id = 151643
                        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=pad_token_id)
                    elif k == "labels":
                        # 使用-100填充labels
                        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=-100)
                    else:  # attention_mask
                        # 使用0填充attention_mask
                        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), value=0)
                    padded_tensors.append(padded_tensor)
                else:
                    padded_tensors.append(tensor)
            # 移除多余的维度并stack
            result[k] = torch.stack([t.squeeze(0) for t in padded_tensors], dim=0)
        elif k == "prompt":
            result[k] = [d["prompt"] for d in dict_list]
        else:
            result[k] = torch.stack([d[k] for d in dict_list], dim=0)

    return result


def tree_detach(tree):
    if isinstance(tree, dict):
        return {k: tree_detach(v) for k, v in tree.items()}
    elif isinstance(tree, list):
        return [tree_detach(v) for v in tree]
    elif isinstance(tree, torch.Tensor):
        return tree.detach()
    else:
        return tree


class MineLightning(L.LightningModule):
    def __init__(
        self,
        mine_policy: MinePolicy,
        callbacks: List[ObjectiveCallback] = [],
        hyperparameters: dict = {},
        model_path: str | None = None,
        *,
        log_freq: int = 20,
        learning_rate: float = 1e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        window_len: int = 128,
        train_mine_policy: bool = False,
        mse_weight: float = 1.0,
        cosine_weight: float = 1.0,
    ):
        super().__init__()
        self.mine_policy = mine_policy
        self.callbacks = callbacks
        self.log_freq = log_freq
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.memory_dict = {
            "memory": None,
            "init_memory": None,
            "last_timestamp": None,
        }
        self.automatic_optimization = True
        self.window_len = window_len
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.mllm_embed_linear = torch.nn.Sequential(
            torch.nn.Linear(3584, 3584 * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(3584 * 2, 512),
        )

        self.mineclip = load_mineclip_wconfig(self.device)
        self.prior = load_vae_model(PRIOR_INFO, self.device)
        self.mineclip.requires_grad_(False)
        self.mineclip.eval()

        self.model.requires_grad_(False)
        self.model.eval()

        if not train_mine_policy:
            self.mine_policy.requires_grad_(False)
            self.mine_policy.eval()

        self.save_hyperparameters(hyperparameters)

    def _make_memory(self, batch):
        if self.memory_dict["init_memory"] is None:
            self.memory_dict["init_memory"] = self.mine_policy.initial_state(batch["image"].shape[0])
        if self.memory_dict["memory"] is None:
            self.memory_dict["memory"] = self.memory_dict["init_memory"]
        if self.memory_dict["last_timestamp"] is None:
            self.memory_dict["last_timestamp"] = torch.zeros(batch["image"].shape[0], dtype=torch.long).to(self.device)
        boe = batch["timestamp"][:, 0].ne(self.memory_dict["last_timestamp"] + 1)
        self.memory_dict["last_timestamp"] = batch["timestamp"][:, -1]
        # if boe's (begin-of-episode) item is True, then we keep the original memory, otherwise we reset the memory
        mem_cache = []
        for om, im in zip(self.memory_dict["memory"], self.memory_dict["init_memory"]):
            boe_f = boe[:, None, None].expand_as(om)
            mem_line = torch.where(boe_f, im, om)
            mem_cache.append(mem_line)
        self.memory_dict["memory"] = mem_cache
        return self.memory_dict["memory"]

    def _batch_step(self, batch, batch_idx, step_name):
        result = {"loss": 0}
        memory_in = self._make_memory(batch)
        batch["mllm"] = self.mllm_embed_linear(batch["mllm"]).squeeze(1)  # [bs, 512]
        with torch.amp.autocast(str(self.device)):
            prompt_embed = self.mineclip.encode_text(batch["task"]).to(batch["mllm"].device)
            text_prompt_embed = self.prior(batch["mllm"].float().to(self.device)).unsqueeze(1)  # [bs, 1, 512]
            # prompt_embed = (
            #     torch.from_numpy(get_prior_embed(batch["task"], self.mineclip, self.prior, batch["mllm"].device))
            #     .unsqueeze(1)
            #     .to(batch["mllm"].device)
            # )
        mse_loss = F.mse_loss(batch["mllm"], prompt_embed, reduction="mean")
        result["mse_loss"] = mse_loss
        cosine_loss = 1 - F.cosine_similarity(batch["mllm"], prompt_embed, dim=-1).mean()
        result["cosine_loss"] = cosine_loss
        batch["mllm"] = text_prompt_embed

        result["loss"] = result.get("loss", 0.0) + self.mse_weight * mse_loss + self.cosine_weight * cosine_loss

        # result["loss"] = result.get("loss", 0.0) + cosine_loss
        # kl_loss = torch.nn.functional.kl_div(
        #     torch.log_softmax(batch["mllm"], dim=-1), torch.softmax(prompt_embed, dim=-1), reduction="batchmean"
        # )
        # result["kl_loss"] = kl_loss
        # result["loss"] = result.get("loss", 0.0) + kl_loss
        memory_in = None
        latents, memory_out = self.mine_policy(batch, memory_in)
        self.memory_dict["memory"] = tree_detach(memory_out)
        for callback in self.callbacks:
            call_result = callback(batch, batch_idx, step_name, latents, self.mine_policy)
            for key, val in call_result.items():
                result[key] = result.get(key, 0) + val

        if batch_idx % self.log_freq == 0:
            for key, val in result.items():
                prog_bar = ("loss" in key) and (step_name == "train")
                self.log(f"{step_name}/{key}", val, sync_dist=True, prog_bar=prog_bar)

        return result

    def _get_action_embedding(self, batch):
        with torch.inference_mode():
            mllm_input = batch.get("mllm")
            mllm_input = merge_dict_tensors(mllm_input)
            mllm_input["tasks"] = torch.tensor([TASK2LABEL["action"]] * mllm_input["input_ids"].shape[0])
            output = self.model.get_action_embedding(**mllm_input)  # [bs,1,3584]
        return output

    def training_step(self, batch, batch_idx):
        batch["mllm"] = self._get_action_embedding(batch)

        batch["image"] = resize_tensor_batch(batch["image"].float())

        return self._batch_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        batch["mllm"] = self._get_action_embedding(batch)
        batch["image"] = resize_tensor_batch(batch["image"].float())

        return self._batch_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.mllm_embed_linear.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / self.warmup_steps, 1))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
