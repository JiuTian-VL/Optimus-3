import re
from typing import Any

import numpy as np
import torch
from huggingface_hub import ModelHubMixin
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from minecraftoptimus.model.agent.base import BaseAgent
from minecraftoptimus.model.optimus3.modeling_optimus3 import Optimus3ForConditionalGeneration
from minecraftoptimus.model.optimus3.modeling_task_router import TaskRouterModel
from minecraftoptimus.model.steve1.agent import Optimus3ActionAgent
from minecraftoptimus.utils import TASK2LABEL, Task_Prompt


def check_inventory(inventory, item, count):
    def _check(inventory_item, item):
        if item == "logs":
            return "log" in inventory_item
        elif item == "planks":
            return "planks" in inventory_item
        else:
            return item in inventory_item

    for k, v in inventory.items():
        if _check(v["type"], item) and v["quantity"] >= count:
            return True, v["quantity"]
    return False, 0


def extract_answer(content: str):
    if content is None:
        return None

    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def extract_steps(content: str):
    pattern = r"step \d+: (.*?)(?=\nstep |$)"  
    steps_list = re.findall(pattern, content, re.DOTALL)  
    return steps_list


def extract_goals(content: str):
    steps = []
    pattern = r"step\s+(\d+):\s*(?:[^\d]*?)(\d+)\s+([^\n]+)"
    matches = re.findall(pattern, content, re.IGNORECASE)
    for step_num, count, item in matches:
        if "log" in item.lower():
            item = "logs"
        steps.append({"step": int(step_num), "count": int(count), "item": item.strip()})
    return steps


class Optimus3Agent(BaseAgent, ModelHubMixin):
    def __init__(
        self,
        policy_ckpt_path: str,
        mllm_model_path: str,
        task_router_ckpt_path: str = "path_to_task_router/optimus3-task-router",
        device="cuda",
    ):
        super().__init__()
        self.mine_policy = Optimus3ActionAgent.from_pretrained(policy_ckpt_path)
        self.device = device

        self.cache_mllm_embed = None
        self.cache_task = None

        self.model = Optimus3ForConditionalGeneration.from_pretrained(
            mllm_model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
        )
        self.processor = AutoProcessor.from_pretrained(mllm_model_path)
        self.task_router = TaskRouterModel.from_pretrained(task_router_ckpt_path)

        self.model.requires_grad_(False)
        self.task_router.requires_grad_(False)

        self.model.eval()
        self.task_router.eval()

        self.model.to(self.device)
        self.task_router.to(self.device)

        self.task: str | None = None

        self.system_prompt = "You are an expert in Minecraft, capable of performing task planning, visual question answering, reflection, grounding and executing low-level actions."

    def reset(self, task: str):
        self.task = task
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": [{"type": "text", "text": task}]},
        ]
        output = self._generate(messages, task_type="action", skip_special_tokens=False)
        task_label = output[0][:-10]
        messages.append({"role": "assistant", "content": task_label})

        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]
        image_inputs, video_inputs = process_vision_info(messages)

        batch_input = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch_input["tasks"] = torch.tensor([TASK2LABEL["action"]] * batch_input["input_ids"].shape[0])
        batch_input["labels"] = batch_input["input_ids"].clone()
        condition = (batch_input["labels"] < 151665) | (batch_input["labels"] > 151674)
        batch_input["labels"][condition] = -100
        batch_input = batch_input.to(self.device)
        with torch.inference_mode():
            output = self.model.get_action_embedding(**batch_input)
        self.cache_mllm_embed = output.float()
        return output.float()

    @torch.inference_mode()
    def get_action(
        self,
        input: dict[str, Any],
        task: str,
        # state_in: list[torch.Tensor] | None,
        deterministic: bool = False,
        input_shape: str = "BT*",
        **kwargs,
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """
        input: {
            "image": [384, 384, 3], np.array
        }
        """
        if self.task is None:
            raise ValueError("Task not set, please call reset() before get_action()")

        # input["image"] = resize_numpy_array_pillow(input["image"], (128, 128))
        # input["mllm"] = self.cache_mllm_embed  # [1,1,512]
        action, _ = self.mine_policy.optimus3_action(self.cache_mllm_embed, input["image"], task=task)

        return action, None

    @torch.inference_mode()
    def _generate(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int = 2048,
        task_type: str = "plan",
        skip_special_tokens=True,
    ) -> list[str]:
        assert task_type in TASK2LABEL, (
            f"task_type {task_type} not supported, only {list(TASK2LABEL.keys())} are supported"
        )
        texts = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = process_vision_info(messages)

        batch_input = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch_input["tasks"] = torch.tensor([TASK2LABEL[task_type]] * batch_input["input_ids"].shape[0])
        batch_input = batch_input.to(self.device)

        # Batch Inference
        generated_ids = self.model.generate(**batch_input, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_input.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False
        )
        return output_texts

    @torch.inference_mode()
    def plan(self, task: str) -> tuple[str, list[str], list[dict]]:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": [{"type": "text", "text": "How to " + task + " from scratch?"}]},
        ]
        original_text = self._generate(messages, task_type="plan", max_new_tokens=512)[0]
        output_texts = extract_answer(original_text)
        goals = extract_goals(output_texts)
        return original_text, extract_steps(output_texts), goals

    @torch.inference_mode()
    def reflection(self, task: str, image: np.ndarray | str | None) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": [{"type": "text", "text": Task_Prompt["reflection"] + task}]},
        ]
        if image is not None:
            
            messages[1]["content"].append({"type": "image", "image": image})
        output_texts = self._generate(messages, task_type="reflection")[0]
        # output_texts = [extract_answer(text) for text in output_texts]
        return output_texts

    @torch.inference_mode()
    def answer(self, question: str, image: np.ndarray | str | None) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": [{"type": "text", "text": Task_Prompt["embodied_qa"] + question}]},
        ]
        if image is not None:
            
            messages[1]["content"].append({"type": "image", "image": image})
        output_texts = self._generate(messages, task_type="vqa")[0]
        # output_texts = [extract_answer(text) for text in output_texts]
        return output_texts

    @torch.inference_mode()
    def grounding(self, query: str, image: np.ndarray | str | None) -> str:
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": [{"type": "text", "text": Task_Prompt["grounding"] + query}]},
        ]
        if image is not None:
            
            messages[1]["content"].append({"type": "image", "image": image})
        output_texts = self._generate(messages, task_type="grounding")[0]
        # output_texts = [extract_answer(text) for text in output_texts]
        return output_texts

    @torch.inference_mode()
    def router(self, query: str) -> int:
        return self.task_router.router(query)
