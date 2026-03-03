import json
import math
import os
import pathlib
from typing import Any, Optional

import torch
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_model
from qwen_vl_utils import process_vision_info
from rich import print
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from transformers import AutoProcessor

import minecraftoptimus.model  # noqa
from minecraftoptimus.evaluation.utils import sharegpt2common_format, sharegpt2qwen2_5format
from minecraftoptimus.utils import TASK2LABEL


class Evaluator:
    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.processor = AutoProcessor.from_pretrained(self.model_args.model_name_or_path)
        self.processor.tokenizer.padding_side = "left"
        self.model = load_model(self.processor.tokenizer, self.model_args, finetuning_args)

        self.meta_data_file = self.eval_args.save_dir.replace(".json", "_meta.json")

        os.makedirs(pathlib.Path(self.eval_args.save_dir).parent, exist_ok=True)

        with open(self.meta_data_file, "w") as fo:
            args = {
                "model_path": self.model_args.model_name_or_path,
                "eval": {"task": self.eval_args.task, "task_type": self.eval_args.task_type},
            }
            json.dump(args, fo, indent=4)

    @torch.inference_mode()
    def batch_inference(self, batch_input: dict[str, "torch.Tensor"]) -> list[str]:
        if "baseline" not in self.eval_args.save_dir:
            batch_input["tasks"] = torch.tensor(
                [TASK2LABEL[self.eval_args.task_type]] * batch_input["input_ids"].shape[0]
            )
        batch_input = batch_input.to("cuda")

        # Batch Inference
        generated_ids = self.model.generate(**batch_input, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch_input.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts

    def eval(self) -> None:
        eval_file = self.eval_args.task
        print(f"Evaluating on {eval_file}...")
        with open(eval_file, encoding="utf-8") as f:
            eval_dataset = json.load(f)
        print(f"Evaluation samples: {len(eval_dataset)}")

        _eval_dataset = eval_dataset.copy()
        eval_dataset = [sharegpt2qwen2_5format(sample) for sample in eval_dataset]

        result = []
        batch_size = self.eval_args.batch_size

        # 添加进度条
        progress_columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total} batches)"),
            TextColumn("Elapsed:"),
            TimeElapsedColumn(),
            TextColumn("Remaining:"),
            TimeRemainingColumn(),
        ]

        total_step = math.ceil(len(eval_dataset) / batch_size)
        with Progress(*progress_columns) as progress:
            task = progress.add_task("[cyan]Evaluating...", total=total_step)

            for idx in range(0, len(eval_dataset), batch_size):
                samples = eval_dataset[idx : idx + batch_size]
                texts = [
                    self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in samples
                ]
                image_inputs, video_inputs = process_vision_info(samples)

                batch_input = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                outputs = self.batch_inference(batch_input)
                print(outputs)
                for output_idx, output in enumerate(outputs):
                    _temp = sharegpt2common_format(_eval_dataset[output_idx + idx])
                    _temp["output"] = output
                    result.append(_temp)
                progress.update(task, advance=1)

        with open(self.eval_args.save_dir, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"[green]Evaluation Done! Save to {self.eval_args.save_dir}")


def run_eval() -> None:
    Evaluator().eval()


if __name__ == "__main__":
    run_eval()
