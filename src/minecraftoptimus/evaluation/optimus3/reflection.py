#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def extract_answer(pred: str) -> str:
    if not isinstance(pred, str):
        return ""
    # Prefer explicit <answer>...</answer>
    ans = re.search(r"<answer>\s*(.*?)\s*</answer>", pred, flags=re.S | re.I)
    if ans:
        return ans.group(1).strip()
    # Otherwise drop any <think>...</think> and use the remaining text
    no_think = re.sub(r"<think>.*?</think>", "", pred, flags=re.S | re.I)
    return no_think.strip()


def normalize(text: str) -> str:
    # Collapse whitespace and compare case-insensitively
    return re.sub(r"\s+", " ", text).strip().lower()


def compute_accuracy(path: Path) -> None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    correct = 0
    for idx, item in enumerate(data):
        gt = normalize(item.get("answer", ""))
        pred = normalize(extract_answer(item.get("output", "")))
        if gt == pred:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    default_path = "/data7/Users/xyq/developer/MinecraftOptimus/output/eval/optimus3/9B/coldstart/20251215_9B_coldstart_plan_vqa_caption_thinkinggrounding_reflection_bs8_ga16_lr3e-5_epoch2/checkpoint-266/reflection.json"
    parser = argparse.ArgumentParser(description="Compute reflection accuracy from a JSON file.")
    parser.add_argument(
        "--eval_file",
        default=default_path,
        help="Path to reflection result JSON (default points to the latest checkpoint result).",
    )
    args = parser.parse_args()
    compute_accuracy(Path(args.eval_file))
