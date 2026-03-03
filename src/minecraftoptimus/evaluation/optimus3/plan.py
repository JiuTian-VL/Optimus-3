import argparse
import json
import re

from tqdm import tqdm


def extract_answer(content: str):
    if content is None:
        return None

    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def extract_steps(content: str):
    steps = []
    pattern = r"step\s+(\d+):\s*(?:[^\d]*?)(\d+)\s+([^\n]+)"
    matches = re.findall(pattern, content, re.IGNORECASE)
    for step_num, count, item in matches:
        steps.append({"step": int(step_num), "count": int(count), "item": item.strip()})
    return steps


# number >= groundtruth
# key in groundtruth


def is_same_steps(answer, pred):
    if len(pred) == 0:
        return False
    all_items = [item["item"] for item in answer]
    item_cnt = {item["item"]: item["count"] for item in answer}
    for item in pred:
        if item["item"] not in all_items:
            return False
        if item["count"] < item_cnt[item["item"]]:
            return False
    if answer[-1]["item"] != pred[-1]["item"]:
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default="plan_result.json")
    args = parser.parse_args()

    eval_file = args.eval_file

    with open(eval_file, "r") as fi:
        data = json.load(fi)

    pass_num = 0
    judge_result = []
    for sample in tqdm(data):
        question = sample["question"]
        answer = extract_steps(extract_answer(sample["answer"]))
        try:
            result = extract_steps(extract_answer(sample["output"]))
        except Exception as e:
            print(f"Error in sample {sample['question']}: {e}")
            continue

        sample["judge"] = is_same_steps(answer, result)
        if sample["judge"]:
            pass_num += 1
        judge_result.append(sample)
    print(f"pass ratio: {pass_num / len(data)}")
