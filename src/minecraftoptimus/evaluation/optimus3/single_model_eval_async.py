import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from litellm import acompletion
from rich import print
from tqdm.asyncio import tqdm_asyncio

SYSTEM_PROMPT = "You are a rigorous evaluator. Your task is to score and analyze one answer based on a provided correct answer."
USER_PROMPT = """Score the answer from 0 to 10 based on the following dimensions:
- Accuracy: Does the answer correctly reflect the information in the correct answer?
- Completeness: Does the answer include all the key points from the correct answer?
- Clarity: Is the answer clear, well-structured, and easy to understand?
- Relevance: Is all content in the answer relevant to the correct answer, without unnecessary additions?

You will provide both a numerical score (0-10) and a detailed reason.
Output Format:
Output1 Score: X/10
Output1 Reason: xxx

Answer: {answer}
Output1: [{output}]
"""


def extract_answer(content: str | None) -> str:
    if content is None:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(content).strip()


def extract_score_reason(content: str) -> tuple[int, str]:
    match = re.search(r"Output1 Score:\s*(\d+)/10\s*[\r\n]+Output1 Reason:\s*(.*)", content, re.DOTALL)
    if not match:
        return 0, "Failed to parse score/reason from judge response."

    try:
        score = int(match.group(1))
    except ValueError:
        score = 0

    score = max(0, min(10, score))
    reason = match.group(2).strip()
    return score, reason


def _from_messages(messages: list[dict[str, Any]], role: str) -> str:
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == role:
            return str(msg.get("content", "")).strip()
    return ""


def normalize_sample(item: dict[str, Any]) -> dict[str, Any]:
    messages = item.get("messages", []) if isinstance(item.get("messages"), list) else []

    question = str(item.get("question", "")).strip() or _from_messages(messages, "user").replace("<image>", "").strip()
    answer = extract_answer(str(item.get("answer", "")).strip() or _from_messages(messages, "assistant"))

    output = item.get("output")
    if output is None:
        output = item.get("response")
    output = extract_answer(str(output or ""))

    image_path = item.get("image")
    if image_path is None:
        images = item.get("images")
        if isinstance(images, list) and images:
            image_path = images[0]

    return {
        "question": question,
        "answer": answer,
        "output": output,
        "image": image_path,
    }


async def evaluate_one(
    idx: int,
    sample: dict[str, Any],
    judge_llm: str,
    api_key: str,
    base_url: str,
    semaphore: asyncio.Semaphore,
    retry_sleep: float,
) -> dict[str, Any]:
    async with semaphore:
        prompt = USER_PROMPT.format(answer=sample["answer"], output=sample["output"])

        while True:
            try:
                judge_response = await acompletion(
                    model=judge_llm,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    api_key=api_key,
                    base_url=base_url,
                )
                break
            except Exception as e:
                print(f"[red]Judge API error at index {idx}: {e}. Retrying in {retry_sleep}s.[/red]")
                await asyncio.sleep(retry_sleep)

        content = ""
        if (
            judge_response
            and judge_response.choices
            and judge_response.choices[0].message
            and judge_response.choices[0].message.content
        ):
            content = judge_response["choices"][0]["message"]["content"]

        score, reason = extract_score_reason(content)

        result = {
            "index": idx,
            "question": sample["question"],
            "answer": sample["answer"],
            "image": sample["image"],
            "output": sample["output"],
            "score": score,
            "reason": reason,
        }

        try:
            result["judge_response"] = judge_response.model_dump()
        except AttributeError:
            result["judge_response"] = judge_response

        return result


def build_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    score_counter = Counter(item.get("score", 0) for item in results)
    score_hist = {str(s): score_counter.get(s, 0) for s in range(10, -1, -1)}

    total = len(results)
    score_sum = sum(item.get("score", 0) for item in results)

    return {
        "count": total,
        "avg_score": (score_sum / total) if total else 0.0,
        "score_hist": score_hist,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True, help="Single model result JSON file.")
    parser.add_argument("--judge_llm", type=str, default="gpt-5.2-chat-latest")
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-MBHdeyudZupuva1U90Cb85B7D65849A48c854a2f2f7a00A9",
        help="Aligned with original vqa_async.py default.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://oai.ppai.pro/v1",
        help="Aligned with original vqa_async.py default.",
    )
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--retry_sleep", type=float, default=5.0)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{args.eval_file} is not a list JSON.")

    normalized: list[tuple[dict[str, Any], int]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        normalized.append((normalize_sample(item), idx))

    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        evaluate_one(
            idx=orig_idx,
            sample=sample,
            judge_llm=args.judge_llm,
            api_key=args.api_key,
            base_url=args.base_url,
            semaphore=semaphore,
            retry_sleep=args.retry_sleep,
        )
        for sample, orig_idx in normalized
    ]

    print(f"Start evaluating {len(tasks)} samples from {args.eval_file}...")
    results = await tqdm_asyncio.gather(*tasks, desc="Scoring")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    stats = build_stats(results)
    stats_file = str(output_path).replace(".json", "_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

    meta_file = str(output_path).replace(".json", "_meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_file": args.eval_file,
                "judge_llm": args.judge_llm,
                "base_url": args.base_url,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(f"\n[green]Detailed results saved:[/green] {output_path}")
    print(f"[green]Stats saved:[/green] {stats_file}")

    print("\n[bold]Score Distribution[/bold]")
    for score in range(10, -1, -1):
        print(f"{score}: {stats['score_hist'][str(score)]}")
    print(f"total: {stats['count']}")
    print(f"avg_score: {stats['avg_score']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
