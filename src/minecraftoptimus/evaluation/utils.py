import re
from typing import Any


def sharegpt2qwen2_5format(sample: dict[str, Any], custom_system_prompt: str | None = None) -> list:
    r"""Convert ShareGPT format to Qwen2.5 format."""
    messages = sample["messages"]
    images = sample.get("images", None)
    qwen2_5_format = []
    if messages[0]["role"] == "system":
        if custom_system_prompt is None:
            qwen2_5_format.append({"role": "system", "content": messages[0]["content"]})
        else:
            qwen2_5_format.append({"role": "system", "content": custom_system_prompt})

        messages = messages[1:]
    for i, message in enumerate(messages):
        if message["role"] == "user":
            if images is not None:
                qwen2_5_format.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"].replace("<image>", ""),
                        },
                        {"type": "image", "image": images[0]},
                    ],
                })
            else:
                qwen2_5_format.append({
                    "role": "user",
                    "content": [{"type": "text", "text": message["content"]}],
                })

    return qwen2_5_format


def extract_answer(content: str):
    if content is None:
        return None

    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def sharegpt2common_format(sample: dict[str, Any]) -> dict:
    r"""Convert ShareGPT format to common format like this:
    {
        "system": "You are a helpful assistant.",
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.",
        "image": "path/to/image.jpg",
    }
    """
    messages = sample["messages"]
    images = sample.get("images", None)
    common_format = {}
    if messages[0]["role"] == "system":
        common_format["system"] = messages[0]["content"]
        messages = messages[1:]
    for i, message in enumerate(messages):
        if message["role"] == "user":
            common_format["question"] = message["content"].replace("<image>", "")
            if images is not None:
                common_format["image"] = images[0]
        else:
            common_format["answer"] = extract_answer(message["content"])
    return common_format
