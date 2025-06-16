TASK2LABEL = {"plan": 0, "vqa": 1, "perception": 1, "reflection": 2, "action": 3, "grounding": 4}
Task_Prompt={
    "planning":["from scratch?","Given that"],
    "action":"Now generate low-level action",
    "captioning":"You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. The user's instruction is: <image> ",
    "embodied_qa":"You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. The user's instruction is: <image> ",
    "grounding":"You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. The user's instruction is: <image> ",
    "reflection": "<image> Given the current image, you are required to determine whether the task can continue."
}