<div align="center">
<h2 align="center">
   <img src="./assets/optimus3.png" style="vertical-align: middle; height: 1em; padding: 0 0.2em;"> <b>Optimus-3: Towards Generalist Multimodal Minecraft Agents
     <br />  with Scalable Task Experts </b>
</h2>
<div>
<a target="_blank" href="https://scholar.google.com/citations?user=TDBF2UoAAAAJ&hl=en&oi=ao">Zaijing&#160;Li</a><sup>1 2</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=KO77A2oAAAAJ&hl=en">Yuquan&#160;Xie</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=9Vc--XsAAAAJ&hl=en&oi=ao">Rui&#160;Shao</a><sup>1&#9993</sup>,
<a target="_blank" href="https://scholar.google.com/citations?user=Mpg0w3cAAAAJ&hl=en&oi=ao">Gongwei&#160;Chen</a><sup>1</sup>,
<br>
<a target="_blank" href="https://ieeexplore.ieee.org/author/37087008154">Weili&#160;Guan</a><sup>1</sup>,
<a target="_blank" href="https://scholar.google.com/citations?hl=en&user=Awsue7sAAAAJ">Dongmei&#160;Jiang</a><sup>2</sup>,
 <a target="_blank" href="https://scholar.google.com/citations?hl=en&user=yywVMhUAAAAJ">Liqiang&#160;Nie</a><sup>1&#9993</sup>
</div>
<sup>1</sup>Harbin Institute of Technology,Shenzhen&#160&#160&#160</span>
<sup>2</sup>Peng Cheng Laboratory, Shenzhen</span>
<br />
<sup>&#9993&#160;</sup>Corresponding author&#160;&#160;</span>
<br/>
<div align="center">
    <a href="https://arxiv.org/abs/2506.10357" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-deepgreen" alt="Paper arXiv"></a>
    <a href="https://cybertronagent.github.io/Optimus-3.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Optimus--3-9cf" alt="Project Page"></a>
</div>
</div>

## :new: Updates
- [06/2025] :fire: [Project page](https://cybertronagent.github.io/Optimus-3.github.io/) released.
- [06/2025] :fire: [Arxiv paper](https://arxiv.org/abs/2506.10357) released.

## :rocket: Optimus-3 

Demonstration of Optimus-3’s capabilities as a generalist agent in Minecraft. It can perform long-horizon task planning, captioning, embodied QA, grounding, low-level action generation, and reflection in an interactive manner. All of these capabilities are seamlessly integrated into a unified end-to-end architecture, enabling robust and coherent performance across diverse task scenarios. 
<img src="./assets/fig1.png" >

## :wrench: Data Generation Pipeline

Given a task pool, we utilize a knowledge graph to generate task plans, forming the planning dataset. These plans are then used as instructions for STEVE-1, which interacts with the environment to produce the action dataset. During this process, we randomly sample images and employ expert models with environmental feedback to generate the captioning, embodied QA, and grounding datasets.
<img src="./assets/fig3.png" >

## :balloon: Framework

A: The architecture of Optimus-3, which includes a task router that selects a specific task expert for each query, a ViT for visual encoding, and a MoE LLM for generating responses and low-level actions. Given a long-horizon task, it can generate a feasible plan and then execute the sub-goals sequentially. B: The proposed Multimodal Reasoning-Augmented Reinforcement Learning effectively enhances the agent's performance. C: Performance comparison of Optimus-3 against current task-specific SOTA agents, GPT-4o, and the original backbone Qwen2.5-VL. 
<img src="./assets/fig2.png" >

## :smile_cat: Evaluation results
Main Result of Optimus-3 on Long-Horizon tasks, Planning, Captioning, Embodied QA, Grounding, and Reflection.

Table 1: Main Result of Optimus-3 on Long-Horizon tasks.
<img src="./assets/table1.png" >

Table 2: Main Result of Optimus-3 on Planning, Captioning, Embodied QA, Grounding, and Reflection.
<img src="./assets/table2.png" >

## :hugs: Citation

If you find this work useful for your research, please kindly cite our paper:

```
@article{li2025optimus,
  title={Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts},
  author={Li, Zaijing and Xie, Yuquan and Shao, Rui and Chen, Gongwei and Guan, Weili and Jiang, Dongmei and Nie, Liqiang},
  journal={arXiv preprint arXiv:2506.10357},
  year={2025}
}
```








