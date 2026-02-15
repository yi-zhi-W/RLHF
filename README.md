# Multi-Modal RLHF for Qwen2-Audio (PPO & GRPO)

This repository is a customized fork of **LLaMA-Factory** and **TRL**, extended to support **Audio-Text Multi-Modal Reinforcement Learning** specifically for the **Qwen2-Audio** model.

While the original libraries primarily support text-only modalities for RLHF, this project modifies the core training loops to handle audio inputs. Additionally, it introduces a **GRPO** implementation compatible with LLaMA-Factory's dependency versions and expands reward mechanism support.

## Key Features

### 1. Multi-Modal PPO Support (Qwen2-Audio)
*   **Audio-Aware Training:** Modified the standard `PPOTrainer` in both LLaMA-Factory and TRL to correctly propagate audio features through the actor, critic, and reference models.
*   **Extended Reward Mechanisms:**
    *   **Reward Model:** Standard reward modeling.
    *   **API-based Reward:** Fetch rewards from external endpoints.
    *   **Custom Reward Functions:** [New] Support for defining rule-based rewards directly within the trainer.

### 2. GRPO Implementation (Backport & Multi-Modal)
*   **GRPO for TRL v0.9.6:** Since LLaMA-Factory relies on an older version of TRL (pre-v1.0) which lacks Group Relative Policy Optimization (GRPO), this repository implements a custom `GRPOTrainer` compatible with the existing environment.
*   **Multi-Modal Support:** Like the PPO implementation, the GRPO trainer handles Qwen2-Audio's specific input requirements.
*   **Flexible Rewards:** Supports Reward Models, APIs, and Custom functions.


## Supported Algorithms

### 1. PPO (Proximal Policy Optimization)
We have patched `trl.trainer.ppo_trainer.py` and `src/llamafactory/train/ppo/trainer.py`.

**Key Changes:**
*   The `step()` function and forward passes now accept and process audio features. 
*   Added support for defining rule-based rewards.


### 2. GRPO (Group Relative Policy Optimization)
Located in `src/llamafactory/train/mygrpo`, this is a standalone implementation designed to work where official TRL support is missing or incompatible.

**Key Changes:**
*   Implements the group-based loss function without a value function critic.
*   Optimized for multi-modal generation and scoring.
*   Supports Custom Reward Functions.

**Disclaimer:** This is a research modification. Please refer to the original repositories for standard usage documentation.

## Citation

If you use this work or the modified libraries, please cite the following:

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={[http://arxiv.org/abs/2403.13372](http://arxiv.org/abs/2403.13372)}
}

@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}