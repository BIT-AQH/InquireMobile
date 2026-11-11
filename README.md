# InquireMobile

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

> **[ACL 2026]** This is the official implementation repository for **InquireMobile**.

**InquireMobile** is an interactive mobile GUI agent that learns to *ask the user*
(`call_user`) instead of blindly acting when it faces uncertain, sensitive, or
ambiguous situations (e.g., login, payment, privacy-related decisions). The agent
is trained with **Reinforcement Learning with Verifiable Rewards (RLVR / GRPO)** on
top of a vision-language model, so that it reasons about *whether* to act or to
request human intervention.

## Overview

- **Interactive action space** — beyond `click` / `swipe` / `type` / `wait` /
  `terminate`, the agent exposes a `call_user` action to request user
  intervention with free-form content.
- **Two-stage training** — supervised fine-tuning (SFT) followed by
  GRPO-based RLVR, built on the [R1-V](https://github.com/Deep-Agent/R1-V)
  framework for verifiable-reward reinforcement learning in vision-language
  models.
- **Verifiable rewards for interaction** — the reward is computed from the
  agent's own "ask vs. act" decision against annotated interaction labels.

## Repository structure

```
.
├── InquireMobile/            # main code for the InquireMobile agent
│   ├── annotation/           # data annotation scripts (GPT-4o / Gemini)
│   ├── data/                 # training & evaluation datasets (.jsonl)
│   ├── result/               # evaluation results
│   ├── prompt/train.txt      # the agent system prompt / action schema
│   ├── eval.py / eval*.bash  # evaluation scripts
│   └── ...
├── R1-V/                     # R1-V fork: RLVR (GRPO) training code
│   └── src/r1-v/src/open_r1/ # grpo_interactive.py, grpo_operator_ai_v2.py, ...
├── eval/                     # environment / emulator helpers
└── homepage/                 # project homepage
```

## Getting started

### Requirements

- Python 3.10+
- PyTorch with CUDA
- `transformers`, `accelerate`, `deepspeed`, `vllm` (optional, for faster GRPO)
- `oss2` (only for the optional OSS upload in the evaluation helpers)

### Model

InquireMobile is trained from a Qwen2.5-VL base. Training proceeds in two
stages:

1. **SFT** on interaction-annotated trajectories.
2. **GRPO / RLVR** using the scripts under `R1-V/`.

> Model checkpoints are released separately. *(TODO: add checkpoint links.)*

## Training

Training is launched from `R1-V/src/r1-v/` using `torchrun`, e.g.:

```bash
cd R1-V/src/r1-v
bash run_interactive.sh   # GRPO with the interactive (call_user) reward
```

See the `run*.sh` scripts and the `grpo_*` scripts under `src/open_r1/` for the
available variants (GUI-agent, interactive, operator-AI).

## Evaluation

```bash
cd InquireMobile
bash eval.bash   # adjust --model_path / --eval_dataset_path / --eval_result_path
```

Evaluation produces per-sample JSONL results under `result/`.

## Data

Training and evaluation data are stored as JSONL files under `InquireMobile/data/`
and `InquireMobile/annotation/`. See [LICENSE-DATA](LICENSE-DATA) for the data
license (CC BY 4.0).

## License

- **Code**: [MIT License](LICENSE)
- **Data**: [CC BY 4.0](LICENSE-DATA)

## Citation

```bibtex
@inproceedings{inquirermobile2026,
  title     = {InquireMobile},
  author    = {},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
  year      = {2026}
}
```

> *(TODO: fill in the full author list, title, and page numbers.)*
