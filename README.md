# InquireMobile

> **InquireMobile: Teaching VLM-based Mobile Agent to Request Human Assistance via Reinforcement Fine-Tuning**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![ACL 2026](https://img.shields.io/badge/ACL-2026-b31b1b.svg)](https://aclanthology.org/2026.acl-long.1487/)

This is the official implementation of **InquireMobile** (ACL 2026).

**Project page:** <https://bit-aqh.github.io/InquireMobile/homepage/> · **Paper:** <https://aclanthology.org/2026.acl-long.1487/>

Recent advances in Vision-Language Models (VLMs) have enabled mobile agents to
perceive and interact with real-world mobile environments from human
instructions. The fully autonomous paradigm, however, poses safety risks when the
model's understanding or reasoning is insufficient. **InquireMobile** is an
interactive system that *actively seeks human confirmation at critical decision
points* instead of blindly acting — it learns to call the user (`call_user`) when
facing uncertain, sensitive, or ambiguous situations (e.g., login, payment,
privacy-related decisions).

## Overview

- **InquireBench** — a comprehensive benchmark for evaluating a mobile agent's
  safe interaction and proactive inquiry, spanning 5 categories and 22
  sub-categories, where most existing VLM-based agents score near zero.
- **Two-stage training** — supervised fine-tuning (SFT) followed by
  reinforcement fine-tuning (GRPO / RLVR), built on the
  [R1-V](https://github.com/Deep-Agent/R1-V) framework for verifiable-reward
  reinforcement learning in vision-language models.
- **Interactive pre-action reasoning** — the agent reasons *whether* to act or to
  request user assistance before each action.
- **Results** — **+46.8%** inquiry success rate and the best overall success rate
  among existing baselines on InquireBench.

## Repository structure

```
.
├── InquireMobile/            # main code for the InquireMobile agent
│   ├── annotation/           # data annotation scripts (GPT-4o / Gemini)
│   ├── data/                 # -> hosted on Hugging Face (see below)
│   ├── result/               # evaluation results
│   ├── prompt/train.txt      # the agent system prompt / action schema
│   ├── eval.py / eval.bash   # model rollout + GPT-4o judge
│   └── ...
├── R1-V/                     # R1-V fork: RLVR (GRPO) training code
│   └── src/r1-v/src/open_r1/ # grpo_interactive.py, sft.py, ...
├── eval/                     # device/emulator + OSS collector
│   ├── gui_agent_watching_draw_bbox.py
│   └── kill_python.sh
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
available variants (interactive, GUI-agent).

## Evaluation

The evaluation pipeline has two stages:

1. **Environment rollout** — drive an Android device/emulator, feed the model
   actions back into the environment, and collect screenshots + actions.
2. **Model rollout + judging** — replay the agent against the collected
   screenshots and score each trajectory with a GPT-4o-compatible judge.

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│  eval/gui_agent_watching_   │      │  InquireMobile/eval.py       │
│  draw_bbox.py               │ ───► │  rollout + GPT-4o judge      │
│  (device + OSS collector)   │      │  -> result/*.jsonl           │
└─────────────────────────────┘      └──────────────────────────────┘
```

### 1. Environment rollout

`eval/gui_agent_watching_draw_bbox.py` watches an OSS bucket for new actions
and executes them on an ADB device/emulator, uploading screenshots back to OSS.

```bash
cd eval

# OSS credentials (required for upload)
export OSS_ACCESS_KEY_ID="..."
export OSS_ACCESS_KEY_SECRET="..."
export OSS_ENDPOINT=""
export OSS_BUCKET_NAME=""

python gui_agent_watching_draw_bbox.py \
    --task_name eval_inquire_stage1_huawei_en \
    --devices huawei \
    --training_step 42 \
    --sample_num 1 \
    --max_steps 15 \
    --image_base /path/to/images/huawei \
    --bbox_base /path/to/images/huawei_bbox
```

Key arguments:

| Argument | Description |
|---|---|
| `--task_name` | Task/experiment name; determines the OSS key prefix. |
| `--devices` | Logical device names (`huawei`, `device0`, …) mapped to ADB serials in the script. |
| `--training_step` | Starting instruction index (incremented before the first run). |
| `--sample_num` | Number of samples per instruction. |
| `--max_steps` | Maximum rollout steps. |
| `--image_base` / `--bbox_base` | Local directories for captured screenshots / bbox images. |
| `--loop` | Keep incrementing `training_step` and rerun (for interactive data collection). |
| `--serial` | Override the ADB serial for the selected devices. |

### 2. Model rollout + judging

`InquireMobile/eval.py` loads a Qwen2.5-VL (or UI-TARS) checkpoint, produces a
trajectory from the collected screenshots, and scores it with GPT-4o.

```bash
cd InquireMobile

# Judge credentials (or pass --api_key / --api_url)
export OPENAI_API_KEY="..."            # or DASHSCOPE_API_KEY
export OPENAI_BASE_URL=""   # fill in your judge API base URL

python eval.py \
    --model_path /path/to/Qwen2.5-VL-3B-Inquire-Stage1 \
    --eval_dataset_path data/tasks_en_huawei.jsonl \
    --eval_result_path result/eval_stage1.jsonl \
    --task_name eval_inquire_stage1_huawei_en \
    --device huawei \
    --arch qwen2_5_vl \
    --oss_base /data/oss_bucket_0/USERNAME/gui_agent \
    --max_steps 15
```

Or use the convenience launcher [`eval.bash`](InquireMobile/eval.bash).

Key arguments:

| Argument | Description |
|---|---|
| `--model_path` | Model checkpoint (a PEFT adapter is merged automatically if present). |
| `--eval_dataset_path` | JSONL of `{"instruction", "category", "intension"}` items. |
| `--eval_result_path` | JSONL of per-sample judge results (`reason`, `reward`, `status`, `inquiry`). |
| `--task_name` / `--device` | Locate the pre-generated screenshots under `--oss_base`. |
| `--arch` | `qwen2_5_vl` (default) or `ui_tars` (Qwen2-VL). |
| `--oss_base` | Base directory with `images/` and `instructions/`. |
| `--max_steps` | Maximum rollout steps. |
| `--api_key` / `--api_url` | Judge credentials (default from `OPENAI_API_KEY`/`DASHSCOPE_API_KEY` and `OPENAI_BASE_URL`). |

The result file is append-only: already-evaluated instructions are skipped on
restart, so the run can be resumed safely.

### Input data format

The evaluation dataset is a JSONL file with one instruction per line:

```json
{"instruction": "Open WeChat and send a message ...", "category": "privacy", "intension": "..."}
```

- `instruction` — the user task.
- `category` — task category (e.g. `privacy`, `intension`).
- `intension` — the user's intent used after a `call_user` interaction.

## Data

Training and evaluation data are stored as JSONL files under `InquireMobile/data/`
and `InquireMobile/annotation/`. See [LICENSE-DATA](LICENSE-DATA) for the data
license (CC BY 4.0).

## License

- **Code**: [MIT License](LICENSE)
- **Data**: [CC BY 4.0](LICENSE-DATA)

## Citation

```bibtex
@inproceedings{ai-etal-2026-inquiremobile,
    title = "{I}nquire{M}obile: Teaching {VLM}-based Mobile Agent to Request Human Assistance via Reinforcement Fine-Tuning",
    author = "Ai, Qihang  and
      Bu, Pi  and
      Cao, Yue  and
      Wang, Yingyao  and
      Gu, Jihao  and
      Xing, Jingxuan  and
      Zhu, Zekun  and
      Jiang, Wei  and
      Zheng, Zhicheng  and
      Song, Jun  and
      Jiang, Yuning",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1487/",
    doi = "10.18653/v1/2026.acl-long.1487",
    pages = "32241--32255",
    ISBN = "979-8-89176-390-6",
    abstract = "Recent advances in Vision-Language Models (VLMs) have enabled mobile agents to perceive and interact with real-world mobile environments based on human instructions. However, the current fully autonomous paradigm poses potential safety risks when model understanding or reasoning capabilities are insufficient. To address this challenge, we first introduce \textbf{InquireBench}, a comprehensive benchmark specifically designed to evaluate mobile agents' capabilities in safe interaction and proactive inquiry with users, encompassing 5 categories and 22 sub-categories, where most existing VLM-based agents demonstrate near-zero performance. In this paper, we aim to develop an interactive system that actively seeks human confirmation at critical decision points. To achieve this, we propose \textbf{InquireMobile}, a novel model inspired by reinforcement learning, featuring a two-stage training strategy and an interactive pre-action reasoning mechanism. Finally, our model achieves an 46.8{\%} improvement in inquiry success rate and the best overall success rate among existing baselines on InquireBench. The project page is available at \url{https://bit-aqh.github.io/InquireMobile/homepage/}."
}
```
