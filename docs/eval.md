# Evaluation

This document describes how to evaluate an **InquireMobile** checkpoint. The
pipeline has two stages:

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

---

## 1. Environment rollout

`eval/gui_agent_watching_draw_bbox.py` watches an OSS bucket for new actions
and executes them on an ADB device/emulator, uploading screenshots back to OSS.

```bash
cd eval

# OSS credentials (required for upload)
export OSS_ACCESS_KEY_ID="..."
export OSS_ACCESS_KEY_SECRET="..."
export OSS_ENDPOINT="oss-cn-beijing.aliyuncs.com"
export OSS_BUCKET_NAME="alimama-creative-public"

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

---

## 2. Model rollout + judging

`InquireMobile/eval.py` loads a Qwen2.5-VL (or UI-TARS) checkpoint, produces a
trajectory from the collected screenshots, and scores it with GPT-4o.

```bash
cd InquireMobile

# Judge credentials (or pass --api_key / --api_url)
export OPENAI_API_KEY="..."            # or DASHSCOPE_API_KEY
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

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

---

## Input data format

The evaluation dataset is a JSONL file with one instruction per line:

```json
{"instruction": "Open WeChat and send a message ...", "category": "privacy", "intension": "..."}
```

- `instruction` — the user task.
- `category` — task category (e.g. `privacy`, `intension`).
- `intension` — the user's intent used after a `call_user` interaction.
