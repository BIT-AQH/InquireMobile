#!/usr/bin/env bash
# Model-side evaluation: rollout + GPT-4o judge.
# Start `evaluation/device_runner.py` on the machine connected to the phone /
# emulator first (see README, "Evaluation").
#
# Usage:
#   MODEL_PATH=/path/to/checkpoint bash scripts/eval.sh
# Run `python evaluation/eval.py --help` for every supported flag.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- paths ---------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen2.5-VL-3B-Inquire-Stage1}"   # checkpoint (or base model)
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/data/InquireMobile/data/tasks_en_huawei.jsonl}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/runs}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"

# ---- run settings --------------------------------------------------------
TASK_NAME="${TASK_NAME:-eval_inquire_stage1_huawei_en}"
DEVICE="${DEVICE:-huawei}"         # huawei / device0 / device1 ... (see evaluation/device_runner.py)
ARCH="${ARCH:-qwen2_5_vl}"         # qwen2_5_vl | ui_tars
OSS_BASE="${OSS_BASE:-${REPO_ROOT}/rollouts/gui_agent}"   # mounted OSS prefix shared with the device runner
MAX_STEPS="${MAX_STEPS:-15}"

# ---- judge credentials ---------------------------------------------------
API_KEY="${OPENAI_API_KEY:-${DASHSCOPE_API_KEY:-}}"
API_URL="${OPENAI_BASE_URL:-}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${REPO_ROOT}/evaluation/eval.py" \
    --model_path "${MODEL_PATH}" \
    --eval_dataset_path "${DATASET_PATH}" \
    --eval_result_path "${RESULT_DIR}/${TASK_NAME}.jsonl" \
    --task_name "${TASK_NAME}" \
    --device "${DEVICE}" \
    --arch "${ARCH}" \
    --oss_base "${OSS_BASE}" \
    --max_steps "${MAX_STEPS}" \
    --api_key "${API_KEY}" \
    --api_url "${API_URL}" \
    2>&1 | tee -a "${LOG_DIR}/${TASK_NAME}.log"

python "${REPO_ROOT}/evaluation/summarize_results.py" "${RESULT_DIR}/${TASK_NAME}.jsonl"
