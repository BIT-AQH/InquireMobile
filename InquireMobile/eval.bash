#!/usr/bin/env bash
# Example evaluation launch script. Adjust the variables below to your layout,
# or run `python eval.py --help` to see every supported flag.

set -euo pipefail

# ---- paths ---------------------------------------------------------------
MODEL_PATH="/path/to/Qwen2.5-VL-3B-Inquire-Stage1"   # checkpoint (or base model)
DATASET_PATH="data/tasks_en_huawei.jsonl"            # instructions JSONL
RESULT_DIR="result"
LOG_DIR="logs"

# ---- run settings --------------------------------------------------------
TASK_NAME="eval_inquire_stage1_huawei_zh"
DEVICE="huawei"                  # huawei / device0 / device1 / emulator ...
ARCH="qwen2_5_vl"                # qwen2_5_vl | ui_tars
OSS_BASE="/data/oss_bucket_0/USERNAME/gui_agent"
MAX_STEPS=15

# ---- judge credentials (leave empty to use env vars) ---------------------
API_KEY="${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}"
API_URL="${OPENAI_BASE_URL:-}"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 python eval.py \
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
