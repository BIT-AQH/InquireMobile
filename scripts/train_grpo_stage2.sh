#!/usr/bin/env bash
# Stage-2 GRPO (interactive / call_user reward) training, initialised from the
# stage-1 checkpoint (Qwen2.5-VL-3B-Inquire-Stage1).
#
# Usage:
#   MODEL_NAME_OR_PATH=/path/to/Qwen2.5-VL-3B-Inquire-Stage1 bash scripts/train_grpo_stage2.sh

set -euo pipefail

# ---- paths (override via environment variables) --------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R1V_DIR="${REPO_ROOT}/R1-V/src/r1-v"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/InquireMobile}"       # see data/README.md
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/path/to/Qwen2.5-VL-3B-Inquire-Stage1}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/Qwen2.5-VL-3B-Inquire-Stage1_2}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-11300}"

# Log every rollout (prompt / completion / reward) to LOG_PATH.
export DEBUG_MODE="${DEBUG_MODE:-true}"
export LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/train_grpo_stage2.log}"
mkdir -p "$(dirname "${LOG_PATH}")" "${OUTPUT_DIR}"

cd "${R1V_DIR}/src/open_r1"
torchrun --nproc_per_node "${NUM_GPUS}" --nnodes 1 --node_rank 0 \
         --master_addr 127.0.0.1 --master_port "${MASTER_PORT}" \
        grpo_interactive.py \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path "${MODEL_NAME_OR_PATH}" \
        --deepspeed "${R1V_DIR}/local_scripts/zero3.json" \
        --dataset_name mock \
        --data_root "${DATA_ROOT}" \
        --train_file data/interactive_grpo_401408.jsonl \
        --max_prompt_length 1024 \
        --max_completion_length 2048 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --bf16 True \
        --gradient_checkpointing true \
        --attn_implementation flash_attention_2 \
        --max_pixels 401408 \
        --num_generations 4 \
        --num_train_epochs 2 \
        --run_name "Inquire_Stage1_2" \
        --save_steps 250 \
        --save_only_model true \
        --report_to none
