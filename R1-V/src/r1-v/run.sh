#!/usr/bin/env bash
# GRPO (operator-AI v2) training launch script. Adjust the paths below.

set -euo pipefail

# ---- paths ---------------------------------------------------------------
CODE_DIR="/path/to/InquireMobile_dev"                    # repo root
TRAIN_SCRIPT="${CODE_DIR}/R1-V/src/r1-v/src/open_r1/grpo_operator_ai_v2.py"
DEEPSPEED_CONFIG="${CODE_DIR}/R1-V/src/r1-v/local_scripts/zero3.json"
MODEL_NAME_OR_PATH="/path/to/models/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/path/to/models/saves/qwen2_5vl-3b/interactive_sft_grpo"

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_3b_agent_v2.txt"

CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="14345" \
    "${TRAIN_SCRIPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --dataset_name mock \
    --max_prompt_length 1024 \
    --max_completion_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_generations 2 \
    --num_train_epochs 1 \
    --run_name qwen25_vl_3b_operator_ai \
    --save_steps 100 \
    --save_only_model true \
    --report_to none
