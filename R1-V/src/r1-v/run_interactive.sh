export DEBUG_MODE="true"
export LOG_PATH="./debug_log_3b_agent_v2.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node 4 --nnodes 1 --node_rank 0 \
         --master_addr 127.0.0.1 --master_port 38123 \
        /home/aiqihang.aqh/R1-V-Action/R1-V/src/r1-v/src/open_r1/grpo_interactive.py \
        --output_dir /home/aiqihang.aqh/models/Qwen2.5-VL-3B-Inquire-Stage1\&2 \
        --model_name_or_path /home/aiqihang.aqh/models/saves/qwen2_5vl-3b/interactive_sft/checkpoint-merge \
        --deepspeed /home/aiqihang.aqh/R1-V-Action/R1-V/src/r1-v/local_scripts/zero3.json \
        --dataset_name mock \
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
        --run_name qwen25_vl_3b_interactive \
        --save_steps 250 \
        --save_only_model true \
        --report_to none

# --report_to wandb \
# --max_pixels 1003520 \
# --num_generations 4 \
  # --max_pixels 401408 \
