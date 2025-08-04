import os
import sys
from subprocess import call

# === Step 1: 设置环境变量 ===
os.environ["DEBUG_MODE"] = "true"
os.environ["LOG_PATH"] = "/home/aiqihang.aqh/R1-V-Action/R1-V/src/logs/debug_0715.log"

# === Step 2: 构建命令行参数 ===
script_path = "/home/aiqihang.aqh/R1-V-Action/R1-V/src/r1-v/src/open_r1/grpo_operator_ai_v2.py"

arguments = [
    "--output_dir", "/home/aiqihang.aqh/models/saves/qwen2_5vl-3b/interactive_sft_grpo",
    "--model_name_or_path", "/home/aiqihang.aqh/models/Qwen2.5-VL-3B-Instruct",
    "--deepspeed", "/home/aiqihang.aqh/R1-V-Action/R1-V/src/r1-v/local_scripts/zero3.json",
    "--dataset_name", "mock",
    "--max_prompt_length", "1024",
    "--max_completion_length", "4096",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "2",
    "--logging_steps", "1",
    "--bf16", "True",
    "--gradient_checkpointing", "true",
    "--attn_implementation", "flash_attention_2",
    "--max_pixels", "401408",
    "--num_generations", "2",
    "--num_train_epochs", "1",
    "--run_name", "qwen25_vl_3b_interactive",
    "--save_steps", "100",
    "--save_only_model", "true",
    "--report_to", "none",
]

# 如果你想用 wandb 报告，可以取消下面这行注释
# arguments.remove("--report_to") and arguments.extend(["--report_to", "wandb"])

# === Step 3: 模拟 torchrun 启动方式（单卡调试）===
full_cmd = [
    sys.executable,
    script_path,
] + arguments

print("🚀 Running command:")
print(" ".join(full_cmd))

# === Step 4: 执行命令 ===
call(full_cmd)
