#!/usr/bin/env python3
"""
Python wrapper that mimics the behaviour of the original Bash script

    torchrun --nproc_per_node=1 ... | tee -a LOG_FILE

Save this as run_eval.py and execute:
    python run_eval.py

Author: your_name
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import time


def main():
    # ------------------------------------------------------------------
    # 0) 配置区：如有需要可以改成 argparse 解析命令行
    # ------------------------------------------------------------------
    LOG_DIR = Path("/home/aiqihang.aqh/Appagent/logs")
    MODEL_PATH = "/home/aiqihang.aqh/models/Qwen2.5-VL-3B-Inquire-Stage2/"
    DATA_PATH = "/home/aiqihang.aqh/Appagent/data/tasks_zh.jsonl"
    RESULT_PATH = "/home/aiqihang.aqh/Appagent/result/stage2_result_zh.jsonl"
    TASK_NAME = "eval_inquire_stage2_zh"
    MASTER_PORT = "63358"
    MASTER_ADDR = "127.0.0.1"

    # 创建日志目录
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 可以给日志文件加个时间戳，避免覆盖
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"eval_inquire_mobile_stage2_zh_{time_tag}.log"

    # ------------------------------------------------------------------
    # 1) 构造 torchrun 命令
    # ------------------------------------------------------------------
    cmd = [
        "torchrun",
        "--nproc_per_node=1",
        "--nnodes=1",
        "--node_rank=0",
        f"--master_addr={MASTER_ADDR}",
        f"--master_port={MASTER_PORT}",
        "/home/aiqihang.aqh/Appagent/eval.py",
        "--model_path", MODEL_PATH,
        "--eval_dataset_path", DATA_PATH,
        "--eval_result_path", RESULT_PATH,
        "--task_name", TASK_NAME,
    ]

    print(" ".join(cmd))
    print(f"[INFO] Log file  : {log_file.absolute()}")

    # ------------------------------------------------------------------
    # 2) 运行并把 stdout+stderr 同时打到屏幕和日志（tee）
    # ------------------------------------------------------------------
    with open(log_file, "a", buffering=1, encoding="utf-8") as lf:  # line-buffered
        # Popen + PIPE 让我们可以逐行读取输出
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,          # line-buffered
        )

        # 实时遍历输出
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)

        # 等待子进程结束
        proc.wait()

        print(f"[INFO] torchrun exited with returncode {proc.returncode}")
        lf.write(f"[INFO] torchrun exited with returncode {proc.returncode}\n")

    # 若需要把非零返回码视为异常
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    time.sleep(5) 
    main()
