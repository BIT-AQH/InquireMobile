LOG_DIR="/home/aiqihang.aqh/Appagent/logs/"
TASK_NAME="eval_inquire_stage1_huawei_zh_7"

CUDA_VISIBLE_DEVICES=0 \
torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port="15103" \
         /home/aiqihang.aqh/Appagent/eval.py \
         --model_path "/home/aiqihang.aqh/models/Qwen2.5-VL-3B-Inquire-Stage1/" \
         --eval_dataset_path "/home/aiqihang.aqh/Appagent/data/tasks_en_huawei.jsonl" \
         --eval_result_path "/home/aiqihang.aqh/Appagent/result/${TASK_NAME}.jsonl" \
         --task_name "${TASK_NAME}" \
         2>&1 | tee -a "${LOG_DIR}/${TASK_NAME}.log"




