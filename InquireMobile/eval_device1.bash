LOG_DIR="/home/aiqihang.aqh/Appagent/logs/"
TASK_NAME="eval_inquire_GUI_R1_simulator_en"

CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port="13289" \
         /home/aiqihang.aqh/Appagent/eval_device1.py \
         --model_path "/home/aiqihang.aqh/models/GUI-R1/GUI-R1-3B/" \
         --eval_dataset_path "/home/aiqihang.aqh/Appagent/data/tasks_en_simulator.jsonl" \
         --eval_result_path "/home/aiqihang.aqh/Appagent/result/${TASK_NAME}.jsonl" \
         --task_name "${TASK_NAME}" \
         2>&1 | tee -a "${LOG_DIR}/${TASK_NAME}.log"
