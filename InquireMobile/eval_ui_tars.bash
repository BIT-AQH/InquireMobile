LOG_DIR="/home/USERNAME/Appagent/logs/"
TASK_NAME="ui_tars_zh_simulator"

CUDA_VISIBLE_DEVICES=1 \
torchrun --nproc_per_node=1 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr="127.0.0.1" \
         --master_port="13127" \
         /home/USERNAME/Appagent/eval_ui_tars.py \
         --model_path "/home/USERNAME/models/UI-TARS-2B-SFT/" \
         --eval_dataset_path "/home/USERNAME/Appagent/data/tasks_zh_simulator.jsonl" \
         --eval_result_path "/home/USERNAME/Appagent/result/${TASK_NAME}.jsonl" \
         --task_name "${TASK_NAME}" \
         2>&1 | tee -a "${LOG_DIR}/${TASK_NAME}.log"

