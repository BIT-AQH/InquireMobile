# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

import datasets
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import smart_resize
import sacrebleu
# 放在文件顶部，循环外
# bleu_metric = sacrebleu.BLEU()      # or sacrebleu.BLEU()
from typing import Dict
from sacrebleu.metrics import BLEU
from argparse import Namespace

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLDAPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


lang = 'none'  # ✅ 改为 'none'，表示不使用特定语言的分词器
args = Namespace(
    tokenize=lang,
    force=False,
    smooth_method='exp',
    smooth_value=None,
    max_ngram_order=4,
    lc=True 
)
bleu_metric = BLEU(args)  

def _safe_bleu(y_pred: str, y_true: str, normalization: float = 1.0) -> float:
    """
    计算归一化 bleu，范围 0~normalization
    y_pred: 预测串
    y_true: 参考串
    normalization: 最大奖励（默认 1.0，可传 0.5）
    """
    y_pred = y_pred.strip().lower()
    y_true = y_true.strip().lower()
    if not y_pred or not y_true:              # 只要有一个为空，直接返回 0
        return 0.0
    bleu_score = bleu_metric.sentence_score(y_pred, [y_true]).score  # 0~100
    return bleu_score / 100 * normalization




@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def check_swipe_directon(x1, y1, x2, y2, width, height):
    x_diff_ratio = (x2-x1) / width
    y_diff_ratio = (y2-y1) / height
    d = ''
    if abs(x_diff_ratio) > abs(y_diff_ratio):
        d = 'left' if x_diff_ratio > 0 else 'right'
    else:
        d = 'up' if y_diff_ratio > 0 else 'down'
    return d

coordinate_threshold = 0.1

def accuracy_reward(completions, action, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    # bleu_metric = BLEU() 
    for i, (content, sol) in enumerate(zip(contents, action)):
        reward = 0.0
        llm_judge = False
        dbg_info = ''
        try:
            resized_width = kwargs['resized_width'][i]
            resized_height = kwargs['resized_height'][i]
            w, h = kwargs['images'][i][-1].size

            label_action = json.loads(sol)['arguments']
            
            # content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            pred_action = json.loads(student_answer)
            pred_action = pred_action['arguments']

            resized_width = kwargs['image_grid_thw'][i][2] * 14
            resized_height = kwargs['image_grid_thw'][i][1] * 14
            action = label_action['action']
            # bbox = kwargs['bbox'][i]
            if 'action' in pred_action:
                action2 = pred_action['action']

                bbox = kwargs['bbox'][i]
                if action2 == action:
                    # action type reward
                    reward += 1.0
                    if action in ['click', 'long_press']:
                        bbox = kwargs['bbox'][i]
                        x1, y1 = label_action['coordinate']
                        x2, y2 = pred_action['coordinate']
                        if x2 <= resized_width and x2 >= 0 and y2 <= resized_height and y2 >= 0:
                            if bbox[0] == -1:
                                x_diff_ratio = abs(x1-x2)/resized_width
                                y_diff_ratio = abs(y1-y2)/resized_height
                                score = (2 - x_diff_ratio - y_diff_ratio) / 2
                                if score > 0.9:
                                    reward += 1.0
                            else:
                                x1 = x1 * w / resized_width
                                y1 = y1 * h / resized_height
                                x2 = x2 * w / resized_width
                                y2 = y2 * h / resized_height
                                if x2 >= bbox[0] and x2 <= bbox[2] and y2 >= bbox[1] and y2 <= bbox[3]:
                                    reward += 1.0
                                dbg_info += f"{bbox}, label: [{int(x1)}, {int(y1)}], pred: [{int(x2)}, {int(y2)}]"

                    if action == 'open':
                        x1 = label_action['text'].strip().lower()
                        x2 = pred_action['text'].strip().lower()
                        if x1 == x2:
                            reward += 1.0

                    if action == 'type':
                        x1 = label_action['text'].strip().lower()
                        x2 = pred_action['text'].strip().lower()
                        if x1 and x2:
                            bleu_score = _safe_bleu(x2, x1, 1.0)
                            reward += bleu_score
                        # if x1 == x2:
                        #     reward += 1.0

                    if action == 'call_user':       
                        x1 = label_action['content'].strip().lower()
                        x2 = pred_action['content'].strip().lower()
                        # if x1 == x2:
                        #     reward += 0.5
                        if x1 and x2:
                            bleu_score = _safe_bleu(x2, x1, 1.0)
                            reward += bleu_score

                    if action == 'swipe':
                        d1 = check_swipe_directon(*pred_action['coordinate'], *pred_action['coordinate2'], resized_width, resized_height)
                        d2 = check_swipe_directon(*label_action['coordinate'], *label_action['coordinate2'], resized_width, resized_height)
                        if d1 == d2:
                            reward += 1.0

                    if action == 'system_button':
                        if pred_action['button'].lower() == label_action['button'].lower():
                            reward += 1.0
                    
                    if action in ['terminate', 'wait']:
                        reward += 1.0
                elif (action == 'click' and action2 == 'system_button') or (action == 'system_button' and action2 == 'click'):
                    if action == 'system_button' and label_action['button'].lower() == 'back':
                        bbox = kwargs['bbox'][i]
                        if bbox[0] != -1:
                            x1, y1 = pred_action['coordinate']
                            x1 = x1 * w / resized_width
                            y1 = y1 * h / resized_height
                            if x1 >= bbox[0] and x1 <= bbox[2] and y1 >= bbox[1] and y1 <= bbox[3]:
                                reward += 1.0
                            dbg_info += f"{bbox}, label: back, pred: [{int(x1)}, {int(y1)}]"
            # reward /= 2.0

        except Exception as e:
            print('### Action Error')
            print(f"Exception details: {e}")
            print("------ Solution (sol) ------")
            print(sol)
            print("------ Student Answer ------")
            print(student_answer)
            print('-' * 50)
            pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                # f.write(f"id: {kwargs['sample_id'][i]}\n")
                f.write(f"goal: {kwargs['goal'][i]}\n")
                f.write(f"Solution: {sol}\n")
                f.write(f"dbg_info: {dbg_info}\n")
                f.write(f"resize: {resized_width}, {resized_height}\n")
                f.write(f"img size: {w}, {h}\n")
    return rewards

def check_tag_count(s):
    if s.count('<think>') > 1:
        return False
    if s.count('</think>') > 1:
        return False
    if s.count('<tool_call>') > 1:
        return False
    if s.count('</tool_call>') > 1:
        return False
    return True

def format_reward(completions, **kwargs):
    # return 0.0
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*"
    completion_contents = [completion[0]["content"] for completion in completions]
    tag_flags = [check_tag_count(content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    for content, match, flag in zip(completion_contents, matches, tag_flags):
        if not match or not flag:
            print('###Format Error')
            print("----- Completion Content Start -----")
            print(content)
            print("------ Completion Content End ------")
            print(f"Regex matched: {bool(match)}")
            print(f"Tag structure valid: {flag}")
            print('-'*50)
    return [1.0 if match and flag else -1.0 for match, flag in zip(matches, tag_flags)]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    features = datasets.Features({
        "goal": datasets.Value("string"),
        "action": datasets.Value("string"),
        "system": datasets.Value("string"),
        "query": datasets.Value("string"),
        "bbox": datasets.Sequence(datasets.Value("int32")),
        "width": datasets.Value("int32"),
        "height": datasets.Value("int32"),
        "resized_width": datasets.Value("int32"),
        "resized_height": datasets.Value("int32"),
        "img_list": datasets.Sequence(datasets.Value("string")),
        "images": datasets.Sequence(datasets.Image())
    })
    dataset_root = "/home/aiqihang.aqh/Appagent/data/"
    
    dataset = load_dataset("json", data_files=os.path.join(dataset_root, "interactive_grpo_401408.jsonl"), features=features)

    def load_images(examples):
        examples["images"] = [
            [Image.open(img_path) for img_path in sample_paths]
            for sample_paths in examples["img_list"]
        ]
        return examples

    def make_conversation_image(example):

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": example["system"]}
                ],
            },
        ]
        messages.append({
            "role": "user",
            "content": []
        })
        query_list = example["query"].split("<image>")
        for i in range(len(query_list)-1):
            messages[-1]["content"].append({"type": "text", "text": query_list[i]})
            messages[-1]["content"].append({"type": "image"})

        data = {
            "prompt": messages
        }
        return data

    dataset = dataset.map(make_conversation_image).map(load_images, batched=True, batch_size=10)#.shuffle(seed=42)
    print("dataset: ", dataset["train"][0])

    # while True:
    #     print('sleep 10') 
    #     time.sleep(10)

    
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    trainer_cls = Qwen2VLDAPOTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
