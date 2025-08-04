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
import ast
import time
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image

import datasets
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


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


def rectangle_intersection_union(rect1, rect2):
    # rect1 和 rect2 是两个长方形，格式为 (x1, y1, x2, y2)
    # 其中 (x1, y1) 是左下角坐标，(x2, y2) 是右上角坐标

    # 解包两个长方形的坐标
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # 计算交集区域的边界
    inter_x1 = max(x1_1, x1_2)  # 交集区域的左边界
    inter_y1 = max(y1_1, y1_2)  # 交集区域的下边界
    inter_x2 = min(x2_1, x2_2)  # 交集区域的右边界
    inter_y2 = min(y2_1, y2_2)  # 交集区域的上边界

    # 如果交集区域的宽度或高度小于等于0，则没有交集
    inter_width = inter_x2 - inter_x1
    inter_height = inter_y2 - inter_y1

    if inter_width <= 0 or inter_height <= 0:
        intersection_area = 0
    else:
        intersection_area = inter_width * inter_height

    # 计算两个长方形的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 并集面积 = 长方形1的面积 + 长方形2的面积 - 交集面积
    union_area = area1 + area2 - intersection_area

    return intersection_area, union_area


def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    try:
        json_output = ast.literal_eval(json_output)
    except Exception as e:
        end_idx = json_output.rfind('"}') + len('"}')
        truncated_text = json_output[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)
    return json_output


def accuracy_reward(completions, bbox, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

    for i, (content, sol) in enumerate(zip(contents, bbox)):
        reward = 0.0
        try:
            resized_width = 0
            resized_height = 0
            iou = 0.0
            w, h = kwargs['image'][i].size

            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()

            resized_width = kwargs['image_grid_thw'][i][2] * 14
            resized_height = kwargs['image_grid_thw'][i][1] * 14

            bbox_match = re.search(bbox_pattern, student_answer)
            if bbox_match:
                bbox = [int(int(bbox_match.group(1))*w/resized_width), int(int(bbox_match.group(2))*h/resized_height), int(int(bbox_match.group(3))*w/resized_width), int(int(bbox_match.group(4))*h/resized_height)]
                sol_bbox = [sol[0]*w, sol[1]*h, sol[2]*w, sol[3]*h]
                intersection_area, union_area = rectangle_intersection_union(bbox, sol_bbox)
                iou = intersection_area / union_area
                if iou > 0.5:
                    reward = 1.0
                elif iou > 0.1:
                    reward = iou * 2

            # pred = parse_json(student_answer)[0]
            # if 'bbox_2d' in pred and len(pred['bbox_2d']) == 4:
            #     # reward += 0.5

            #     x1_1, y1_1, x2_1, y2_1 = [sol[0]*w, sol[1]*h, sol[2]*w, sol[3]*h]
            #     x1_2, y1_2, x2_2, y2_2 = [pred['bbox_2d'][0]*w/resized_width, pred['bbox_2d'][1]*h/resized_height, pred['bbox_2d'][2]*w/resized_width, pred['bbox_2d'][3]*h/resized_height]
            #     cx_1 = (x1_1 + x2_1) / 2
            #     cy_1 = (y1_1 + y2_1) / 2
            #     cx_2 = (x1_2 + x2_2) / 2
            #     cy_2 = (y1_2 + y2_2) / 2
            #     area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            #     area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            #     # reward -= abs(cx_1 - cx_2) / w
            #     # reward -= abs(cy_1 - cy_2) / h
            #     # reward -= (abs(x1_1 - x1_2) / w + abs(y1_1 - y1_2) / h) / 2
            #     # reward -= (abs(x2_1 - x2_2) / w + abs(y2_1 - y2_2) / h) / 2

            #     # reward -= max(area2 / area1 - 1, 0)

            #     # if x1_2 >= x1_1 and x2_2 <= x2_1:
            #     #     reward += 0.1
            #     # if y1_2 >= y1_1 and y2_2 <= y2_1:
            #     #     reward += 0.1

            #     intersection_area, union_area = rectangle_intersection_union(
            #         [x1_1, y1_1, x2_1, y2_1],
            #         [x1_2, y1_2, x2_2, y2_2]
            #     )
            #     iou = intersection_area / union_area
            #     # if iou > 0.1:
            #     #     reward += intersection_area / union_area * 2
            #     if iou > 0.5:
            #         reward += 1.0

            #     # cx = (pred['bbox_2d'][0] + pred['bbox_2d'][2]) / 2 / resized_width
            #     # cy = (pred['bbox_2d'][1] + pred['bbox_2d'][3]) / 2 / resized_height
            #     # if cx > sol[0] and cx < sol[2] and cy > sol[1] and cy < sol[3]:
            #     #     reward += 0.5
        except Exception as e:
            print(e)
            print(sol)
            print(student_answer)
            pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"img: {kwargs['img_filename'][i]}\n")
                f.write(f"resized: {resized_width} x {resized_height}\n")
                f.write(f"instruction: {kwargs['instruction'][i]}\n")
                # f.write(f"Solution: {sol}\n")
                if iou > 0.0:
                    f.write("label: [{}, {}, {}, {}]\n".format(int(sol[0]*w), int(sol[1]*h), int(sol[2]*w), int(sol[3]*h)))
                    # f.write("pred: [{}, {}, {}, {}]\n".format(int(pred['bbox_2d'][0]*w/resized_width), int(pred['bbox_2d'][1]*h/resized_height), int(pred['bbox_2d'][2]*w/resized_width), int(pred['bbox_2d'][3]*h/resized_height)))
                    f.write("pred: [{}, {}, {}, {}]\n".format(*bbox))
                    f.write(f"area reward: {round(float(intersection_area / union_area), 4)}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


# QUESTION_TEMPLATE = 'Generate the bounding box of "{Question}".'
# QUESTION_TEMPLATE = 'Instruction: {Question}. To complete the instruction, output the bounding box of the target area.'
QUESTION_TEMPLATE = 'Please provide the bounding box coordinate of the region this sentence describes: "{Question}".'
# TEMPLATE = '''Output the thinking process in <think> </think> tags, and bbox in <answer> </answer> tags in json format.
# i.e., <think> reasoning process here </think>
# <answer>```json
# [{"bbox_2d": [x1, y1, x2, y2], "label": "text"}]
# ```</answer>'''
TEMPLATE = 'First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.'


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    features = datasets.Features({
        "instruction": datasets.Value("string"),
        "bbox": datasets.Sequence(datasets.Value("float")),
        "img_filename": datasets.Value("string"),
        "image": datasets.Image()
    })
    dataset_root = "/tmp/datasets/rico"
    dataset = load_dataset("json", data_files=os.path.join(dataset_root, "metadata.jsonl"), features=features)

    def load_images(examples):
        examples["image"] = [
            Image.open(os.path.join(dataset_root, 'combined', img_path)) for img_path in examples["img_filename"]
        ]
        return examples

    def make_conversation_image(example):
        data = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["instruction"]) + TEMPLATE},
                    ],
                },
            ],
        }
        return data

    dataset = dataset.map(make_conversation_image).map(load_images, batched=True, batch_size=10)
    print("dataset: ", dataset["train"][0])

    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
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
