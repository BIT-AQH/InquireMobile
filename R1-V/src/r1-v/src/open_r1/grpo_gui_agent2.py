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

ds_map = {
    "input_text": "type",
    "scroll": "swipe",
    "navigate_back": "system_button",
    "open_app": "open"
}


def trans2qwen(ori_action, width, height, resized_width, resized_height):
    ori_action_type = ori_action['action_type']
    action_type = ds_map.get(ori_action_type, ori_action_type)
    action = {"name": "mobile_use", "arguments": {"action": action_type}}
    if ori_action_type in ['click', 'long_press']:
        x = int(ori_action['x'] * width / resized_width)
        y = int(ori_action['y'] * height / resized_height)
        action['arguments']['coordinate'] = [x, y]
    if ori_action_type == 'open_app':
        action['arguments']['text'] = ori_action['app_name']
    if ori_action_type == 'input_text':
        action['arguments']['text'] = ori_action['text']
    if ori_action_type == 'scroll':
        x_p1 = int(resized_width * 0.25)
        x_p2 = int(resized_width * 0.5)
        x_p3 = int(resized_width * 0.75)
        y_p1 = int(resized_height * 0.25)
        y_p2 = int(resized_height * 0.5)
        y_p3 = int(resized_height * 0.75)
        d = ori_action['direction']
        if d == 'up':
            c1 = [x_p2, y_p1]
            c2 = [x_p2, y_p3]
        elif d == 'down':
            c2 = [x_p2, y_p1]
            c1 = [x_p2, y_p3]
        elif d == 'left':
            c1 = [x_p1, y_p2]
            c2 = [x_p3, y_p2]
        elif d == 'right':
            c2 = [x_p1, y_p2]
            c1 = [x_p3, y_p2]
        action['arguments']['coordinate'] = c1
        action['arguments']['coordinate2'] = c2
    if ori_action_type == 'navigate_back':
        action['arguments']['button'] = 'back'

    return f'<tool_call>\n{json.dumps(action)}\n</tool_call>'


coordinate_threshold = 0.1

def accuracy_reward(completions, actions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for i, (content, sols) in enumerate(zip(contents, actions)):
        sol = sols[-1]
        reward = 0.0
        try:
            resized_width = 0
            resized_height = 0
            w, h = kwargs['images'][i][0].size

            label_action = json.loads(sol)
            
            # content_match = re.search(r'<answer>(.*?)</answer>', content)
            content_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            pred_action = json.loads(student_answer)
            pred_action = pred_action['arguments']

            resized_width = kwargs['image_grid_thw'][i][2] * 14
            resized_height = kwargs['image_grid_thw'][i][1] * 14
            action = label_action['action_type']
            if 'action' in pred_action:
                # reward += 0.5
                action2 = pred_action['action']

                if action2 == ds_map.get(action, action):
                    # action type reward
                    reward += 1.0
                    if action in ['click', 'long_press']:
                        x1 = label_action['x']
                        y1 = label_action['y']
                        x2, y2 = pred_action['coordinate']
                        # x2 = round(x2*w/resized_width)
                        # y2 = round(y2*h/resized_height)
                        x2 = x2*w/resized_width
                        y2 = y2*h/resized_height
                        
                        x_diff_ratio = abs(x1-x2)/w
                        y_diff_ratio = abs(y1-y2)/h
                        score = (2 - x_diff_ratio - y_diff_ratio) / 2
                        if score > 0.5:
                            reward += score
                        # if abs(x1-x2)/w < coordinate_threshold and abs(y1-y2)/h < coordinate_threshold:
                        #     print('='*50)
                        #     print(x1, y1, x2, y2)
                        #     reward += 1.0

                    if action == 'open_app':
                        x1 = label_action['app_name'].strip().lower()
                        x2 = pred_action['text'].strip().lower()
                        if x1 == x2:
                            reward += 1.0

                    if action == 'input_text':
                        x1 = label_action['text'].strip().lower()
                        x2 = pred_action['text'].strip().lower()
                        if x1 == x2:
                            reward += 1.0

                    if action == 'scroll':
                        x1, y1 = pred_action['coordinate']
                        x2, y2 = pred_action['coordinate2']
                        x_diff_ratio = (x2-x1) / resized_width
                        y_diff_ratio = (y2-y1) / resized_height
                        d = ''
                        if abs(x_diff_ratio) > abs(y_diff_ratio):
                            d = 'left' if x_diff_ratio > 0 else 'right'
                        else:
                            d = 'up' if y_diff_ratio > 0 else 'down'
                        if label_action['direction'].strip().lower() == d:
                            reward += 1.0

                    if action == 'navigate_back':
                        if pred_action['button'].lower() == 'back':
                            reward += 1.0

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
                f.write(f"id: {kwargs['sample_id'][i]}\n")
                f.write(f"goal: {kwargs['goal'][i]}\n")
                f.write(f"step_instruction: {kwargs['step_instructions'][i][-1]}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<tool_call>.*?</tool_call>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


SYSTEM_PROMPT = '''You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Check the screenshots to see what’s already been done, and then think about the next step. Output the thinking process in <think> </think> and final answer in <tool_call> </tool_call> tags.

User goal: '''


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    features = datasets.Features({
        "sample_id": datasets.Value("string"),
        "goal": datasets.Value("string"),
        "step_instructions": datasets.Sequence(datasets.Value("string")),
        "actions": datasets.Sequence(datasets.Value("string")),
        "image_paths": datasets.Sequence(datasets.Value("string")),
        "images": datasets.Sequence(datasets.Image())
    })
    dataset_root = "/tmp/datasets/android_control/sel_dataset"
    dataset = load_dataset("json", data_files=os.path.join(dataset_root, "metadata_multi_step.jsonl"), features=features)

    MAX_IMG_NUM = 5
    def load_images(examples):
        examples["images"] = [
            [Image.open(os.path.join(dataset_root, img_path)) for img_path in sample_paths[-MAX_IMG_NUM:]]
            for sample_paths in examples["image_paths"]
        ]
        return examples

    def make_conversation_image(example):
        # 只能先写死
        width, height = 1080, 2400
        resized_height, resized_width  = smart_resize(height,
            width,
            factor=28,
            min_pixels=script_args.min_pixels,
            max_pixels=script_args.max_pixels,)

        n = min(len(example["image_paths"]), MAX_IMG_NUM)
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT+example["goal"]}
                ],
            },
        ]
        if n > 1:
            for i in range(n-1):
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"}
                    ]
                })
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": trans2qwen(json.loads(example['actions'][i]), width, height, resized_width, resized_height)}
                    ]
                })
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"}
            ]
        })

        data = {
            "prompt": messages
        }
        return data

    dataset = dataset.map(make_conversation_image).map(load_images, batched=True, batch_size=10)
    print("dataset: ", dataset["train"][0])

    # while True:
    #     print('sleep 10')
    #     time.sleep(10)

    
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
