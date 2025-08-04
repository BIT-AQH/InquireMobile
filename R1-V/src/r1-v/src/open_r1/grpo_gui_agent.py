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
coordinate_threshold = 0.1

def accuracy_reward(completions, action, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for i, (content, sol) in enumerate(zip(contents, action)):
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
                f.write(f"step_instruction: {kwargs['step_instruction'][i]}\n")
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


TEMPLATE = '''

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 672x1484.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `key`: Perform a key event on the mobile device.\n    - This supports adb's `keyevent` syntax.\n    - Examples: \"volume_up\", \"volume_down\", \"power\", \"camera\", \"clear\".\n* `click`: Click the point on the screen with coordinate (x, y).\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `system_button`: Press the system button.\n* `open`: Open an app on the device.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "click", "long_press", "swipe", "type", "system_button", "open", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=key`, `action=type`, and `action=open`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Think about what action might have been done between the two adjacent screenshots, and then decide what the next step should be to achieve the user's goal. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
'''

# TEMPLATE = '''

# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device."}}
# </tools>

# For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>

# Check the screenshots to see what’s already been done, and then think about the next step. Output the thinking process in <think> </think> and final answer in <tool_call> </tool_call> tags.
# '''
QUESTION_TEMPLATE = 'User goal: {Question}'
# QUESTION_TEMPLATE = "User goal: {Question}\nUse the screenshots from the previous steps to decide what to do next. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    features = datasets.Features({
        "sample_id": datasets.Value("string"),
        "goal": datasets.Value("string"),
        "step_instruction": datasets.Value("string"),
        "action": datasets.Value("string"),
        "image_paths": datasets.Sequence(datasets.Value("string")),
        "images": datasets.Sequence(datasets.Image())
    })
    dataset_root = "/tmp/datasets/android_control/sel_dataset"
    dataset = load_dataset("json", data_files=os.path.join(dataset_root, "metadata.jsonl"), features=features)

    MAX_IMG_NUM = 10
    def load_images(examples):
        examples["images"] = [
            [Image.open(os.path.join(dataset_root, img_path)) for img_path in sample_paths[-MAX_IMG_NUM:]]
            for sample_paths in examples["image_paths"]
        ]
        return examples

    def make_conversation_image(example):
        data = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                    ],
                },
            ],
        }
        if len(example["image_paths"]) > 1 and MAX_IMG_NUM > 1:
            data["prompt"][0]["content"].append({"type": "text", "text": "action list:"})
            data["prompt"][0]["content"].extend([{"type": "image"} for _ in range(len(example["image_paths"][-MAX_IMG_NUM:-1]))])
        data["prompt"][0]["content"].append({"type": "text", "text": "current screen:"})
        data["prompt"][0]["content"].extend([{"type": "image"} for _ in range(len(example["image_paths"][-1:]))])
        data["prompt"][0]["content"].append({"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["goal"]) + TEMPLATE})
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
