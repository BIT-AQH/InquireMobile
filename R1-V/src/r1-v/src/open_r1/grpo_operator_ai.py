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

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLDAPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from judger import check_click_match, check_back_match

# from qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
# monkey_patch_qwen2_5vl_flash_attn()


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

def check_swipe_directon(x1, y1, x2, y2, width, height):
    x_diff_ratio = (x2-x1) / width
    y_diff_ratio = (y2-y1) / height
    d = ''
    if abs(x_diff_ratio) > abs(y_diff_ratio):
        d = 'left' if x_diff_ratio > 0 else 'right'
    else:
        d = 'up' if y_diff_ratio > 0 else 'down'
    return d

def operatorai_action_parser(action_str, width = None, height = None):
    action = None
    arguments = {}
    if action_str == "UNKNOWN":
        action = "Wait"
    if action_str == "Stop":
        action = "Terminate"
    if action_str == "KEY_BACKSPACE":
        action = "Backspace"
    if action_str == "KEY_BACK":
        action = "Back"
    if action_str == "KEY_HOME":
        action = "Home"
    if action_str == "KEY_ENTER":
        action = "Enter"
    if action_str.startswith("Click"):
        action = 'Tap'
        x, y = [x.strip().strip('(').strip(')') for x in action_str[5:].split(',')]
        x, y = int(float(x) * width), int(float(y) * height)
        arguments =  {'x': x, 'y': y}
    if action_str.startswith("Swipe"):
        action = 'Swipe'
        x1, y1, x2, y2 = [x.strip().strip('(').strip(')') for x in action_str[5:].split(',')]
        x1, y1, x2, y2 = int(float(x1) * width), int(float(y1) * height), int(float(x2) * width), int(float(y2) * height)
        direction = check_swipe_directon(x1, y1, x2, y2, width, height)
        arguments = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'direction': direction}
    if action_str.startswith('Type'):
        action = "Type"
        arguments = {'text': action_str.split(':')[1].strip()}
    
    return {'action': action, 'arguments': arguments}

def standard_to_qwenvl25_parser(action_json):
    valid = False
    if 'action' in action_json:
        action = None
        arguments = {}
        action_type = action_json['action']
        if action_type =='Tap':
            action = 'click'
            coordinate = [action_json['arguments']['x'], action_json['arguments']['y']]
            arguments = {'action': action, 'coordinate': coordinate}
            valid = True
        if action_type =='Long_Press':
            action = 'long_press'
            coordinate = [action_json['arguments']['x'], action_json['arguments']['y']]
            arguments = {'action': action, 'coordinate': coordinate, "time": 5}
            valid = True
        if action_type == 'Swipe':
            action = "swipe"
            coordinate1 = [action_json['arguments']['x1'], action_json['arguments']['y1']]
            coordinate2 = [action_json['arguments']['x2'], action_json['arguments']['y2']]
            arguments = {'action': action, 'coordinate': coordinate1, 'coordinate2': coordinate2}
            valid = True
        if action_type == 'Type':
            action = 'type'
            text=action_json['arguments']['text']
            arguments = {'action': action, 'text': text}
            valid = True
        if action_type == 'Open_App':
            action = 'open'
            app_name=action_json['arguments']['app_name']
            arguments = {'action': action, 'text': app_name}
            valid = True
        if action_type == 'Home':
            action = 'system_button'
            arguments = {'action': action, 'button': "Home"}
            valid = True
        if action_type == 'Back':
            action = 'system_button'
            arguments = {'action': action, 'button': "Back"}
            valid = True
        if action_type == 'Enter':
            action = 'system_button'
            arguments = {'action': action, 'button': "Enter"}
            valid = True
        if action_type == 'Menu':
            action = 'system_button'
            arguments = {'action': action, 'button': "Menu"}
            valid = True
        if action_type == 'Wait':
            action = 'wait'
            arguments = {'action': action, 'time': 5}
            valid = True
        if action_type == 'Terminate':
            action = 'terminate'
            arguments = {'action': action, 'status': "success"}
            valid = True
    if valid:
        return {"name": "mobile_use", "arguments": arguments}
    else:
        print(action_json)
        return {}

def trans2qwen(ori_action, width, height):
    res = operatorai_action_parser(ori_action, width, height)
    return json.dumps(standard_to_qwenvl25_parser(res), ensure_ascii=False)

def process_coordinate(text, width, height):
    pattern = r'\(([0-9.]+),\s*([0-9.]+)\)'
    for match in re.finditer(pattern, text):
        x, y = match.groups()
        if float(x) <= 1.0 and float(y) <= 1.0:
            x, y = int(float(x) * width), int(float(y) * height)
            text = text.replace(match.group(), f"({x}, {y})")
    return text

def process_thought(text):
    if text.startswith('用户选择'):
        text = text[len('用户选择'):]
    elif text.startswith('用户'):
        text = text[len('用户'):]
    return text


coordinate_threshold = 0.1

def accuracy_reward(completions, action_list, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for i, (content, sols) in enumerate(zip(contents, action_list)):
        sol = sols[-1]
        reward = 0.0
        llm_judge = False
        dbg_info = ''
        try:
            resized_width = 0
            resized_height = 0
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
            bbox = kwargs['bbox'][i]
            if 'action' in pred_action:
                # reward += 0.5
                action2 = pred_action['action']

                bbox = kwargs['bbox'][i]
                if action2 == action:
                    # action type reward
                    # reward += 1.0
                    if action in ['click', 'long_press']:
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
                                    llm_judge = True
                                    if check_click_match(kwargs['images'][i][-1], [x1/resized_width, y1/resized_height], [x2/resized_width, y2/resized_height]) == 1:
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
                        if x1 == x2:
                            reward += 1.0

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
                        if bbox[0] == -1:
                            llm_judge = True
                            x1, y1 = pred_action['coordinate']
                            if check_back_match(kwargs['images'][i][-1], [x1/resized_width, y1/resized_height]) == 1:
                                reward += 1.0
                        else:
                            x1, y1 = pred_action['coordinate']
                            x1 = x1 * w / resized_width
                            y1 = y1 * h / resized_height
                            if x1 >= bbox[0] and x1 <= bbox[2] and y1 >= bbox[1] and y1 <= bbox[3]:
                                reward += 1.0
                            dbg_info += f"{bbox}, label: back, pred: [{int(x1)}, {int(y1)}]"
                    elif action2 == 'system_button' and pred_action['button'].lower() == 'back':
                        llm_judge = True
                        x1, y1 = label_action['coordinate']
                        if check_back_match(kwargs['images'][i][-1], [x1/resized_width, y1/resized_height]) == 1:
                            reward += 1.0
            # reward /= 2.0

        except Exception as e:
            print('Exception'+'-'*50)
            print(e)
            print(sol)
            print(student_answer)
            print('-'*50)
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
                f.write(f"subtask: {kwargs['subtask_list'][i][-1]}\n")
                f.write(f"Solution: {sol}\n")
                f.write(f"llm_judge: {llm_judge}\n")
                f.write(f"dbg_info: {dbg_info}\n")
                f.write(f"resize: {resized_width}, {resized_height}\n")
                f.write(f"img size: {w}, {h}\n")
    return rewards

def check_tag_count(s):
    if s.count('<think>') > 1:
        return False
    if s.count('</think>') > 1:
        return False
    if s.count('<action>') > 1:
        return False
    if s.count('</action>') > 1:
        return False
    if s.count('<tool_call>') > 1:
        return False
    if s.count('</tool_call>') > 1:
        return False
    return True

def format_reward(completions, **kwargs):
    # return 0.0
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<action>.*?</action>\s*<tool_call>.*?</tool_call>\s*"
    completion_contents = [completion[0]["content"] for completion in completions]
    tag_flags = [check_tag_count(content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    for content, match, flag in zip(completion_contents, matches, tag_flags):
        if not match or not flag:
            print('format error'+'-'*50)
            print(content, match, flag)
            print('-'*50)
    return [0.5 if match and flag else 0.0 for match, flag in zip(matches, tag_flags)]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


SYSTEM_PROMPT = '''You are a mobile GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

分析任务和历史动作，给出下一步操作。
在标签<think> </think>内输出思考过程。
在标签<action> </action>内输出这一步将要执行的动作。
在标签<tool_call> </tool_call>内输出最终答案。

用户任务: '''


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    features = datasets.Features({
        "sample_id": datasets.Value("string"),
        "goal": datasets.Value("string"),
        "subtask_list": datasets.Sequence(datasets.Value("string")),
        "action_list": datasets.Sequence(datasets.Value("string")),
        "thought_list": datasets.Sequence(datasets.Value("string")),
        "img_list": datasets.Sequence(datasets.Value("string")),
        "bbox": datasets.Sequence(datasets.Value("int32")),
        "img_width": datasets.Value("int32"),
        "img_height": datasets.Value("int32"),
        "images": datasets.Sequence(datasets.Image())
    })
    dataset_root = "/home/jingxuan.xjx/notebook/datasets/OperatorAI/"
    dataset = load_dataset("json", data_files=os.path.join(dataset_root, "agent_train_correction_20250506.jsonl"), features=features)

    MAX_IMG_NUM = 5
    def load_images(examples):
        examples["images"] = [
            [Image.open(os.path.join(dataset_root, 'images', os.path.basename(img_path))) for img_path in sample_paths[-MAX_IMG_NUM:]]
            for sample_paths in examples["img_list"]
        ]
        return examples

    def make_conversation_image(example):
        width, height = example['img_width'], example['img_height']
        resized_height, resized_width  = smart_resize(height,
            width,
            factor=28,
            min_pixels=script_args.min_pixels,
            max_pixels=script_args.max_pixels,)

        n = len(example["img_list"])

        example['action_list'] = [trans2qwen(action, resized_width, resized_height) for action in example['action_list']]
        example['thought_list'] = [process_coordinate(text, resized_width, resized_height) for text in example['thought_list']]
        # example['thought_list'] = [process_thought(text) for text in example['thought_list']]
        example['subtask_list'] = [process_coordinate(text, resized_width, resized_height) for text in example['subtask_list']]

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT+example["goal"]}
                ],
            },
        ]
        if n > 1:
            for i in range(max(0, n-MAX_IMG_NUM), n-1):
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image"}
                    ]
                })
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"<think>\n{example['thought_list'][i]}\n</think>\n"},
                        {"type": "text", "text": f"<action>\n{example['subtask_list'][i]}\n</action>\n"},
                        {"type": "text", "text": f"<tool_call>\n{example['action_list'][i]}\n</tool_call>\n"}
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
