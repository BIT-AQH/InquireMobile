import os
import re
import sys
import time
import json
from PIL import Image
import datasets
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "/tmp/saves/qwen25_vl_3b_agent"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda:3"
)

min_pixels = 256*28*28
max_pixels = 1280*28*28
max_pixels = 512*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

TEMPLATE = '''

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Check the screenshots to see what’s already been done, and then think about the next step. Output the thinking process in <think> </think> and final answer in <tool_call> </tool_call> tags.
'''
QUESTION_TEMPLATE = 'User goal: {Question}'

features = datasets.Features({
    "sample_id": datasets.Value("string"),
    "goal": datasets.Value("string"),
    "step_instruction": datasets.Value("string"),
    "action": datasets.Value("string"),
    "image_paths": datasets.Sequence(datasets.Value("string")),
    "images": datasets.Sequence(datasets.Image())
})
dataset_root = "/tmp/datasets/android_control/"
dataset = load_dataset("json", data_files=os.path.join(dataset_root, "metadata_test.jsonl"), features=features)

MAX_IMG_NUM = 5
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
    data["prompt"][0]["content"].extend([{"type": "image"} for _ in range(len(example["image_paths"][-MAX_IMG_NUM:]))])
    data["prompt"][0]["content"].append({"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["goal"]) + TEMPLATE})
    return data

dataset = dataset.map(make_conversation_image).map(load_images, batched=True, batch_size=10)
print("dataset: ", dataset["train"][0])

fout = open('/tmp/saves/pred.tsv', 'w')
for i in range(len(dataset["train"])):
    st = time.time()
    example = dataset["train"][i]
    images = example["images"]
    prompts_text = processor.apply_chat_template(
        example["prompt"], tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = processor(
        text=prompts_text,
        images=images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    image_grid_thw = prompt_inputs["image_grid_thw"]
    resized_width = int(image_grid_thw[0][2] * 14)
    resized_height = int(image_grid_thw[0][1] * 14)
    prompt_inputs.to(model.device)

    prompt_completion_ids = model.generate(**prompt_inputs, max_new_tokens=512)

    prompt_length = prompt_inputs['input_ids'].size(1)
    completion_ids = prompt_completion_ids[:, prompt_length:]
    output_text = processor.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(output_text)
    content_match = re.search(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL)
    content = content_match.group(1).strip() if content_match else content.strip()
    
    fout.write('\t'.join([example['sample_id'], str(resized_width), str(resized_height), example['action'], content])+'\n')
    print(i, time.time()-st)
fout.close()