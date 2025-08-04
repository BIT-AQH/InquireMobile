from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch

import os
import re
import sys
import ast
import json
import time
from PIL import Image

import pandas as pd

def calc_iou(rect1, rect2):
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

    iou = 1.0 * intersection_area / union_area

    return iou

# model_path = '/tmp/models/Qwen2.5-VL-3B-Instruct'
model_path = '/tmp/saves/qwen25_vl_3b_grounding/checkpoint-100'
# model_path = '/tmp/saves/qwen25_vl_3b_grounding/'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda:3"
)
# print(model)

min_pixels = 256*28*28
max_pixels = 1280*28*28
# max_pixels = 512*28*28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
# print(processor)


dataset_path = '/tmp/datasets/ScreenSpot'
dataset = json.load(open(os.path.join(dataset_path, 'screenspot_mobile.json')))
# print(len(dataset))
# print(dataset[0])

QUESTION_TEMPLATE = 'Please provide the bounding box coordinate of the region this sentence describes: "{Question}".'
TEMPLATE = 'First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format.'
answer_tag_pattern = r'<answer>(.*?)</answer>'
bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

n = 0
c = 0
iou_list = []
sum_iou = 0.0
for d in dataset:
    # print(d)
    img_path = os.path.join(dataset_path, d['img_filename'])
    img = Image.open(img_path)
    w, h = img.size
    # print(img.size)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": QUESTION_TEMPLATE.format(Question=d['instruction']) + TEMPLATE},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # print(inputs)
    image_grid_thw = inputs['image_grid_thw']
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])

    resized_height = image_grid_thw[0][1] * 14
    resized_width = image_grid_thw[0][2] * 14

    content_match = re.search(r'<answer>(.*?)</answer>', output_text[0], re.DOTALL)
    content = content_match.group(1).strip() if content_match else output_text[0].strip()
    bbox_match = re.search(bbox_pattern, content)
    if bbox_match:
        bbox = [int(int(bbox_match.group(1))*w/resized_width), int(int(bbox_match.group(2))*h/resized_height), int(int(bbox_match.group(3))*w/resized_width), int(int(bbox_match.group(4))*h/resized_height)]
    else:
        bbox = [-1, -1, -1, -1]

    n += 1
    cx = (bbox[0]+bbox[2])/2
    cy = (bbox[1]+bbox[3])/2
    if cx >= d['bbox'][0] and cx <= d['bbox'][0] + d['bbox'][2] and cy >= d['bbox'][1] and cy <= d['bbox'][1] + d['bbox'][3]:
        c += 1
    iou = calc_iou(bbox, [d['bbox'][0], d['bbox'][1], d['bbox'][0] + d['bbox'][2], d['bbox'][1] + d['bbox'][3]])
    iou_list.append(iou)
    sum_iou += iou
    print(n, c, round(sum_iou / n, 6))
print(n, c/n, round(sum_iou / n, 6))

bins = pd.cut(iou_list, bins=10, include_lowest=True)
distribution = pd.value_counts(bins, sort=False)
print(distribution)