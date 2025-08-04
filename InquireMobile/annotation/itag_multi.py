# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from openai import OpenAI
from tqdm import tqdm
import random
import time
import base64
import json
import re


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_model_response(image_path, system_prompt, user_prompt, model_name="gpt-4o-0806", temperature=0.1, seed=42, max_tokens=4096):

    max_retries = 3
    retry_delay = 3  # 初始等待时间（秒）

    for attempt in range(max_retries):
        try:
            # 初始化OpenAI客户端
            client = OpenAI(
                api_key=api_key,
                base_url=api_url,
            )

            # API调用

            response = client.chat.completions.create(
                model="gpt-4o-0806",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                'type': 'text',
                                'text': user_prompt
                            },
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/png;base64,{encode_image_to_base64(image_path)}'
                            }
                            }
                        ]
                    },
                ],
                response_format= {"type": "json_object" },
                # response_format = CalendarEvent,
            )

            response_json = json.loads(response.choices[0].message.content)
            return response_json

        except Exception as e:
            if attempt < max_retries:
                wait_time = retry_delay
                time.sleep(wait_time)
            else:
                print(f"JUDGE FAILED, Image Path: {image_path}, Error: {str(e)}")
                raise

def process_single_image(image_name, image_folder_path, image2instruction, system_prompt, done_image_names):
    """单个图片处理函数"""
    if image_name not in done_image_names:
        image_path = os.path.join(image_folder_path, image_name)
        user_instruction = image2instruction[image_name]
        user_prompt = f"用户命令：{user_instruction}"
        response_json = get_model_response(image_path, system_prompt, user_prompt)
        response_json["image_name"] = image_name
        return response_json
    return None

def main():
    # 加载数据
    image2instruction = json.load(open("/Users/aiqihang/code/data/annotation/image2instruction.json", "r"))
    image_folder_path = "/Users/aiqihang/code/data/annotation/images"
    image_paths = [f for f in os.listdir(image_folder_path) if f.endswith(".png")]
    output_path = "/Users/aiqihang/code/data/annotation/gpt4o_judge_result.json"

    # 加载已完成的数据
    with open(output_path, "r") as f:
        done_data = json.load(f)
    done_image_names = [data["image_name"] for data in done_data]

    system_prompt = '''
    请扮演一个gui agent的专家，我会给你一个用户命令和对应的操作截图，你只需要判断该图片是否需要与用户交互，并且给出json格式的答案。
    #任务背景
    基于大模型的手机智能代理在执行用户任务时，可能遇到需要与用户交互的场景，如用户支付或者用户意图模糊的情况，需要人工标注出需要与人工交互的截图。现在请你模仿我给出的一些指令，给出更多交互式的任务；

    #任务定义
    交互式任务定义：执行时需要与用户交互，比如
    1.风险场景（涉及转账，红包，vip功能订阅，文件删除）
    2.隐私安全（涉及账号登录，权限授权，账号头像更换，账号个签更改等，社媒内容发布）
    3.意图确认（用户意图不明显，需要人工给出解决方案）
    4.其他不确定（剩余的其他情况都算其他不确定）

    #输出格式：
    ```json
    {
        "interaction": "Y/"N"/"U"
    }
    ```
    其中"Y"表示需要与用户交互，"N"表示不需要与用户交互，"U"表示不确定，不要输出多余内容和解释，直接给出可以用json.loads解析的答案，如果涉及用户意图不明确的情况，请尽量标注为"U"和"Y"。
    '''

    # 设置进程池
    num_processes = max(1, cpu_count() - 2)  # 留一个CPU核心给系统
    pool = Pool(processes=num_processes)

    # 准备偏函数
    process_func = partial(
        process_single_image,
        image_folder_path=image_folder_path,
        image2instruction=image2instruction,
        system_prompt=system_prompt,
        done_image_names=done_image_names
    )

    # 使用进程池处理图片
    results = []
    with tqdm(total=len(image_paths), desc="Processing images") as pbar:
        for result in pool.imap_unordered(process_func, image_paths):
            if result is not None:
                results.append(result)
            pbar.update(1)

    # 关闭进程池
    pool.close()
    pool.join()

    # 合并结果并保存
    final_results = done_data + results
    with open(output_path, "w") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()