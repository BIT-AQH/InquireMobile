# -*- coding: utf-8 -*-
from openai import OpenAI
from tqdm import tqdm
import random
import time
import base64
import json
import re
import os
from multiprocessing import Pool, cpu_count
import functools


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_model_response(system_prompt, user_prompt, model_name="gpt-4o-0806", temperature=0.1, seed=42, max_tokens=4096):
    max_retries = 3
    retry_delay = 3  # 初始等待时间（秒）

    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url=api_url)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{'type': 'text', 'text': user_prompt}]},
                ],
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"翻译失败, 错误: {str(e)}")
                return None

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def is_english(text):
    pattern = r'^[a-zA-Z0-9\s.,!?\'\"()\-:;]*$'
    return bool(re.match(pattern, text))

def process_data(data, system_prompt):
    try:
        think = data['think']
        answer = data['answer']['content']

        if is_english(think):
            user_prompt = f"content: {think}"
            think_translated = get_model_response(system_prompt, user_prompt)
            if think_translated:
                data['think'] = think_translated

        if is_english(answer):
            user_prompt = f"content: {answer}"
            answer_translated = get_model_response(system_prompt, user_prompt)
            if answer_translated:
                data['answer']['content'] = answer_translated

        return data
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        return None

def main():
    input_file_path = "/home/aiqihang.aqh/Appagent/annotation/result/gpt4o_reason_result.json"        
    output_file_path = "/home/aiqihang.aqh/Appagent/annotation/result/gpt4o_reason_result_zh.json"    

    with open(input_file_path, "r") as f:
        en_zh_data = json.load(f)

    system_prompt = "请把content的内容翻译为中文，如果content本身就是中文，请直接返回原始内容。请直接输出翻译后的中文，不需要带content前缀。"
    
    # 获取CPU核心数，设置为进程数
    num_processes = cpu_count()-2
    
    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        # 使用functools.partial固定system_prompt参数
        process_func = functools.partial(process_data, system_prompt=system_prompt)
        results = list(tqdm(pool.imap(process_func, en_zh_data), total=len(en_zh_data), desc="翻译进度"))

    # 过滤掉处理失败的数据
    zh_data = [result for result in results if result is not None]

    with open(output_file_path, "w") as f:
        json.dump(zh_data, f, indent=4, ensure_ascii=False)
        print(f"数据已保存至 {output_file_path}")

if __name__ == '__main__':
    main()
