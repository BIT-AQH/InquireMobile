import json
import os
from tqdm import tqdm
import requests
from requests.exceptions import RequestException
import time
from openai import OpenAI
from tqdm import tqdm
import random
import time
import base64
import json
import re
import os

def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    
    :param file_path: Path to the JSONL file
    :return: List of dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def download_image(image_url, save_path, max_retries=2, timeout=5):
    """
    Downloads an image from a URL and saves it to a specified path with retry and timeout mechanisms.
    
    :param image_url: URL of the image
    :param save_path: Path where the image will be saved
    :param max_retries: Maximum number of retry attempts (default: 2)
    :param timeout: Timeout in seconds for the request (default: 5)
    """

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(image_url, timeout=timeout)
            response.raise_for_status()  # 抛出非200状态码的异常
            
            with open(save_path, 'wb') as file:
                file.write(response.content)
            # print(f"Successfully downloaded image to {save_path}")
            return True
            
        except RequestException as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 指数退避策略
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download image from {image_url} after {max_retries + 1} attempts")
                print(f"Error: {str(e)}")
                return False
            

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


image_folder_path = "/home/aiqihang.aqh/Appagent/annotation/images"
input_file_path = "/home/aiqihang.aqh/Appagent/annotation/result/【iTAG】不皮_人工介入YN判断3.0_UTF__20250523114232.jsonl"

all_data = read_jsonl(input_file_path)
interactive_data = [data for data in all_data if data["是否需要人工介入"] == "Y"]
# interactive_data = interactive_data[:3]
# len(interactive_data)

# ------------------ 2.下载图片 ------------------
if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)
    
for data in interactive_data:
    image_url = data['screenshot']
    image_name = image_url.split('/')[-1]
    image_path = os.path.join(image_folder_path, image_name)
    
    if not os.path.exists(image_path):
        download_image(image_url, image_path)
        print(f"Downloaded {image_name} to {image_folder_path}")
    else:
        continue
    
    
# ------------------ 3.获取模型反馈 ------------------
output_path = "/home/aiqihang.aqh/Appagent/annotation/result/gpt4o_reason_result.json"
output_data = []

try:
    done_data = json.load(open(output_path, 'r', encoding='utf-8'))
except:
    done_data = []
    
undone_data = [data for data in interactive_data if data['screenshot'].split('/')[-1] not in [d['image_name'] for d in done_data]]
output_data += done_data

system_prompt = '''
请扮演一个gui agent的专家，我会给你一个用户命令，该命令的完成需要与用户交互，请根据截图给出合适的交互式内容，按照json格式给出答案。

#任务背景
基于大模型的手机智能代理在执行用户任务时，可能遇到需要与用户交互的场景，如用户支付或者用户意图模糊的情况，请根据截图和用户命令，给出合适的交互式内容。

#任务定义
交互式任务定义：执行时需要与用户交互，比如
1.风险场景（涉及转账，红包，vip功能订阅，文件删除）
2.隐私安全（涉及账号登录，权限授权，账号头像更换，账号个签更改等，社媒内容发布）
3.意图确认（用户意图不明显，需要人工给出解决方案）
4.其他不确定（剩余的其他情况都算其他不确定）

#动作空间
call_user(content='') # Call the user to ask for more information like logging in, verification, payment etc

#输出格式：
```json
{
    "think": "Your reasoning and thought process."
    "answer": {
        "type": "call_user",
        "content": "string"
    }
}
```
不要输出多余内容和解释，直接给出可以用json.loads解析的答案。
'''

for data in tqdm(undone_data, desc="GPT4o annotation: ", total = len(undone_data)):
    image_url = data['screenshot']
    image_name = image_url.split('/')[-1]
    image_path = os.path.join(image_folder_path, image_name)
    user_prompt = data['instruction']
    
    # 获取模型反馈
    try:
        response_json = get_model_response(image_path, system_prompt, user_prompt)
        # print(f"Response for {image_name}: {response_json}")
        
        response_json['image_name'] = image_name
        response_json['original_info'] = data
        output_data.append(response_json)
            
    except Exception as e:
        print(f"Error processing {image_name}: {str(e)}")

# 保存结果
with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(output_data, output_file, ensure_ascii=False, indent=4)  