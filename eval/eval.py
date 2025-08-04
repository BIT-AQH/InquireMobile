# evaluate_agent.py (High-Fidelity Version)

import os
import json
import argparse
import torch
import base64
import time
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from openai import OpenAI
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    GenerationConfig,
)
from transformers.utils import is_peft_available

if is_peft_available():
    from peft import PeftModel

# ==============================================================================
# 1. 辅助函数
# ==============================================================================
def check_and_read_image(file_path, sleep_duration=1):
    # file_path='/data/oss_bucket_0/'+file_path
    while not os.path.exists(file_path):
        print(f"Path '{file_path}' does not exist. Sleeping for {sleep_duration} seconds...")
        time.sleep(sleep_duration)
    
    # 如果路径存在，可以读取文件
    print(f"Path '{file_path}' exists! Reading the file...")
    retries = 3
    attempt = 0
    while attempt < retries:
        try:
            image = Image.open(file_path)
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in 3 seconds...")
            attempt += 1
            time.sleep(3)
    try:
        width, height = image.size
        # print('width, height=')
        # print(width, height)
        # 计算当前像素总数
        scale_factor=0.3
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        images = [image.convert("RGB")]
    except Exception as e:
        print(f"无法处理图像: {e}")
        image = Image.open('/data/oss_bucket_0/yueyin/gui_agent/images/eval/device0/training_step_1/sample_0/step_0.png')
        # 你可以选择跳过这个图像或者用默认图像替代
        # 这里我们选择跳过这个图像
        images = [image.convert("RGB")]
    # images = [Image.open(file_path).convert("RGB")]
    return images

def extract_single_answer(input_string):
    try:
        if '\nassistant\n' in input_string:
            parts_after_assistant = input_string.split('\nassistant\n', 1)[1]
        else:
            parts_after_assistant = input_string

        start_index = parts_after_assistant.find('<tool_call>')
        end_index = parts_after_assistant.find('</tool_call>')

        if start_index == -1 or end_index == -1:
            return parts_after_assistant, {'action': 'continue', 'thought': 'No tool call found.'}

        start = start_index + len('<tool_call>')
        answer_content = parts_after_assistant[start:end_index].strip().replace("'", '"')
        
        try:
            result = json.loads(answer_content)
            if not isinstance(result, dict):
                 return parts_after_assistant, {'action': None}
            return parts_after_assistant, result.get('arguments', {})
        except json.JSONDecodeError:
            return parts_after_assistant, {'action': 'error', 'thought': 'JSON decode error.'}

    except Exception as e:
        print(f'Error during answer extraction: {e}')
        return input_string, {'action': 'error', 'thought': str(e)}

def create_dummy_image(text, output_path, width=1080, height=1920):
    img = Image.new('RGB', (width, height), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except IOError:
        font = ImageFont.load_default()
    
    text_lines = text.split('\n')
    y_text = 50
    for line in text_lines:
        d.text((50, y_text), line, fill=(0,0,0), font=font)
        y_text += 50
    img.save(output_path)

# ==============================================================================
# 2. 你的 GPT-4o 打分函数 (保持不变)
# ==============================================================================
# ... (从上一版回复中复制 gpt4o_reward 和相关函数到这里) ...
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_model_response(image_paths, system_prompt, user_prompt):
    # 注意：这里的 API Key 和 URL 是你提供的，请确保其安全和有效

    max_retries = 3
    retry_delay = 5

    # 动态构建 image_url 列表
    image_content = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            image_content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{encode_image_to_base64(img_path)}'
                }
            })
        else:
            print(f"[Warning] Image path not found, skipping: {img_path}")
            
    if not image_content:
        print("[Error] No valid images to send to GPT-4o.")
        return {"Reason": "No images were provided for evaluation.", "reward": 0}

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {'type': 'text', 'text': user_prompt},
                *image_content  # 将图片内容解包到列表中
            ]
        },
    ]

    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key, base_url=api_url)
            response = client.chat.completions.create(
                model="gpt-4o-0806", # 使用 "gpt-4o" 通常更通用
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            response_json = json.loads(response.choices[0].message.content)
            return response_json
        except Exception as e:
            print(f"GPT-4o API call failed (Attempt {attempt + 1}/{max_retries}). Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return {"Reason": f"API call failed after {max_retries} retries.", "reward": 0}

def gpt4o_reward(ending_image_paths, user_instruction, trajectory, **kwargs):
    rewards = []
    SYSTEM_PROMPT='''
    你是一位评估 GUI 代理任务轨迹的专家。你的任务是评估 GUI 操作任务轨迹的质量和有效性。

    一个轨迹包含以下组件：
    用户指令：描述用户的预期任务（例如，“打开淘宝，将‘iphone16 pro max’加入购物车”）。
    动作历史：包括两个关键部分：
    每一步的推理和动作：由代理执行的一系列动作，包括推理过程和最终执行的动作。
    GUI 截图：初始界面和在执行完每一步动作后的界面截图（从上到下依次排列）。

    在评估轨迹时，请考虑以下关键方面：

    评估标准：

    轨迹连贯性：
    低级步骤和相应动作是否遵循朝向目标指令靠近？动作是否清晰描述且具体？是否存在冗余或不必要的动作？
    任务完成情况：
    轨迹是否成功完成了指令任务？是否完成了所有必要的交互？错误情况是否得到适当处理？
    评分指南：

    根据评估标准，按1到5的等级对轨迹进行评分：

    4: 任务完美完成，成功执行多项动作实现目标。序列逻辑清晰且没有明显冗余。
    3: 任务大部分完成，成功执行多项动作。然而，由于指令中的挑战或不确定性，完成情况不完美，或者过程存在效率低下。
    2: 任务部分完成，执行了一些成功动作。然而，由于任务或环境限制，目标未完全实现，或者序列以循环或错误结束。
    1: 仅执行了少量动作。虽然有完成任务的尝试，但轨迹早期偏离目标或在执行和逻辑上表现出显著低效。
    0: 任务完全失败，开始时没有执行有意义的动作。序列要么立即陷入死锁、重复循环，或在完成任务上没有价值。或者任务完全不可访问。
    注意：如果任务相对复杂，但轨迹表现出有价值的尝试，即使任务没有完全完成，也请考虑向上调整分数。然而，如果任务复杂但轨迹未能执行对任务完成有意义贡献的动作，则不应奖励额外分数。

    您需要根据代理的动作和截图综合评估得分。

    输出格式：
        ```json
        {
            "Reason": <your thoughts and reasoning process for the score>,
            "reward": <your score from 0-4>
        }
        ```
    一定不要输出多余内容，直接输出json格式的答案。
    '''
    
    # gpt4o_reward 只需要处理一个 trajectory
    user_prompt = f"用户命令：{user_instruction}\n\n动作历史：\n{trajectory}"
    reward_json = get_model_response(ending_image_paths, SYSTEM_PROMPT, user_prompt)
    print("GPT-4o Evaluation Response:", reward_json)
    
    try:
        reward = reward_json.get("reward") # 使用 .get() 更安全
        if isinstance(reward, (int, float)):
            # 假设你的评分是0-4，我们将其归一化到0-1
            rewards.append(float(reward) / 4.0)
        else:
            print(f"[Warning] 'reward' field has invalid type {type(reward)}. Returning 0.")
            rewards.append(0.0)
    except Exception:
        print(f"[Error] Failed to parse 'reward' from GPT-4o response. Returning 0.")
        rewards.append(0.0)
        
    return rewards[0] if rewards else 0.0


# ==============================================================================
# 3. 核心的 Rollout 函数
# ==============================================================================

def _prepare_inputs(processor, text, images, device):
    """ 模拟 Trainer 的 _prepare_inputs 行为 """
    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in inputs.items()}

def _generate_for_stage1(processor, prompts, images, device):
    stage_one_prompt = {
        "prompt": [
            prompts[0][0],
            {"role": "user", "content": [{"type": "image"}]},
        ]
    }
    _prompts1 = stage_one_prompt.get('prompt')
    prompts_text = processor.apply_chat_template(_prompts1, tokenize=False, add_generation_prompt=True)
    return _prepare_inputs(processor, prompts_text, images, device)

def _generate_for_stage2(processor, prompts, history_all, images, history, step, device):
    if step == 1:
        stage_two_prompt = {
            "prompt": [
                prompts[0][0],
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": history}]},
                {"role": "user", "content": [{"type": "image"}]},
            ]
        }
    else:
        stage_two_prompt = {
            "prompt": [
                prompts[0][0],
                {"role": "assistant", "content": [{"type": "text", "text": f"这是历史轨迹：{history_all}"}]},
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": history}]},
                {"role": "user", "content": [{"type": "image"}]},
            ]
        }
    _prompts = stage_two_prompt.get('prompt')
    prompts_text = processor.apply_chat_template(_prompts, tokenize=False, add_generation_prompt=True)
    return _prepare_inputs(processor, prompts_text, images, device)


def perform_rollout(model, processor, instruction, oss_image_folder_all, oss_text_folder_all, device, max_steps=15):
    SYSTEM_PROMPT = '''You are a mobile GUI agent. You are given a task and your action history, with the current screenshot and the previous state preceding the last action. You need to perform the next action to complete the task.

    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device."}}}}
    </tools>

    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"name": <function-name>, "arguments": <args-json-object>}}
    </tool_call>

    分析任务和历史动作，给出下一步操作。
    在标签<think> </think>内输出思考过程。
    在标签<action> </action>内输出这一步将要执行的动作。
    在标签<tool_call> </tool_call>内输出最终答案。

    用户任务: {instruction}'''
    model.eval()
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    # 初始化
    # os.makedirs(output_dir, exist_ok=True)
    history_all = ""
    history_record = []
    action_log_path = os.path.join(oss_text_folder_all, "case.jsonl")
    
    # 你的初始 prompt 结构
    prompts = [[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT.format(instruction=instruction)}, # f"{query}  {QUESTION_TEMPLATE}" f"{query}  {QUESTION_TEMPLATE}" QUESTION_TEMPLATE.format(Question=example["problem"])
            ],
        },
    ]]

    step = -1
    history = ""
    while step < max_steps -1 :
        step += 1
        print(f"--- Step {step} ---")

        # 1. 模拟环境交互：创建/加载截图
        current_image_path = os.path.join(oss_image_folder_all, f"step_{step}.png")
        images = check_and_read_image(current_image_path)
        
        # 2. 构建多模态 prompt (高保真复刻)
        if step > 0:
            prev_image_path = os.path.join(oss_image_folder_all, f"step_{step-1}.png")
            images2 = check_and_read_image(prev_image_path)
            images = images2 + images
            prompt_inputs = _generate_for_stage2(processor, prompts, history_all, images, history, step, device)
            history_all = history_all + f'第{step}步:' + history
            history_record.append(history)
        else:
            prompt_inputs = _generate_for_stage1(processor, prompts, images, device)

        # 3. 模型生成
        with torch.no_grad():
            generated_ids = model.generate(**prompt_inputs, generation_config=generation_config)
        
        prompt_length = prompt_inputs["input_ids"].shape[1]
        completion_ids = generated_ids[:, prompt_length:]
        completion_text = processor.batch_decode(completion_ids, skip_special_tokens=True)[0]

        # 4. 解析输出
        history, action = extract_single_answer(completion_text)
        print(f"Model Thought/Action: {action}")
        
        with open(action_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(action, ensure_ascii=False) + '\n')
    with open(action_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({"action": "finished"}, ensure_ascii=False))
    
    # 收集最终用于评估的截图路径
    screenshot_paths = [os.path.join(oss_image_folder_all, f"step_{s}.png") for s in range(step + 1) if os.path.exists(os.path.join(oss_image_folder_all, f"step_{s}.png"))]

    return history_all, screenshot_paths

# ==============================================================================
# 4. 主函数和脚本入口
# ==============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"{args.model_path=}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(device)

    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        print("PEFT adapter detected. Merging model...")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
    
    with open(args.eval_dataset_path, 'r', encoding='utf-8') as f:
        eval_instructions = [json.loads(line)['instruction'] for line in f]
    
    print(f"Loaded {len(eval_instructions)} instructions for evaluation.")

    all_scores = []
    for i, instruction in enumerate(tqdm(eval_instructions, desc="Evaluating Instructions")):
        print(f"\n===== Evaluating Instruction {i+1}/{len(eval_instructions)} =====")
        print(f"Instruction: {instruction}")
        oss_image_folder_all = f'/data/oss_bucket_0/yueyin/gui_agent/images/eval/device0/training_step_{i+1}/sample_0'
        oss_text_folder_all = f'/data/oss_bucket_0/yueyin/gui_agent/instructions/eval/device0/training_step_{i+1}/sample_0'
        os.makedirs(oss_image_folder_all, exist_ok=True)
        os.makedirs(oss_text_folder_all, exist_ok=True)

        trajectory, screenshot_paths = perform_rollout(
            model, processor, instruction, oss_image_folder_all, oss_text_folder_all, device
        )
        
        print("--- Scoring with GPT-4o ---")
        score = gpt4o_reward(
            ending_image_paths=screenshot_paths,
            user_instruction=instruction,
            trajectory=trajectory
        )
        
        all_scores.append(score)
        print(f"Score for this instruction: {score:.4f}")

    if all_scores:
        average_score = sum(all_scores) / len(all_scores)
        print("\n===== Evaluation Finished =====")
        print(f"Total instructions evaluated: {len(all_scores)}")
        print(f"Average Score (0-1 scale): {average_score:.4f}")
    else:
        print("No instructions were evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GUI Agent Model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model checkpoint (or base model)."
    )
    parser.add_argument(
        "--eval_dataset_path", 
        type=str, 
        required=True, 
        help="Path to the evaluation JSONL file with instructions."
    )
    
    args = parser.parse_args()
    main(args)
