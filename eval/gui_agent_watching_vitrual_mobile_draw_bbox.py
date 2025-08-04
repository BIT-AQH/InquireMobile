import oss2
import time
import json
import warnings
import os
import subprocess
import oss2
import multiprocessing
import logging
import signal
from typing import Dict, Any, Optional, List
from pypinyin import lazy_pinyin
import re
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from typing import List, Optional

warnings.filterwarnings("ignore")

OSS_BUCKETS = {}

oss_name = 'alimama-creative-public'
# oss_name = 'lab-mllm'
oss_auth = oss2.Auth(OSS_BUCKETS[oss_name]['accessKeyID'], OSS_BUCKETS[oss_name]['accessKeySecret'])
oss_bucket = oss2.Bucket(oss_auth, OSS_BUCKETS[oss_name]['endpoint'], oss_name)
print(f"oss_bucket is {oss_bucket}")    

def convert_text_to_pinyin(text):
    """
    将文本中的中文和中文标点转换为对应的拼音或英文标点
    """
    # 中文标点符号映射表
    punctuation_map = {
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '；': ';',
        '：': ':',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '【': '[',
        '】': ']',
        '（': '(',
        '）': ')',
        '、': ',',
        '《': '<',
        '》': '>',
        '～': '~',
        '…': '...',
        '￥': '$',
        '·': '`',
        '—': '-',
        '－': '-',
        '～': '~',
        '「': '[',
        '」': ']',
        '『': '[',
        '』': ']',
        '〈': '<',
        '〉': '>',
        '＜': '<',
        '＞': '>',
        '［': '[',
        '］': ']',
        '｛': '{',
        '｝': '}',
        '／': '/',
        '＼': '\\',
        '｜': '|',
        '＋': '+',
        '－': '-',
        '＝': '=',
        '％': '%',
        '＃': '#',
        '＠': '@',
        '＆': '&',
        '＊': '*',
    }
    
    # 首先替换标点符号
    for cn_punct, en_punct in punctuation_map.items():
        text = text.replace(cn_punct, en_punct)
    
    # 然后处理中文字符
    parts = re.findall(r'[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+', text)
    result = []
    
    for part in parts:
        # 检查是否为中文
        if any('\u4e00' <= char <= '\u9fff' for char in part):
            result.append(''.join(lazy_pinyin(part)))
        else:
            # 非中文部分保持不变
            result.append(part)
    
    return ''.join(result)

def verify_pc_image(path, min_bytes=1024):
    """
    只校验 PC 端图片：
        1. 文件存在且大于 min_bytes
        2. PIL.verify() 通过
    """
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < min_bytes:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False
    except Exception:
        return False


def capture_and_pull_screenshot(
    mobile_image_path: str,
    pc_image_path: str,
    device_serial: str = "emulator-5554",
    retries: int = 3,
    wait_after_cap: int = 0.3,
) -> bool:
    """
    截图->拉取->校验 PC 图片。
    若校验失败，会自动重试(重新截屏)；全部失败返回 False。
    """
    os.makedirs(os.path.dirname(pc_image_path), exist_ok=True)

    capture_cmd = f'adb -s {device_serial} shell "screencap -p > {mobile_image_path}"'
    pull_cmd    = f'adb -s {device_serial} pull {mobile_image_path} "{pc_image_path}"'
    remove_cmd  = f'adb -s {device_serial} shell "rm {mobile_image_path}"'

    for attempt in range(1, retries + 1):
        try:
            # 1. 远端截屏
            subprocess.run(capture_cmd, shell=True, check=True)
            time.sleep(wait_after_cap*attempt)

            # 2. 拉取到本地
            subprocess.run(pull_cmd, shell=True, check=True)
            time.sleep(wait_after_cap*attempt)

            # 3. 删除手机端临时文件（无论成功与否均尝试）
            subprocess.run(remove_cmd, shell=True, check=False)

            # 4. 校验 PC 端文件
            if verify_pc_image(pc_image_path):
                return True

            print(f"[{attempt}/{retries}] PC image verification failed, retrying...")
            # 若校验失败先删除无效文件再重试
            try:
                os.remove(pc_image_path)
            except FileNotFoundError:
                pass

        except subprocess.CalledProcessError as e:
            print(f"[{attempt}/{retries}] adb error: {e}")
            # 若 adb 命令失败也继续重试
        except Exception as e:
            print(f"[{attempt}/{retries}] unexpected error: {e}")

    print(f"[PC Image Broken]: {pc_image_path}.")
    return False



# def capture_and_pull_screenshot(mobile_image_path, pc_image_path, device_serial='emulator-5554'):
#     try:

#         os.makedirs(os.path.dirname(pc_image_path), exist_ok=True)
        
#         capture_cmd = f'adb -s {device_serial} shell "screencap -p > {mobile_image_path}"'
#         pull_cmd = f'adb -s {device_serial} pull {mobile_image_path} "{pc_image_path}"'
#         remove_cmd = f'adb -s {device_serial} shell "rm {mobile_image_path}"'

#         subprocess.run(capture_cmd, shell=True, check=True)
#         time.sleep(1.5)  # 等待截图完成
#         subprocess.run(pull_cmd, shell=True, check=True)
#         subprocess.run(remove_cmd, shell=True, check=True)
        
#         # log_message(f"Screenshot captured and saved to {pc_image_path}")
#         return True
    
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred: {e}")
#         # return False
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         # return False
        
def local2oss(pc_image_path, oss_image_path):
    print("Uploading to OSS...")
    try:
        oss_bucket.put_object_from_file(key=oss_image_path, filename=pc_image_path)
        print(f"Successfully uploaded to {oss_image_path}")
    except Exception as e:
        print(f"OSS upload error: {repr(e)}")
        
def create_image_folder(local_image_folder, oss_image_folder):
    if not os.path.exists(local_image_folder):
        os.makedirs(local_image_folder)
        print(f"Created local folder {local_image_folder}")
    else:
        print(f"Local folder {local_image_folder} already exists")

    if not oss_bucket.object_exists(oss_image_folder): 
        oss_bucket.put_object(oss_image_folder, b'') 
        print(f"Created oss folder {oss_image_folder}")
    else:
        print(f"Oss folder {oss_image_folder} already exists")
        
def go_to_home_screen(device_serial='emulator-5554'):
    """
    将设备返回到主界面（Home 界面）
    
    :param device_serial: ADB 设备序列号，默认为 emulator-5554
    :return: 执行结果 True/False
    """
    try:
        # 构建 ADB 命令
        cmd = f'adb -s {device_serial} shell input keyevent 3'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Device returned to Home screen successfully.")
            return True
        else:
            print(f"Failed to return to Home screen. Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error returning to Home screen: {e}")
        return False

def execute_mobile_action(action_json, device_serial='emulator-5554'):

    """
    将 JSON 格式的 mobile_use action 转换为 ADB 命令并执行
    
    :param action_json: JSON 格式的 action 字典
    :param device_serial: ADB 设备序列号，默认为 emulator-5554
    :return: 执行结果 True/False
    """
    try:
        # 解析 JSON（如果传入的是字符串）
        if isinstance(action_json, str):
            action_json = json.loads(action_json)
        
        action = action_json.get('action')
        
        # 构建基础 ADB 命令前缀
        base_cmd = f'adb -s {device_serial}'
        # scatter = 1/0.3 # fix
        scatter = 2 # fix
        # 根据不同 action 构建不同的命令
        if action == 'key':
            # 按键事件
            cmd = f'{base_cmd} shell input keyevent {action_json.get("text", "")}'
        
        elif action == 'click':
            x, y = action_json.get('coordinate', [0, 0])
            cmd = f'{base_cmd} shell input tap {x*scatter} {y*scatter}'
        
        elif action == 'long_press':
            x, y = action_json.get('coordinate', [0, 0])
            duration = action_json.get('time', 1)
            if duration>2:
                duration=2
            cmd = f'{base_cmd} shell input swipe {x*scatter} {y*scatter} {x*scatter} {y*scatter} {int(duration * 1000)}'
        
        elif action == 'swipe':
            x1, y1 = action_json.get('coordinate', [0, 0])
            x2, y2 = action_json.get('coordinate2', [0, 0])
            cmd = f'{base_cmd} shell input swipe {x1*scatter} {y1*scatter} {x2*scatter} {y2*scatter}'
        
        # elif action == 'type':
        #     text = action_json.get('text', '')
        #     escaped_text = text.replace(' ', '\\ ')
        #     cmd = f'{base_cmd} shell input text "{escaped_text}"'

        elif action == 'type':
            text = action_json.get('text', '')
            text+=' '
            converted_text = convert_text_to_pinyin(text)
            escaped_text = converted_text.replace(' ', '\\ ')
            cmd = f'{base_cmd} shell input text "{escaped_text}"'
        
        elif action == 'system_button':
            button_map = {
                'Back': '4',
                'Home': '3',
                'Menu': '82',
                'Enter': '66'
            }
            button = action_json.get('button', 'Home')
            key_code = button_map.get(button, '3')
            cmd = f'{base_cmd} shell input keyevent {key_code}'
        
        # elif action == 'open':
        #     app = action_json.get('text', '')
        #     cmd = f'{base_cmd} shell monkey -p {app} -c android.intent.category.LAUNCHER 1'
        
        elif action == 'wait':
            wait_time = action_json.get('time', 1)
            if wait_time>5:
                wait_time=3
            import time
            time.sleep(wait_time)
            return True, action
        
        elif action == 'terminate':
            status = action_json.get('status', 'success')
            print(f"[{device_serial}] Task terminated with status: {status}")
            return status == 'success'
        
        elif action == 'call_user':
            return True, action
              
        
        else:
            print(f"[{device_serial}] Unsupported action: {action}")
            return False, action
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"[{device_serial}] Action '{action}' executed successfully")
            return True, action
        else:
            print(f"[{device_serial}] Action '{action}' failed. Error: {result.stderr}")
            return False, action
    
    except Exception as e:
        print(f"[{device_serial}] Error executing mobile action: {e}")
        return False, action

def get_screen_size(device_serial="emulator-5554"):
    """Get the physical screen size of the Android device."""
    result = subprocess.run(['adb', '-s', device_serial, 'shell', 'wm', 'size'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8').strip()
    _, size = output.split(': ')
    width, height = map(int, size.split('x'))
    return width, height

def clear_tasks(device_serial="emulator-5554", times=5, delay=0.3):
    """Swipe up from the bottom middle to the top middle to clear tasks multiple times."""
    width, height = get_screen_size(device_serial)
    # Send APP_SWITCH key event to open recent tasks
    subprocess.run(['adb', '-s', device_serial, 'shell', 'input', 'keyevent', '187'])
    
    start_x = width // 2
    start_y = height // 2  
    end_x = width // 2
    end_y = 200  # End near the top edge

    for _ in range(times):
        swipe_command = [
            'adb', '-s', device_serial, 'shell', 'input', 'swipe',
            str(start_x), str(start_y), str(end_x), str(end_y)
        ]
        subprocess.run(swipe_command)
        time.sleep(delay)

    

def draw_bbox_on_screenshot(pc_image_path, pc_image_bbox_path, action_json):
    """
    在截图上画框，根据不同action类型进行操作，然后保存到pc_image_bbox_path
    """
    try:
        # 解析 JSON（如果传入的是字符串）
        if isinstance(action_json, str):
            action_json = json.loads(action_json)
        
        action = action_json.get('action')
        
        # 打开截图
        if not os.path.exists(pc_image_path):
            print(f"Image file does not exist: {pc_image_path}")
            return False

        image = Image.open(pc_image_path)
        draw = ImageDraw.Draw(image)

        # 指定字体路径（需要可访问的字体文件路径）
        try:
            font = ImageFont.truetype("arial.ttf", size=50)  # Windows环境下
        except IOError:
            font = ImageFont.load_default(size=50)  # 默认字体
        
        # scatter = 1/0.3 # fix
        scatter = 2 # fix

        # 根据不同 action 构建不同的命令
        if action == 'key':
            # 按键事件
            text = action_json.get('text', '')
            draw.text((30, 30), f"Type: {text}", fill="green", font=font)
        
        elif action == 'click':
            x, y = action_json.get('coordinate', [0, 0])
            x_scaled, y_scaled = x * scatter, y * scatter
            radius = 30
            draw.ellipse([x_scaled - radius, y_scaled - radius, x_scaled + radius, y_scaled + radius], fill="green", outline="green")
            draw.text((x_scaled + 10, y_scaled - 10), "Click", fill="green", font=font)
        
        elif action == 'long_press':
            x, y = action_json.get('coordinate', [0, 0])
            x_scaled, y_scaled = x * scatter, y * scatter
            radius = 30
            draw.ellipse([x_scaled - radius, y_scaled - radius, x_scaled + radius, y_scaled + radius], fill="green", outline="green")
            draw.text((x_scaled + 10, y_scaled - 10), "Long Press", fill="green", font=font)
        
        elif action == 'swipe':
            x1, y1 = action_json.get('coordinate', [0, 0])
            x2, y2 = action_json.get('coordinate2', [0, 0])
            x1_scaled, y1_scaled = x1 * scatter, y1 * scatter
            x2_scaled, y2_scaled = x2 * scatter, y2 * scatter
            draw.line([x1_scaled, y1_scaled, x2_scaled, y2_scaled], fill="green", width=5)
            draw.text((x1_scaled, y1_scaled - 25), "Swipe", fill="green", font=font)

        elif action == 'type':
            text = action_json.get('text', '')
            text+=' '
            converted_text = convert_text_to_pinyin(text)
            escaped_text = converted_text.replace(' ', '\\ ')
            draw.text((30, 30), f"Type: {escaped_text}", fill="green", font=font)
        
        elif action == 'system_button':
            button = action_json.get('button', 'Home')
            draw.text((30, 30), f"System Button: {button}", fill="green", font=font)
        
        # elif action == 'open':
        #     app = action_json.get('text', '')
        #     cmd = f'{base_cmd} shell monkey -p {app} -c android.intent.category.LAUNCHER 1'
        
        elif action == 'wait':
            wait_time = action_json.get('time', 1)
            draw.text((30, 30), f"Wait: {wait_time}s", fill="green", font=font)
        
        elif action == 'terminate':
            status = action_json.get('status', 'success')
            draw.text((30, 30), f"Terminate: {status}", fill="green", font=font)
        
        else:
            draw.text((30, 30), f"Unknown Action: {action}", fill="green", font=font)
            return False
        
        # 保存绘制后的图片
        image.save(pc_image_bbox_path)
        return True
    
    except Exception as e:
        print(f"Error drawing on screenshot: {e}")
        return False


def watch_and_execute_oss_action(
    oss_name: str,
    device_serial: str,
    max_steps: int,
    finish_action: str,
    action_path: str,
    oss_image_folder: str,
    pc_image_folder: str,
    pc_image_bbox_folder: str,
    oss_config: dict
):
    oss_auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    oss_bucket = oss2.Bucket(oss_auth, oss_config['endpoint'], oss_name)

    last_file_size = 0
    image_name = "step_0.png"
    mobile_image_path = os.path.join('/sdcard/', image_name)
    pc_image_path = os.path.join(pc_image_folder, image_name)
    pc_image_bbox_path = os.path.join(pc_image_bbox_folder, image_name)
    oss_image_path = os.path.join(oss_image_folder, image_name)
    capture_and_pull_screenshot(mobile_image_path, pc_image_path, device_serial)
    if not os.path.exists(pc_image_path):
        print(f'{pc_image_path} does not exits')
        pc_image_path = '/Users/aiqihang/code/GUIAgent/images/huawei/home.png'
    time.sleep(0.5)
    local2oss(pc_image_path, oss_image_path)

    while True:
        try:
            meta = oss_bucket.head_object(action_path)
            current_file_size = meta.content_length

            if current_file_size != last_file_size:
                print(f"[{device_serial}] File size changed: {last_file_size} → {current_file_size}")
                action_content = oss_bucket.get_object(action_path).read().decode('utf-8')
                actions = [json.loads(line) for line in action_content.strip().split('\n') if line.strip()]
                step = len(actions)
                latest_action = actions[-1] if actions else None

                if step > max_steps:
                    print(f"[{device_serial}] Reached maximum steps ({max_steps}), stopping.")
                    return True

                if not latest_action:
                    print(f"[{device_serial}] No action found.")
                    latest_action={'action': None}
                    return True

                if latest_action.get('action') == finish_action:
                    print(f"[{device_serial}] Received finish action '{finish_action}', stopping.")
                    return True

                # draw bbox on the screenshot
                draw_bbox_on_screenshot(pc_image_path, pc_image_bbox_path, latest_action)

                print(f"[{device_serial}] Step {step}: {latest_action}")
                result, action = execute_mobile_action(latest_action, device_serial)
                print(f"[{device_serial}] Action result: {result}")

                step_id = step
                image_name = f"step_{step_id}.png"
                mobile_image_path = os.path.join('/sdcard/', image_name)
                pc_image_path = os.path.join(pc_image_folder, image_name)
                pc_image_bbox_path = os.path.join(pc_image_bbox_folder, image_name)
                oss_image_path = os.path.join(oss_image_folder, image_name)
                
                if action not in ['call_user']:
                    if step==1:
                        time.sleep(7)
                    else:
                        time.sleep(1.8)
                    capture_and_pull_screenshot(mobile_image_path, pc_image_path, device_serial)
                    temp=0
                    while not os.path.exists(pc_image_path):
                        temp+=1
                        print(f'{pc_image_path} does not exits')
                        if step_id-temp < 0:
                            pc_image_path = '/Users/aiqihang/code/GUIAgent/images/huawei/home.png'
                        pc_image_path = os.path.join(pc_image_folder, f"step_{step_id-temp}.png")
                    local2oss(pc_image_path, oss_image_path)
                    print(f"[{device_serial}] Screenshot saved as {image_name}")

                    last_file_size = current_file_size
                else:
                    print(f"[{device_serial}] Action is 'call_user', screenshot saved after 10s.")
                    time.sleep(3)
                    capture_and_pull_screenshot(mobile_image_path, pc_image_path, device_serial)
                    temp=0
                    while not os.path.exists(pc_image_path):
                        temp+=1
                        print(f'{pc_image_path} does not exits')
                        if step_id-temp < 0:
                            pc_image_path = '/Users/aiqihang/code/GUIAgent/images/huawei/home.png'
                        pc_image_path = os.path.join(pc_image_folder, f"step_{step_id-temp}.png")
                    local2oss(pc_image_path, oss_image_path)
                    print(f"[{device_serial}] Screenshot saved as {image_name}")

            time.sleep(0.5)
            print("waiting for action...")
        except Exception as e:
            # print(f"[{device_serial}] Error monitoring action.jsonl: {e}", exc_info=True)
            action_content = oss_bucket.get_object(action_path).read().decode('utf-8')
            actions = [json.loads(line) for line in action_content.strip().split('\n') if line.strip()]
            step = len(actions)
            step_id= step
            image_name = f"step_{step}.png"
            pc_image_path = os.path.join(pc_image_folder, image_name)
            pc_image_bbox_path = os.path.join(pc_image_bbox_folder, image_name)
            while not os.path.exists(pc_image_path):
                temp+=1
                print(f'{pc_image_path} does not exits')
                if step_id-temp < 0:
                    pc_image_path = '/Users/aiqihang/code/GUIAgent/images/huawei/home.png'
                pc_image_path = os.path.join(pc_image_folder, f"step_{step_id-temp}.png")
                pc_image_bbox_path = os.path.join(pc_image_bbox_folder, f"step_{step_id-temp}.png")
            local2oss(pc_image_path, oss_image_path)
            print(f"Error monitoring action.jsonl: {e}")
            time.sleep(1)

    return False


def continuous_watch_oss_actions(
    oss_name: str,
    oss_image_folder: str,
    oss_text_folder: str,
    pc_image_folder: str,
    pc_image_bbox_folder: str,
    device_serial: str,
    max_steps: int,
    oss_config: dict,
    finish_action: str = 'finished',
    sample_id: Optional[int] = 1
):
    action_path = os.path.join(oss_text_folder, 'case.jsonl')

    while True:
        try:
            should_stop = watch_and_execute_oss_action(
                oss_name=oss_name,
                device_serial=device_serial,
                max_steps=max_steps,
                finish_action=finish_action,
                action_path=action_path,
                oss_image_folder=oss_image_folder,
                pc_image_folder=pc_image_folder,
                pc_image_bbox_folder=pc_image_bbox_folder,
                oss_config=oss_config
            )
            if should_stop:
                print(f"###DEVICE {device_serial} SAMPLE {sample_id}: Stopping monitoring.")
                go_to_home_screen(device_serial)
                clear_tasks(device_serial=device_serial, times=5, delay=0.3)
                go_to_home_screen(device_serial)
                break
        except KeyboardInterrupt:
            print(f"### DEVICE {device_serial} SAMPLE {sample_id}: Monitoring stopped by user.")
            break
        except Exception as e:
            print(f"### DEVICE {device_serial} SAMPLE {sample_id}: Error occurred. Restarting... {e}")
            time.sleep(5)

 
def process_device(training_step: int, device: str, cuda2virtualenv: Dict[str, str], oss_config: dict, task_name: str, sample_num: int, max_steps: int):
    try:
        serial_number = cuda2virtualenv[device]
        sample_num = sample_num # 需要按照实际运行时 num_generations 修改？
        max_steps = max_steps

        oss_image_folder_all = f'qihang/gui_agent/images/{task_name}/{device}/instruction_{training_step}/'
        oss_text_folder_all = f'qihang/gui_agent/instructions/{task_name}/{device}/instruction_{training_step}/'

        for sample_id in range(0, sample_num):
            try:
                print(f"Device {device}, Sample {sample_id} is running...")
                oss_image_folder = os.path.join(oss_image_folder_all, f'sample_{sample_id}/')
                oss_text_folder = os.path.join(oss_text_folder_all, f'sample_{sample_id}/')
                pc_image_folder = os.path.join('/Users/aiqihang/code/GUIAgent/images/huawei', task_name, device, f'step{training_step}', f'sample{sample_id}/')
                pc_image_bbox_folder = os.path.join('/Users/aiqihang/code/GUIAgent/images/screenshots_bbox', task_name, device, f'step{training_step}', f'sample{sample_id}/')

                os.makedirs(pc_image_folder, exist_ok=True)
                os.makedirs(pc_image_bbox_folder, exist_ok=True)
                # print(f"Created/verified folder: {pc_image_folder}")
                print(oss_text_folder)
                continuous_watch_oss_actions(
                    oss_name='alimama-creative-public',
                    oss_image_folder=oss_image_folder,
                    oss_text_folder=oss_text_folder,
                    pc_image_folder=pc_image_folder,
                    pc_image_bbox_folder=pc_image_bbox_folder,
                    device_serial=serial_number,
                    max_steps=max_steps,
                    oss_config=oss_config,
                    sample_id=sample_id,
                )
            except Exception as e:
                print(f"Error processing sample {sample_id} for device {device}: {e}")
                continue
    except Exception as e:
        print(f"Error in process_device for device {device}: {e}")
        raise

def grpo_sample(training_step: int, cuda_devices: List[str], cuda2virtualenv: Dict[str, str], task_name: str, sample_num: int, max_steps: int):
    oss_config = OSS_BUCKETS['alimama-creative-public']
    processes = []

    try:
        for device in cuda_devices:
            p = multiprocessing.Process(
                target=process_device,
                args=(training_step, device, cuda2virtualenv, oss_config, task_name, sample_num, max_steps)
            )
            processes.append(p)
            p.start()
            print(f"Started process for device {device} (PID: {p.pid})")

        for p in processes:
            p.join()

        print("All devices have completed processing.")

    except Exception as e:
        print(f"Error in grpo_sample: {e}")
        for p in processes:
            if p.is_alive():
                print(f"Terminating process PID: {p.pid}")
                os.kill(p.pid, signal.SIGINT)
        time.sleep(2)
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise


if __name__ == '__main__':
    
    cuda2virtualenv = {
        "device_test": "emulator-5554",
        "device0": "emulator-5554",
        "device1": "emulator-5556",
        "device2": "emulator-5558",
        "device3": "emulator-5560",
        "device4": "emulator-5558",
        "device5": "emulator-5559",
        "device6": "emulator-5560",
        "device7": "emulator-5561",
        "huawei": "PQY0221129026710",
    }
    
    # train
    # cuda_devices = ["device0","device1"] # ,"device2"
    # task_name = "Qwen2_5_VL_3B_Mobile_R1_v0"
    # sample_num = 4

    # eval
    cuda_devices = ["huawei"] # ,"device2"
    max_steps = 15
    task_name = "eval_inquire_stage2_zh" # eval_mobile_r1_complex_inst
    sample_num = 1

    training_step = 40
    while True:
        training_step+=1
        grpo_sample(training_step, cuda_devices, cuda2virtualenv, task_name, sample_num, max_steps)
