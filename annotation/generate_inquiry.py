"""Step 2 — generate the reasoning and ``call_user`` content for interactive samples.

Input is the JSONL exported from human verification, one record per line with:

* ``screenshot``  — URL (or local path) of the screenshot,
* ``instruction`` — the user instruction,
* a label field (``--label_field``) whose value ``Y`` marks samples that
  require user interaction.

For every ``Y`` sample the model produces ``{"think": ..., "answer": {"type":
"call_user", "content": ...}}``.

Example
-------
    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    python annotation/generate_inquiry.py \
        --input annotation_data/human_labels.jsonl \
        --image_dir annotation_data/images \
        --output annotation_data/inquiry.json
"""

import argparse
import json
import os
import shutil
import time

import requests
from tqdm import tqdm

from common import chat, load_json, save_json

SYSTEM_PROMPT = """
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
"""


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch_image(src, dst, max_retries=2, timeout=5):
    """Download (or copy) ``src`` to ``dst``; returns True on success."""
    if os.path.exists(dst):
        return True
    if not src.startswith(("http://", "https://")):
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            return True
        return False
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(src, timeout=timeout)
            response.raise_for_status()
            with open(dst, "wb") as f:
                f.write(response.content)
            return True
        except requests.RequestException as e:
            if attempt == max_retries:
                print(f"Failed to download {src}: {e}")
                return False
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Human-verified labels (JSONL).")
    parser.add_argument("--image_dir", required=True, help="Local screenshot cache directory.")
    parser.add_argument("--output", required=True, help="Output JSON (resumable).")
    parser.add_argument("--label_field", default="是否需要人工介入",
                        help="Field holding the Y/N human label.")
    args = parser.parse_args()

    os.makedirs(args.image_dir, exist_ok=True)
    interactive = [d for d in read_jsonl(args.input) if d.get(args.label_field) == "Y"]

    output = load_json(args.output, default=[])
    done_names = {d["image_name"] for d in output}

    for data in tqdm(interactive, desc="Generating inquiries"):
        image_name = data["screenshot"].split("/")[-1]
        if image_name in done_names:
            continue
        image_path = os.path.join(args.image_dir, image_name)
        if not fetch_image(data["screenshot"], image_path):
            continue
        try:
            result = chat(SYSTEM_PROMPT, data["instruction"], image_path, json_mode=True)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            continue
        result["image_name"] = image_name
        result["original_info"] = data
        output.append(result)
        save_json(output, args.output)

    save_json(output, args.output)
    print(f"Saved {len(output)} samples to {args.output}")


if __name__ == "__main__":
    main()
