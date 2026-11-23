"""Step 1 — pre-label whether a (screenshot, instruction) pair needs user interaction.

A GPT-4o-compatible model labels every screenshot with ``Y`` (needs interaction),
``N`` (does not) or ``U`` (uncertain). These labels are then verified by human
annotators.

Example
-------
    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    python annotation/judge_interaction.py \
        --image_dir annotation_data/images \
        --instruction_file annotation_data/image2instruction.json \
        --output annotation_data/interaction_labels.json
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from common import chat, load_json, save_json

SYSTEM_PROMPT = """
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
"""


def process_single_image(image_name, image_dir, image2instruction):
    user_prompt = f"用户命令：{image2instruction[image_name]}"
    try:
        result = chat(SYSTEM_PROMPT, user_prompt, os.path.join(image_dir, image_name), json_mode=True)
    except Exception:
        return None
    result["image_name"] = image_name
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image_dir", required=True, help="Directory of PNG screenshots.")
    parser.add_argument("--instruction_file", required=True,
                        help="JSON mapping {image_name: user instruction}.")
    parser.add_argument("--output", required=True, help="Output JSON (resumable).")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    image2instruction = load_json(args.instruction_file)
    done = load_json(args.output, default=[])
    done_names = {d["image_name"] for d in done}
    todo = [f for f in sorted(os.listdir(args.image_dir))
            if f.endswith(".png") and f in image2instruction and f not in done_names]

    func = partial(process_single_image, image_dir=args.image_dir, image2instruction=image2instruction)
    results = []
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(func, todo), total=len(todo), desc="Judging"):
            if result is not None:
                results.append(result)

    save_json(done + results, args.output)
    print(f"Saved {len(done) + len(results)} labels to {args.output}")


if __name__ == "__main__":
    main()
