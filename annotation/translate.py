"""Step 3 — translate the English ``think`` / ``call_user`` content into Chinese.

Takes the output of ``generate_inquiry.py`` and writes a Chinese version, which
is used to build the bilingual training set.

Example
-------
    export OPENAI_API_KEY=...  OPENAI_BASE_URL=...
    python annotation/translate.py \
        --input annotation_data/inquiry.json \
        --output annotation_data/inquiry_zh.json
"""

import argparse
import re
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from common import chat, load_json, save_json

SYSTEM_PROMPT = "请把content的内容翻译为中文，如果content本身就是中文，请直接返回原始内容。请直接输出翻译后的中文，不需要带content前缀。"


def is_english(text: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9\s.,!?\'\"()\-:;]*$", text))


def translate(text: str) -> str:
    try:
        return chat(SYSTEM_PROMPT, f"content: {text}") or text
    except Exception:
        return text


def process_data(data):
    try:
        if is_english(data["think"]):
            data["think"] = translate(data["think"])
        if is_english(data["answer"]["content"]):
            data["answer"]["content"] = translate(data["answer"]["content"])
        return data
    except Exception as e:
        print(f"Failed to process sample: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Output of generate_inquiry.py (JSON).")
    parser.add_argument("--output", required=True, help="Translated output (JSON).")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    data = load_json(args.input)
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_data, data), total=len(data), desc="Translating"))

    results = [r for r in results if r is not None]
    save_json(results, args.output)
    print(f"Saved {len(results)} samples to {args.output}")


if __name__ == "__main__":
    main()
