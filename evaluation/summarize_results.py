"""Summarise judge results produced by ``evaluation/eval.py``.

Each line of a result file is ``{"reason", "reward", "status", "inquiry"}``.
Reported metrics:

* ``SR``     — task success rate    (``status == "success"``)
* ``ISR``    — inquiry success rate (``inquiry == "success"``)
* ``Reward`` — mean normalised trajectory score in [0, 1]

Usage
-----
    python evaluation/summarize_results.py results/en/*.jsonl results/zh/*.jsonl
"""

import argparse
import json
import os


def summarize(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    n = len(rows)
    if n == 0:
        return {"n": 0, "sr": 0.0, "isr": 0.0, "reward": 0.0}

    def reward(r):
        v = r.get("reward")
        return float(v) if isinstance(v, (int, float)) else 0.0

    return {
        "n": n,
        "sr": 100.0 * sum(r.get("status") == "success" for r in rows) / n,
        "isr": 100.0 * sum(r.get("inquiry") == "success" for r in rows) / n,
        "reward": sum(reward(r) for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="Result JSONL files.")
    args = parser.parse_args()

    width = max(len(os.path.basename(p)) for p in args.files)
    print(f"{'file':<{width}}  {'#':>4}  {'SR(%)':>6}  {'ISR(%)':>6}  {'Reward':>6}")
    for path in args.files:
        s = summarize(path)
        print(f"{os.path.basename(path):<{width}}  {s['n']:>4}  {s['sr']:>6.1f}  {s['isr']:>6.1f}  {s['reward']:>6.3f}")


if __name__ == "__main__":
    main()
