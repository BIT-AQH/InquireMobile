from typing import Dict
from sacrebleu.metrics import BLEU
from argparse import Namespace


# 设置语言和 tokenization 方式
lang = 'none'  # ✅ 改为 'none'，表示不使用特定语言的分词器

args = Namespace(
    tokenize=lang,
    force=False,
    smooth_method='exp',
    smooth_value=None,
    max_ngram_order=4,
    lc=True  # 使用 lc 而不是 lower
)


bleu_metric = BLEU(args)

def _safe_bleu(y_pred: str, y_true: str, normalization: float = 1.0) -> float:
    """
    计算归一化 bleu，范围 0~normalization
    y_pred: 预测串
    y_true: 参考串
    normalization: 最大奖励（默认 1.0，可传 0.5）
    """
    y_pred = y_pred.strip().lower()
    y_true = y_true.strip().lower()
    if not y_pred or not y_true:              # 只要有一个为空，直接返回 0
        return 0.0
    bleu_score = bleu_metric.sentence_score(y_pred, [y_true]).score  # 0~100
    return bleu_score / 100 * normalization


# 测试用例列表
test_cases = [
    {
        "description": "完全匹配",
        "y_pred": "This is a test.",
        "y_true": "this is a test.",
        "normalization": 1.0,
    },
    {
        "description": "完全不匹配",
        "y_pred": "hello world",
        "y_true": "goodbye everyone",
        "normalization": 1.0,
    },
    {
        "description": "空预测",
        "y_pred": "",
        "y_true": "something",
        "normalization": 1.0,
    },
    {
        "description": "空参考",
        "y_pred": "something",
        "y_true": "",
        "normalization": 1.0,
    },
    {
        "description": "部分匹配",
        "y_pred": "The quick brown fox jumps over the lazy dog",
        "y_true": "the fast brown fox leaps over the lazy dog",
        "normalization": 1.0,
    },
    {
        "description": "中文匹配",
        "y_pred": "你啊好啊",
        "y_true": "你好",
        "normalization": 1.0,
    },
]

# 执行测试
for i, case in enumerate(test_cases):
    score = _safe_bleu(case["y_pred"], case["y_true"], case["normalization"])
    print(f"Test Case {i+1}: {case['description']}")
    print(f"  y_pred: {case['y_pred']}")
    print(f"  y_true: {case['y_true']}")
    print(f"  normalization: {case['normalization']}")
    print(f"  Score: {score:.4f}\n")
