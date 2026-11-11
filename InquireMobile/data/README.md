# Data

The training and evaluation datasets for InquireMobile are hosted on Hugging
Face rather than in this repository:

**<https://huggingface.co/datasets/PG23/InquireMobile>**

## Download

```bash
# install the HF CLI once
pip install "huggingface_hub[cli]"

# download the whole dataset
huggingface-cli download PG23/InquireMobile --repo-type dataset --local-dir ./data
```

Or use the helper script:

```bash
bash download.sh
```

The download restores the original layout under `data/`:

```
data/
├── tasks_en.jsonl            # InquireBench evaluation tasks (en)
├── tasks_zh.jsonl            # InquireBench evaluation tasks (zh)
├── tasks_en_extended.jsonl   # extended English task set
├── tasks_{en,zh}_huawei.jsonl   # Huawei-device subsets
├── tasks_{en,zh}_simulator.jsonl  # simulator subsets
├── tasks_zh_huawei_7.jsonl   # Huawei-device subset (7 tasks)
├── interactive_sft.jsonl     # stage-1 SFT trajectories
├── interactive_grpo_401408.jsonl  # stage-2 GRPO / RLVR trajectories
├── general_grpo_5k.jsonl     # general GRPO training data
├── sft_general.jsonl         # SFT data (general)
├── sft_pos.jsonl             # SFT data (positive)
├── instructions.jsonl        # raw instruction pool
├── interactive_added.jsonl   # added interaction samples
└── case_study.jsonl          # case-study examples
```

See the dataset card on Hugging Face for the full file list and field schemas.

## License

The data is licensed under [CC BY 4.0](../../LICENSE-DATA).
