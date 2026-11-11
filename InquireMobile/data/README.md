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
├── tasks_en.jsonl            # InquireBench evaluation tasks (en/zh)
├── tasks_zh.jsonl
├── tasks_en_huawei.jsonl     # ... device / simulator subsets
├── interactive_sft.jsonl     # SFT training trajectories
├── interactive_grpo_*.jsonl  # GRPO / RLVR training trajectories
├── general_grpo_3k.jsonl     # general GRPO training data
├── sft_general.jsonl         # SFT data
├── itag_agent_trajectory_0528_rl_v2.jsonl
└── ...
```

See the dataset card on Hugging Face for the full file list and field schemas.

## License

The data is licensed under [CC BY 4.0](../../LICENSE-DATA).
