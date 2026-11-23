# Data

The training and evaluation datasets for InquireMobile are hosted on Hugging
Face rather than in this repository:

**<https://huggingface.co/datasets/PG23/InquireMobile>**

## Download

```bash
pip install -U huggingface_hub
bash data/download.sh               # JSONL files + screenshots
bash data/download.sh --no-images   # JSONL files only
```

The dataset is placed under `data/InquireMobile/`:

```
data/InquireMobile/
├── data/
│   ├── tasks_en.jsonl            # InquireBench evaluation tasks (en)
│   ├── tasks_zh.jsonl            # InquireBench evaluation tasks (zh)
│   ├── tasks_en_extended.jsonl   # extended English task set
│   ├── tasks_{en,zh}_huawei.jsonl     # physical-device subsets
│   ├── tasks_{en,zh}_simulator.jsonl  # emulator subsets
│   ├── tasks_zh_huawei_7.jsonl   # physical-device subset (7 tasks)
│   ├── interactive_sft.jsonl     # stage-1 SFT trajectories
│   ├── interactive_grpo_401408.jsonl  # stage-2 GRPO / RLVR trajectories
│   ├── general_grpo_5k.jsonl     # general GRPO training data
│   ├── sft_general.jsonl         # SFT data (general)
│   ├── sft_pos.jsonl             # SFT data (positive)
│   └── instructions.jsonl        # raw instruction pool
└── images/                       # screenshots referenced by the JSONL files
    ├── trajectory/
    └── instruction/
```

Image paths inside the JSONL files (e.g. `img_list`) are relative to
`data/InquireMobile/`; pass this directory as `--data_root` to the training
script.

See the dataset card on Hugging Face for the full file list and field schemas.

## License

The data is licensed under [CC BY 4.0](../LICENSE-DATA).
