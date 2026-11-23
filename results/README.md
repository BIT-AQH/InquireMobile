# Results

Judge outputs (`evaluation/eval.py`) on InquireBench. Every line is one task:

```json
{"reason": "...", "reward": 0.25, "status": "success|failure", "inquiry": "success|failure"}
```

| Path | Content |
|---|---|
| `en/`, `zh/` | Final per-model results on the full English / Chinese benchmark (95 tasks each). |
| `raw/` | Raw per-environment runs (`*_huawei_*`: physical device, `*_simulator_*`: emulator), case studies and intermediate runs from which the final files were assembled. |

Models: `Qwen25vl` (Qwen2.5-VL-3B), `UI_R1`, `UI_R1_E`, `GUI_R1`, and our
`Stage1`, `Stage2` and `Stage1_2` variants.

```bash
python evaluation/summarize_results.py results/en/*.jsonl results/zh/*.jsonl
```
