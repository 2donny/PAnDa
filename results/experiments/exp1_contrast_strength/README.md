# Experiment 1: Contrast Strength

This experiment isolates the first motivation:

- official DoLa as the baseline
- matched-alpha DoLa variants that keep the official DoLa structure, but scale the shallow log-prob view by `alpha`

Decoder set:

- `official_dola`
- `matched_alpha_dola_0p10`
- `matched_alpha_dola_0p50`
- `matched_alpha_dola_0p95`

Run example:

```bash
python3 results/experiments/exp1_contrast_strength/run_experiment.py \
  --model-name HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration \
  --mode subset \
  --truthfulqa-limit 50
```

