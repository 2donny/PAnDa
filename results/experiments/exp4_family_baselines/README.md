# Experiment 4: Family Baselines

This experiment compares the repo-native decoder families before any alpha-specific
interpretation:

- `top_k`
  - teacher-forced scoring under the final-layer distribution truncated to the top `k`
    tokens, then renormalized
- `top_p`
  - teacher-forced scoring under the final-layer nucleus distribution with cumulative
    mass `p`, then renormalized
- `dola`
  - official DoLa log-prob contrast rule with relative-top filtering
- `fixed_alpha_dola_low`
  - previous successful raw-logit fixed-alpha family, now evaluated at `alpha=0.0`
- `fixed_alpha_dola`
  - same raw-logit fixed-alpha family at `alpha=0.5`
- `fixed_alpha_dola_high`
  - same raw-logit fixed-alpha family at `alpha=1.0`

This intentionally uses the repo's built-in fixed-alpha decoders directly.

`top_k` and `top_p` are also adapted to this teacher-forced evaluator:

- `top_k` uses `--top-k-value` (default `50`)
- `top_p` uses `--top-p-value` (default `0.9`)

These are truncated-distribution scoring baselines, not stochastic free-generation runs.

Decoder set:

- `top_k`
- `top_p`
- `dola`
- `fixed_alpha_dola_low`
- `fixed_alpha_dola`
- `fixed_alpha_dola_high`

Run example:

```bash
python3 results/experiments/exp4_family_baselines/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --top-k-value 50 \
  --top-p-value 0.9 \
  --mode subset \
  --truthfulqa-limit 50
```
