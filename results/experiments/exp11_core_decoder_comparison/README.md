# Experiment 11: Core Decoder Comparison

This experiment puts the main baseline families and the original switched PAnDa
path into one direct TruthfulQA comparison:

- `pure_greedy`
  - plain final-layer greedy scoring baseline

- `top_k`
  - final-layer distribution truncated to the top `k` tokens, then renormalized

- `top_p`
  - final-layer nucleus distribution with cumulative mass `p`, then renormalized

- `top_p_backoff`
  - same nucleus truncation as `top_p`, but teacher-forced MC scoring falls back to
    the full-distribution gold-token logprob whenever the gold token falls outside
    the kept nucleus

- `dola`
  - official DoLa contrast rule with relative-top filtering

- `fanda`
  - fixed contrast-subtracted endpoint using `final_logits - shallow_logits`

- `panda_switch`
  - original speculative PAnDa block refinement with greedy/contrast arbitration

This is meant to answer a simple practical question:

- how does `fanda` compare against standard truncation and DoLa baselines?
- does the original `panda_switch` recover anything meaningful when judged against
  the same baseline family?

Defaults:

- `top_k` uses `--top-k-value 50`
- `top_p` uses `--top-p-value 0.9`
- `panda_switch` uses the standard block defaults here: `jacobi_window_size=4`,
  `jacobi_max_iters=2`
- `top_p_backoff` is opt-in via `--include-top-p-backoff` so the historical exp11
  default decoder set stays unchanged

Run example:

```bash
./.venv/bin/python results/experiments/exp11_core_decoder_comparison/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --run-id run_01_default
```

To include the finite-scoring `top_p_backoff` variant in the same comparison:

```bash
./.venv/bin/python results/experiments/exp11_core_decoder_comparison/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --run-id run_01_default \
  --include-top-p-backoff
```
