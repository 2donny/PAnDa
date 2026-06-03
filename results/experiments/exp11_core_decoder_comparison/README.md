# Experiment 11: Core Decoder Comparison

This experiment puts the main baseline families and the original switched PAnDa
path into one direct TruthfulQA comparison:

- `pure_greedy`
  - plain final-layer greedy scoring baseline

- `top_k`
  - final-layer distribution truncated to the top `k` tokens, then renormalized

- `top_p`
  - final-layer nucleus distribution with cumulative mass `p`, then renormalized

- `dola`
  - official DoLa contrast rule with relative-top filtering

- `always_contrast`
  - fixed contrast-subtracted endpoint using `final_logits - shallow_logits`

- `panda_switch`
  - original speculative PAnDa block refinement with greedy/contrast arbitration

This is meant to answer a simple practical question:

- how does `always_contrast` compare against standard truncation and DoLa baselines?
- does the original `panda_switch` recover anything meaningful when judged against
  the same baseline family?

Defaults:

- `top_k` uses `--top-k-value 50`
- `top_p` uses `--top-p-value 0.9`
- `panda_switch` uses the standard block defaults here: `jacobi_window_size=4`,
  `jacobi_max_iters=2`

Run example:

```bash
./.venv/bin/python results/experiments/exp11_core_decoder_comparison/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --run-id run_01_default
```
