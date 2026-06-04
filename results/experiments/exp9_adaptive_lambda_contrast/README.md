# Experiment 9: Adaptive Lambda Contrast

This experiment tests whether a **conservative continuous contrast strength**
works better than fixed full contrast.

Compared decoders:

- `fanda`
  - fixed contrast-subtracted endpoint on the stateful carried-layer scaffold
  - carries a shallow layer across steps and refreshes it every `update_every`
  - uses `scores = final_logits - shallow_logits`

- `adaptive_lambda_contrast`
  - uses the same stateful carried-layer scaffold as `fanda`
  - replaces hard full contrast with
    `scores = final_logits - lambda_t * shallow_logits`
  - computes `lambda_t` from:
    - normalized final-layer entropy
    - a penalty when greedy is more confident than contrast
  - defaults to a conservative range `lambda_t in [0.5, 1.0]`

How to read the result:

- if `adaptive_lambda_contrast` beats `fanda`, then full contrast was
  probably too blunt and a continuous correction strength is worthwhile
- if it ties or loses, then the current setup likely prefers persistent full
  contrast over per-token strength adaptation

Default adaptive rule:

```text
signal_t =
    uncertainty_weight * normalized_final_entropy
    - confidence_gap_weight * max(0, greedy_confidence - contrast_confidence)

lambda_t =
    clamp(lambda_min + (lambda_max - lambda_min) * signal_t,
          lambda_min, lambda_max)
```

Run example:

```bash
python3 results/experiments/exp9_adaptive_lambda_contrast/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode subset \
  --truthfulqa-limit 50
```

Included run rows:

- `run_01_default`
  - conservative adaptive floor: `lambda_min = 0.5`

- `run_02_lambda_floor_025`
  - more aggressive adaptive floor: `lambda_min = 0.25`
