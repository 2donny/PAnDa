# Experiment 10: EMA Risk Switch

This experiment tests whether a **single-pass temporal risk memory** can improve
on a matched `fanda` control without returning to speculative blocks.

Compared decoders:

- `fanda`
  - fixed contrast-subtracted endpoint on the carried-layer scaffold
  - under this experiment, the shallow layer is fixed by the run row

- `ema_risk_switch`
  - uses the same contrast view `final_logits - shallow_logits`
  - keeps one causal EMA risk state across the decoded sequence
  - switches to contrast only when:
    - the greedy and contrast top tokens differ
    - the EMA risk crosses a threshold
  - can optionally keep contrast active for a short sticky hold

Default risk rule:

```text
instantaneous_risk_t =
    weighted_mean(
        normalized_final_entropy,
        1 - top1_top2_probability_gap,
        normalized_jsd(final_probs, shallow_probs)
    )

ema_risk_t =
    beta * ema_risk_{t-1} + (1 - beta) * instantaneous_risk_t

use_contrast_t =
    token_mismatch_t and ema_risk_t >= risk_threshold
```

Important constraint:

- this is a **single-pass** experiment
- no speculative block is used
- no adaptive lambda is used
- the intended novelty is the **horizontal temporal memory**, not a new contrast view

Run example:

```bash
python3 results/experiments/exp10_ema_risk_switch/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode subset \
  --truthfulqa-limit 50
```

Included run rows:

- `run_01_layer4_default`
  - fixed layer `4`
  - `beta = 0.8`
  - equal weights on entropy, margin risk, and layer JSD
  - `risk_threshold = 0.55`

- `run_02_layer8_default`
  - same rule, but fixed layer `8`

How to read the result:

- if `ema_risk_switch` beats `fanda`, then temporal smoothing of local
  risk is doing something useful beyond memoryless switching
- if it ties or loses, then the repo likely still prefers persistent contrast on
  this benchmark slice
