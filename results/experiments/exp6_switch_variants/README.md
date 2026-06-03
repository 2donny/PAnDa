# Experiment 6: Switch Variants

This experiment isolates low-latency, single-pass switch rules for choosing
between the two binary views:

- greedy view: `final_logits`
- contrast-subtracted view: `final_logits - shallow_logits`

The goal is to test whether an intuitive local switch can recover useful
contrast corrections without paying the latency cost of speculative blocks or
multi-step horizon rollouts.

Decoder set:

- `pure_argmax_switch`
  - if the greedy and contrast views choose the same top token, keep greedy
  - otherwise switch directly to contrast

- `guarded_argmax_switch`
  - if the top tokens agree, keep greedy
  - if they disagree but the greedy token still remains inside the contrast
    view's top-`k`, keep greedy
  - otherwise switch to contrast

- `sticky_contrast_switch`
  - use the same hard-disagreement rule as the guarded switch
  - once a hard disagreement triggers contrast, stay in contrast briefly for
    the next `sticky_hold_steps` token(s)

- `contrast_margin_switch`
  - if the top tokens agree, keep greedy
  - if they disagree, measure the contrast-view score gap between the contrast
    winner and the greedy winner
  - switch to contrast only when that pairwise gap exceeds
    `exp6_margin_threshold`

Important constraint:

- all four variants are single-pass
- no speculative block is used
- no horizon rollout is used
- the comparison is about the switch rule itself, not alpha tuning

Run example:

```bash
python3 results/experiments/exp6_switch_variants/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode subset \
  --truthfulqa-limit 50
```
