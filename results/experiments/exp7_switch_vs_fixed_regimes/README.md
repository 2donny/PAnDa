# Experiment 7: Switch vs Fixed Regimes

This experiment asks the next decision directly:

- is switching between the binary views actually better than committing to one
  fixed regime?

The compared binary views are:

- greedy view: `final_logits`
- contrast-subtracted view: `final_logits - shallow_logits`

Decoder set:

- `pure_greedy`
  - true final-layer greedy baseline using the repo's direct `greedy` decoder path

- `fanda_greedy`
  - stateful fixed-endpoint baseline that always decodes from `final_logits`
  - still carries a shallow layer across steps and refreshes it every `update_every`

- `fanda`
  - stateful fixed-endpoint baseline that always decodes from `final_logits - shallow_logits`
  - carries a shallow layer across steps and refreshes it every `update_every`

- `pure_argmax_switch`
  - if the greedy and contrast views choose the same top token, keep greedy
  - otherwise switch to contrast

- `pure_argmax_switchv2`
  - uses the same top-token switch rule as `pure_argmax_switch`
  - but runs on the stateful carried-layer scaffold used by `fanda_greedy` and `fanda`
  - carries a shallow layer across steps and refreshes it every `update_every`

- `dola`
  - external baseline from the original decoder family

Why this experiment exists:

- `exp6` showed that the switch variants were close to each other
- the next question is not "which switch rule is cutest?"
- the next question is whether switching beats the fixed endpoints at all
- adding `pure_greedy` separates a true greedy baseline from the stateful binary-endpoint scaffold
- adding `pure_argmax_switchv2` tests whether the simple switch gets stronger or weaker when it shares the same carried-layer scaffold as the fixed endpoints

Run example:

```bash
python3 results/experiments/exp7_switch_vs_fixed_regimes/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode subset \
  --truthfulqa-limit 50
```
