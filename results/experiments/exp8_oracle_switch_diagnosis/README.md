# Experiment 8: Oracle Switch Diagnosis

This experiment tests whether the current switching idea is weak because of the
**gate**, or because fixed contrast is simply the better regime under the
current TruthfulQA multiple-choice scoring setup.

Compared decoders:

- `pure_greedy`
  - true final-layer greedy baseline using the repo's direct `greedy` decoder path

- `fanda`
  - fixed contrast-subtracted endpoint on the stateful carried-layer scaffold
  - carries a shallow layer across steps and refreshes it every `update_every`

- `pure_argmax_switchv2`
  - practical stateful switch
  - uses the same carried-layer scaffold as `fanda`
  - switches to contrast only when the greedy and contrast top tokens differ

- `oracle_token_switch`
  - diagnostic upper bound for a switch rule under teacher forcing
  - at each candidate token, chooses whichever view assigns higher logprob to the
    actual token being scored
  - uses the same carried-layer scaffold as `fanda` and `pure_argmax_switchv2`

- `dola`
  - external baseline from the original decoder family

How to read the result:

- if `oracle_token_switch` barely beats or fails to beat `fanda`, then
  fixed contrast is probably the stronger regime on this benchmark
- if `oracle_token_switch` clearly beats `fanda`, then the switching
  idea is viable and the current practical gate is the weak part

Run example:

```bash
python3 results/experiments/exp8_oracle_switch_diagnosis/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode subset \
  --truthfulqa-limit 50
```
