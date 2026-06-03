# Experiment 2: Horizon Arbitration

This experiment isolates the horizon-arbitration question in a simplified no-block PAnDa:

Both variants use the same binary local choice:

- greedy view: `final_logits`
- contrast-subtracted view: `final_logits - shallow_logits`

- `simple_panda_h1`
  - no block
  - resolve disagreement with horizon-1 branch scoring

- `simple_panda_h2`
  - same no-block setup
  - resolve disagreement with horizon-2 rollout scoring

Run example:

```bash
python3 results/experiments/exp2_horizon_arbitration/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode subset \
  --truthfulqa-limit 50
```
