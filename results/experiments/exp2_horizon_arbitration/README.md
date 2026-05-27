# Experiment 2: Horizon Arbitration

This experiment isolates the sequential-awareness hypothesis in a simplified no-block PAnDa:

- `simple_panda_h1`
  - no block
  - resolve disagreement with horizon-1 branch scoring

- `simple_panda_h2`
  - same no-block setup
  - resolve disagreement with horizon-2 rollout scoring

Run example:

```bash
python3 results/experiments/exp2_horizon_arbitration/run_experiment.py \
  --model-name HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration \
  --mode subset \
  --truthfulqa-limit 50
```

