# Experiment 18: Cross-Model Core Transfer

`exp18` is the first cross-model transfer check for the core factual-decoding claim.

The question is:

- if we keep the exact same `50` TruthfulQA multiple-choice questions,
- and keep the decoder family comparison minimal,
- does `fanda` still beat `pure_greedy` and `dola` on models outside the original Qwen-centered setup?

This experiment is intentionally narrower than `exp11`.

- `exp11` asked whether `fanda` beat a broader baseline family on one anchor model
- `exp18` asks whether that simpler `greedy vs dola vs fanda` ranking transfers across models

## Design

- decoders:
  - `pure_greedy`
  - `dola`
  - `fanda`
- metric focus:
  - primary: `mc2`
  - secondary: `mc1`, `mc3`
- subset policy:
  - by default, reuse the exact `50` source indices saved in `exp11_core_decoder_comparison/run_01_default`

The run matrix stores one row per model.

- every enabled row reuses the same locked question set
- every row writes its own results into `runs/<run_id>/`
- one ambiguous model-id row is disabled by default until you replace it with the exact model you want to test

## Why This Ordering

This is meant to answer the highest-priority next question without spending full-dataset compute too early:

- does the overall decoder family result transfer across model families?

Only after that should you expand into:

- `top_k` and `top_p` baselines
- layer-refresh schedule ablations like `update1` / `update2` / `update4` / `frozen`
- full-dataset runs on the most promising transferred models

## Run Examples

List the configured rows:

```bash
./.venv/bin/python results/experiments/exp18_cross_model_transfer/run_experiment.py \
  --list
```

Dry-run one model row without loading weights:

```bash
./.venv/bin/python results/experiments/exp18_cross_model_transfer/run_experiment.py \
  --run-id run_01_gemma3_4b_it \
  --local-files-only \
  --dry-run
```

Run one enabled model row:

```bash
./.venv/bin/python results/experiments/exp18_cross_model_transfer/run_experiment.py \
  --run-id run_01_gemma3_4b_it \
  --local-files-only
```

Run every enabled row:

```bash
./.venv/bin/python results/experiments/exp18_cross_model_transfer/run_experiment.py \
  --local-files-only
```

## Notes

- `--mode subset` and `--truthfulqa-limit 50` are the defaults here because this experiment is designed around the locked `exp11` subset.
- If you want a fresh random subset later, change the row's `subset_strategy` to `random_seed_subset`.
- If you want to activate the disabled DeepSeek-Llama row, replace the placeholder `model_name` in `run_matrix.csv` and set `enabled=true`.
