# exp16_mc_persistence_seed_sweep

`exp16` is a one-experiment MC robustness sweep over selected-layer persistence schedules.

The question is:

- if we keep the TruthfulQA subset size at `50`, does the ranking of
  `update1`, `update2`, `update4`, `update8`, and `frozen`
  stay stable across different sampled subsets?

This experiment keeps the evaluator fixed and only changes:

- the selected-layer refresh schedule
- the sampled TruthfulQA subset

## Design

- decoders:
  - `fanda_update1`
  - `fanda_update2`
  - `fanda_update4`
  - `fanda_update8`
  - `fanda_frozen`
- metric focus:
  - primary: `mc2`
  - secondary: `mc1`, `mc3`
- subset size:
  - `50` questions per run by default

The run matrix has five rows:

1. `run_01_anchor_subset`
   - reuses the exact 50-question source-index sequence from
     `exp12_state_persistence_diagnostics/run_01_default`
2. `run_02_seed_101`
3. `run_03_seed_202`
4. `run_04_seed_303`
5. `run_05_seed_404`

The first row gives direct continuity with the earlier mechanism study.
The other four rows test whether the MC ranking survives fresh random subsets.

## Main outputs

Each run writes the usual MC artifacts:

- `*_raw_predictions.csv`
- `*_summary.csv`
- `*_pairwise_summary.csv`
- `*_metadata.json`
- `progress.json`
- `progress.ndjson`

The metadata for the anchor run records:

- the anchor source indices
- the reference experiment and run id
- the path to the original `exp12` metadata file

## How to interpret it

Possible outcomes:

- one decoder wins `mc2` across nearly every subset
  - good evidence for a stable MC default

- `update4` and `frozen` stay close across subsets
  - MC supports stronger persistence but does not cleanly separate the best schedule

- rankings move a lot from subset to subset
  - subset sensitivity is itself the main result

## Suggested run

```bash
./.venv/bin/python results/experiments/exp16_mc_persistence_seed_sweep/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50
```
