# Experiments

This folder is intentionally self-contained.

- All experiment-specific code lives under `results/experiments/`.
- The scripts here import `src/panda`, but they do not require edits outside this folder.
- Deleting `results/experiments/` removes the local experiment runners, matrices, logs, and outputs together.

Current experiments:

- `exp1_contrast_strength/`
  - Tests whether matched-alpha contrast strength over the shallow view changes TruthfulQA results.
  - Decoders: `official_dola`, `matched_alpha_dola_0p10`, `matched_alpha_dola_0p50`, `matched_alpha_dola_0p95`

- `exp2_horizon_arbitration/`
  - Tests whether longer-horizon disagreement resolution helps in a simplified no-block PAnDa.
  - Decoders: `simple_panda_h1`, `simple_panda_h2`

