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

- `exp4_family_baselines/`
  - Tests the repo-native decoder families before alpha-specific interpretation.
  - Decoders: `top_k`, `top_p`, `dola`, `fixed_alpha_dola_low`, `fixed_alpha_dola`, `fixed_alpha_dola_high`

- `exp5_block_ablation/`
  - Tests whether block refinement or slower carried-layer refresh still helps after `always_contrast` became the strongest fixed baseline.
  - Decoders: `always_contrast`, `panda_switch`, `panda_switch_update4`, `panda_always_contrasts`

- `exp6_switch_variants/`
  - Tests four single-pass switch rules for choosing between the greedy view and the contrast-subtracted view without speculative blocks.
  - Decoders: `pure_argmax_switch`, `guarded_argmax_switch`, `sticky_contrast_switch`, `contrast_margin_switch`

- `exp7_switch_vs_fixed_regimes/`
  - Tests whether a simple switch rule beats the two fixed binary endpoints while staying competitive with `dola`.
  - Decoders: `pure_greedy`, `always_greedy`, `always_contrast`, `pure_argmax_switch`, `pure_argmax_switchv2`, `dola`

- `exp8_oracle_switch_diagnosis/`
  - Tests whether switch underperformance comes from a weak practical gate or from fixed contrast already being the stronger regime.
  - Decoders: `pure_greedy`, `always_contrast`, `pure_argmax_switchv2`, `oracle_token_switch`, `dola`

- `exp9_adaptive_lambda_contrast/`
  - Tests whether a conservative adaptive contrast strength can beat fixed always-contrast without returning to hard switching.
  - Decoders: `always_contrast`, `adaptive_lambda_contrast`

- `exp10_ema_risk_switch/`
  - Tests whether a single-pass EMA risk gate can beat matched always-contrast while keeping the same contrast view.
  - Decoders: `always_contrast`, `ema_risk_switch`

- `exp11_core_decoder_comparison/`
  - Compares pure greedy, truncation baselines, DoLa, always-contrast, and the original `panda_switch` in one direct benchmark.
  - Decoders: `pure_greedy`, `top_k`, `top_p`, `dola`, `always_contrast`, `panda_switch`
