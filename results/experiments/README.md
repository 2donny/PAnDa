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
  - Tests whether block refinement or slower carried-layer refresh still helps after `fanda` became the strongest fixed baseline.
  - Decoders: `fanda`, `panda_switch`, `panda_switch_update4`, `panda_fandas`

- `exp6_switch_variants/`
  - Tests four single-pass switch rules for choosing between the greedy view and the contrast-subtracted view without speculative blocks.
  - Decoders: `pure_argmax_switch`, `guarded_argmax_switch`, `sticky_contrast_switch`, `contrast_margin_switch`

- `exp7_switch_vs_fixed_regimes/`
  - Tests whether a simple switch rule beats the two fixed binary endpoints while staying competitive with `dola`.
  - Decoders: `pure_greedy`, `fanda_greedy`, `fanda`, `pure_argmax_switch`, `pure_argmax_switchv2`, `dola`

- `exp8_oracle_switch_diagnosis/`
  - Tests whether switch underperformance comes from a weak practical gate or from fixed contrast already being the stronger regime.
  - Decoders: `pure_greedy`, `fanda`, `pure_argmax_switchv2`, `oracle_token_switch`, `dola`

- `exp9_adaptive_lambda_contrast/`
  - Tests whether a conservative adaptive contrast strength can beat fixed always-contrast without returning to hard switching.
  - Decoders: `fanda`, `adaptive_lambda_contrast`

- `exp10_ema_risk_switch/`
  - Tests whether a single-pass EMA risk gate can beat matched always-contrast while keeping the same contrast view.
  - Decoders: `fanda`, `ema_risk_switch`

- `exp11_core_decoder_comparison/`
  - Compares pure greedy, truncation baselines, DoLa, always-contrast, and the original `panda_switch` in one direct benchmark.
  - Decoders: `pure_greedy`, `top_k`, `top_p`, `dola`, `fanda`, `panda_switch`

- `exp12_state_persistence_diagnostics/`
  - Tests the selected-layer persistence theory behind `fanda` using matched variants that differ only in layer refresh schedule.
  - Decoders: `fanda_update1`, `fanda_update2`, `fanda_update4`, `fanda_frozen`

- `exp13_logit_filter_factorial/`
  - Tests a matched `update1` 2x2 that isolates score space (`logprob` vs `raw-logit`) and the relative-top filter (`on` vs `off`).
  - Decoders: `exp13_logprob_top`, `exp13_logprob_no_top`, `exp13_logit_top`, `exp13_logit_no_top`

- `exp14_openended_factuality/`
  - Tests whether longer open-ended generations expose a stronger selected-layer refresh tradeoff than short multiple-choice answer scoring.
  - Decoders: `exp14_update1`, `exp14_update2`, `exp14_update4`, `exp14_update8`, `exp14_frozen`

- `exp15_prefix_probe/`
  - Tests where `exp14_frozen` and `exp14_update1` begin to diverge by scoring prefixes of the same long generation on targeted disagreement questions.
  - Decoders: `exp14_frozen`, `exp14_update1`

- `exp16_mc_persistence_seed_sweep/`
  - Tests whether the MC ranking of `update1`, `update2`, `update4`, `update8`, and `frozen` stays stable across one anchor subset plus four fresh random TruthfulQA subsets.
  - Decoders: `fanda_update1`, `fanda_update2`, `fanda_update4`, `fanda_update8`, `fanda_frozen`

- `exp17_openended_dola_trace/`
  - Replays selected `exp14_update1` open-ended answers and dumps true token-level traces, using the `exp14` codex manual-eval sheet to pick strong (`manual_score_0_2=2`) and weak (`manual_score_0_2=0`) cases.
  - Decoders: `exp14_update1`
