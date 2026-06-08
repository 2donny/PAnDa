# exp17_openended_dola_trace

This experiment replays a small set of `exp14_update1` open-ended generations and
dumps real token-level selected-layer traces for plotting.

Selection source:

- `results/experiments/exp14_openended_factuality/runs/run_01_default/run_01_default_manual_eval.csv`
- decoder filter: `exp14_update1`
- strong group: `manual_score_0_2 = 2`
- weak group: `manual_score_0_2 = 0`
- ranking within each group: `proxy_oref_margin`

Why this exists:

- `exp14` saved full answers and summary metrics, but not per-token traces.
- `exp17` reuses the already reviewed open-ended answers to choose a focused set of
  cases, then reruns only those prompts to capture the real generation-time trace.

Main artifacts:

- `run_01_default_selected_cases.csv`
  - one row per selected question with the source manual-eval label, the saved
    source answer, the regenerated answer, and trace summaries
- `run_01_default_token_trace.csv`
  - one row per generated token with `selected_layer`, `token_step`, and token text
- `run_01_default_metadata.json`
  - experiment metadata and source-manual-eval provenance
