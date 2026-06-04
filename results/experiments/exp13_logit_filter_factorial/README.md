# Experiment 13: Logit/Filter Factorial

This experiment is a matched component ablation for the `fanda`
story.

The earlier family baselines already suggested that the raw-logit family was
stronger than official DoLa on this TruthfulQA setup, but those comparisons did
not fully isolate:

- direct raw-logit contrast vs official log-prob contrast
- removal of the relative-top filter

The main confound is that the existing fixed-alpha family uses carried
`selected_layer` state with `update_every = 4`, while official DoLa reselects
its layer every token. So `exp13` rebuilds the comparison as a matched `update1`
factorial study.

All four cells in this experiment:

- use the same step-local JSD layer-selection rule as official DoLa
- refresh `selected_layer` every token
- differ only in:
  - score space: `logprob` vs `raw-logit`
  - relative-top filter: `on` vs `off`

Decoder set:

- `exp13_logprob_top`
  - official-style log-prob contrast with the relative-top filter enabled

- `exp13_logprob_no_top`
  - official-style log-prob contrast with the relative-top filter disabled

- `exp13_logit_top`
  - direct raw-logit contrast with the same relative-top filter applied

- `exp13_logit_no_top`
  - direct raw-logit contrast with no relative-top filter

## Main Question

Which part matters more for the gain over official DoLa:

- switching from log-prob contrast to direct raw-logit contrast?
- removing the relative-top filter?
- or the interaction between them?

## Why This Experiment Exists

If we only compare official DoLa against `fanda`, we change too many
things at once:

- score space
- relative-top filtering
- persistence schedule

`exp12` already isolates the persistence part. `exp13` is meant to isolate the
other two.

## How To Read It

This is a 2x2 factorial.

If `raw-logit` helps regardless of filter state, then the score-space main
effect is likely important.

If `no-filter` helps regardless of score space, then the relative-top removal
main effect is likely important.

If the biggest gain only appears in `raw-logit + no-filter`, then the story is
mostly about the interaction between the two changes rather than either one in
isolation.

## Expected Outputs

The same experiment artifacts as the other runners:

- raw prediction CSV
- summary CSV
- pairwise summary CSV
- metadata JSON
- progress logs

## Run Example

```bash
./.venv/bin/python results/experiments/exp13_logit_filter_factorial/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --run-id run_01_default
```
