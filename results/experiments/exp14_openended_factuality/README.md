# Experiment 14: Open-Ended Factuality Stress Test

This experiment asks whether selected-layer persistence matters more when the
decoder has to generate a longer answer instead of only scoring short fixed
multiple-choice strings.

The motivation comes from `exp12`: `fanda_update4` and
`fanda_frozen` were close on the TruthfulQA multiple-choice setup,
but that evaluator is teacher-forced over short candidate answers. In that
setting, `update4` and `frozen` may differ only a few times before the answer
ends.

`exp14` keeps the same TruthfulQA questions but switches the task to
open-ended generation.

Decoder set:

- `exp14_update1`
- `exp14_update2`
- `exp14_update4`
- `exp14_update8`
- `exp14_frozen`

All five use the same direct logit-contrast view and differ only in
`selected_layer` refresh schedule.

## Main Question

When the model must generate a longer factual answer, is:

- `frozen` still enough?
- an interior refresh schedule better?
- or does the schedule barely matter even in free generation?

## Evaluation

This experiment uses a lightweight factuality proxy instead of the standard
TruthfulQA multiple-choice score.

For each TruthfulQA question:

- generate a free-form answer
- compare that answer against the known `mc2` true-answer bank
- compare the same answer against the known `mc2` false-answer bank
- score overlap with token-level F1

Primary metric:

- `oref_margin`
  - `best_true_f1 - best_false_f1`
  - higher is better

Secondary metrics:

- `oref_true_f1`
  - best token-F1 against any true reference
- `oref_win`
  - `1` if the answer matches a true reference better than any false reference

This is intentionally an exploratory proxy:

- it is useful for stress-testing longer generations today
- it is not a substitute for a stronger semantic factuality judge

## Why This Exists

If short fixed answers compress the difference between `update4` and `frozen`,
then a longer autoregressive answer should give stale carried state more time to
drift.

That makes `exp14` the natural follow-up to `exp12`:

- `exp12`: controlled mechanism study on multiple-choice scoring
- `exp14`: longer-generation stress test for whether staleness starts to hurt

## Expected Outputs

- raw prediction CSV
- summary CSV
- pairwise summary CSV
- metadata JSON
- progress logs

## Run Example

```bash
./.venv/bin/python results/experiments/exp14_openended_factuality/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --max-new-tokens 64 \
  --run-id run_01_default
```
