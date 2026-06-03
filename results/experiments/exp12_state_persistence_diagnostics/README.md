# Experiment 12: State Persistence Diagnostics

This experiment is not mainly about winning TruthfulQA. It is a mechanism study
for the theory behind `always_contrast`:

- a hallucination-prone local mode may persist for a few nearby tokens
- reselecting the correction layer every token may be noisier than necessary
- carrying `selected_layer` for a short span may stabilize the correction signal
- but holding it too long may make it stale

So `exp12` compares matched `always_contrast` variants that differ only in how
often the shallow correction layer is refreshed:

- `always_contrast_update1`
  - refresh the selected shallow layer every token

- `always_contrast_update2`
  - refresh every 2 teacher-forced token steps

- `always_contrast_update4`
  - refresh every 4 steps
  - this matches the current default stateful `always_contrast` behavior

- `always_contrast_frozen`
  - pick the best JSD layer at step 0, then keep it for the rest of the answer

## Main Question

Does short-lived persistence reduce correction-signal thrash without becoming too
stale?

## Primary Mechanism Metrics

The main evidence is in the summary CSV, not only in `mc1/mc2/mc3`:

- `switch_rate`
  - how often the selected layer changes across the answer

- `selected_layer_match_rate`
  - how often the carried layer still matches the per-step JSD-best layer

- `refresh_rate`
  - how often a decoder actually refreshes the layer

- `avg_oracle_jsd_gap`
  - average gap between the per-step JSD-best layer and the actually used layer
  - larger means the carried layer is becoming stale

- `avg_selection_margin`
  - average contrast-vs-greedy selection separation

- `avg_risk_score`
  - average disagreement intensity between greedy and contrast views

- `trigger_rate`
  - how often disagreement crosses the current risk trigger

## How To Read It

The theory is supported if:

- `update4` lowers `switch_rate` relative to `update1`
- `update4` keeps a reasonably high `selected_layer_match_rate`
- `update4` keeps `avg_oracle_jsd_gap` much lower than `frozen`
- any factuality gains, if they appear, are accompanied by these mechanism shifts

The theory is weakened if:

- `update4` does not reduce switching
- `update4` reduces switching but only by becoming almost as stale as `frozen`
- `update1` and `update4` look almost identical on the mechanism metrics

## Figure Guide

The companion figure for this experiment is:

- `results/figures/exp12_state_persistence_hypothesis.svg`

The left panel is the main mechanism plot:

- `x-axis`: `switch_rate`
  - farther left means the decoder changes `selected_layer` less often
  - this is less token-level layer thrash

- `y-axis`: `selected_layer_match_rate`
  - higher means the carried layer still agrees more often with the step-local best layer
  - this is less stale carried state

- bubble color: `avg_oracle_jsd_gap`
  - cooler means smaller gap to the step-local best layer
  - warmer means the carried layer has drifted farther away

- bubble size: `mc2`
  - larger means better quality on the same run

The right panel is just a quality check:

- it shows `mc1`, `mc2`, and `mc3` for the same four refresh schedules
- this keeps the mechanism story tied to actual evaluation quality

The intended reading is:

- `update1` is very fresh but very jittery
- `frozen` is perfectly stable but more stale
- `update4` is the compromise point: much less jitter than `update1`, but less stale than `frozen`

## What `oracle` Means Here

`oracle` does **not** mean ground-truth answer correctness.

Here it means:

- if we ignored persistence at this token step and re-checked every candidate shallow layer right now
- which layer would look best under the same JSD layer-selection rule?

So:

- `oracle_best_layer` = the step-local best shallow layer at that token
- `selected_layer` = the layer the decoder is actually carrying and using
- `selected_layer_match_rate` = how often those two are the same
- `avg_oracle_jsd_gap` = how far the carried layer has drifted from that step-local best layer

This is why a high oracle-match rate means the carried state is still fresh, not merely stuck.

## Plain-Language Intuition

`selected_layer` is the shallow layer used as the correction source.

Instead of re-choosing that layer at every token, we can keep it for a short
stretch of nearby tokens. In plain language:

- use the same correction layer for a short span
- so the decoder stays more stable
- and does not overreact every step the way DoLa can

Another short way to say it:

- do not change your mind every token

## Why This Matters For Token Selection

The carried layer changes the scoring rule used to rank the next token. Roughly:

```text
token_score(v) = final_logits(v) - shallow_logits(selected_layer, v)
```

So if `selected_layer` changes:

- the penalty on each candidate token changes
- the ranking of candidate tokens can change
- the chosen best token can flip

That means token choice can change for two different reasons:

1. the context genuinely changed
2. the correction layer changed, so the scoring function itself changed

The hypothesis is that DoLa can suffer from too much of `2`.

Carrying the same `selected_layer` for a short span helps because:

- nearby tokens often belong to the same phrase
- the decoder keeps the same correction lens for that phrase
- token selection changes more because the context changed than because the layer selector twitched

But holding the layer too long creates the opposite problem:

- token ranking becomes stale
- this is why `frozen` is not ideal
- `update4` is trying to sit in the middle: stable enough to avoid noisy flips, fresh enough to still pick good tokens

## Why These Metrics Help The Hypothesis

The hypothesis is not only that `update4` gets a better score. It is more specific:

- `update1` should be too jittery
- `frozen` should be too stale
- `update4` should be the better compromise

So we need two kinds of evidence:

- `switch_rate` tells us whether jitter was reduced
- `oracle` metrics tell us whether the decoder became stale while reducing that jitter

This rules out the bad interpretation:

- maybe `update4` works only because it blindly sticks to one layer

If `update4` lowers `switch_rate` **and** stays clearly better than `frozen` on
`selected_layer_match_rate` and `avg_oracle_jsd_gap`, then it is not just stuck.
It is smoothing noisy layer flips while still tracking the step-local best layer
reasonably well.

## Run Example

```bash
./.venv/bin/python results/experiments/exp12_state_persistence_diagnostics/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --run-id run_01_default
```
