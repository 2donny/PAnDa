# exp15_prefix_probe

`exp15` is a small case-study follow-up to `exp14_openended_factuality`.

The question is not "which decoder wins on average?" but:

- does `frozen` help mainly by giving a better early factual anchor?
- does `update1` help later by avoiding stale drift?
- at what answer prefix do the two decoders start to separate?

Instead of running another full benchmark, `exp15` reuses the completed
`exp14` manual evaluation sheet and selects a small targeted set of questions:

- some where `frozen > update1`
- some where `update1 > frozen`
- some where both were weak / noisy

For each selected question, it:

1. generates one long open-ended answer with `exp14_frozen`
2. generates one long open-ended answer with `exp14_update1`
3. slices the same answer into prefixes such as:
   - first `10` words
   - first `30` words
   - first `60` words
   - first `100` words
   - full answer
4. saves both:
   - a prefix-level manual review sheet
   - a per-token trace with cumulative word counts

This avoids a common confound:

- we do **not** ask the model to separately generate a "10-word answer" and a
  "30-word answer"
- we evaluate prefixes of the **same** long generation

That makes the experiment a cleaner probe of when drift begins.

## Default design

- decoders: `exp14_frozen`, `exp14_update1`
- question source: `exp14` manual eval CSV
- default case mix:
  - `3` frozen-win questions
  - `3` update1-win questions
  - `2` both-bad questions
- generation length: `120` max new tokens
- prefix checkpoints: `10,30,60,100,full`

## Main outputs

For each run, the runner writes:

- `selected_questions.csv`
  - which questions were chosen and why
- `full_generations.csv`
  - one full answer per `question x decoder`
- `token_trace.csv`
  - per-token trace rows with cumulative word counts
- `prefix_manual_eval.csv`
  - one row per `question x decoder x prefix`
  - includes blank manual review fields ready for annotation
- `metadata.json`

## How to interpret it

Possible patterns:

- `frozen` wins at `10` / `30` words but loses by `60` / `100`
  - evidence for late stale drift
  - supports a decaying or adaptive refresh schedule

- `frozen` wins at every prefix
  - evidence that one fixed shallow layer acts like a question-level anchor

- `update1` already wins at `10`
  - freshness matters even for the core claim
  - freezing is not the right general mechanism

- no consistent pattern
  - failures may depend more on question type than on answer age alone

## Suggested first run

```bash
./.venv/bin/python results/experiments/exp15_prefix_probe/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --max-new-tokens 120 \
  --run-id run_01_default
```

Optional plot after the run:

```bash
./.venv/bin/python scripts/plot_exp15_prefix_probe.py
```

This creates:

- `results/figures/exp15_prefix_probe_summary.svg`
- `results/figures/exp15_prefix_probe_summary.json`

The top panel works immediately from the saved proxy scores.
The bottom panel becomes informative after the prefix manual review sheet is filled.
