# PAnDa

PAnDa is a public research repository for **Parallel-block Adaptive Contrast DoLa** and related decoding baselines. It packages the current evaluator, benchmark runners, saved development artifacts, and paper notes into a compact public layout.

The repo is built for **re-running decoder comparisons and inspecting saved development artifacts**, not as a polished benchmark suite with publication-final claims.

## What PAnDa does

PAnDa keeps two fixed contrast regimes inside the same block-parallel decoding pass:

- a safer low-alpha view
- a stronger truth-seeking high-alpha view

Instead of generating two full answers and reranking them afterward, PAnDa compares those views locally inside each speculative block, detects the first meaningful divergence point, and only applies truth-biased arbitration from that point onward.

The current evaluator also includes these comparison decoders:

- `greedy`
- `dola`
- `fixed_alpha_dola_low`
- `fixed_alpha_dola`
- `fixed_alpha_dola_high`
- `panda`

## Repository status

- This repository is a cleaned public research release, not a full historical workspace mirror.
- The strongest checked-in PAnDa evidence is still a **development TruthfulQA sanity artifact**, not a publication-final benchmark package.
- Historical `stage*` names remain in some saved metadata and directories for provenance compatibility, even though the active public preset name is `panda`.

## Install

Run from the repository root:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -e .
```

If the selected model is gated or not already cached locally, export one of:

```bash
export HF_TOKEN=...
# or
export HUGGINGFACE_HUB_TOKEN=...
```

## Quick start

Re-run the checked-in PAnDa preset:

```bash
./scripts/run_panda_truthfulqa.py
```

Run the same preset on the shared 3B backbone used by the isolated experiments:

```bash
./scripts/run_panda_truthfulqa.py \
  --model-name Qwen/Qwen2.5-3B-Instruct
```

That wrapper:

- apply `--comparison-preset panda`
- run in `--mode sanity`
- save outputs under `results/dev/...`
- work from any current directory because they resolve the repo root themselves
- accept the normal evaluator CLI flags after the preset defaults

If the target Python environment is missing dependencies, bootstrap once from the script itself:

```bash
./scripts/run_panda_truthfulqa.py --bootstrap
```

If the executable bit is missing on your machine, enable it once:

```bash
chmod +x scripts/run_panda_truthfulqa.py
```

If you do not override `--model-name`, the public PAnDa preset switches from the generic CLI default to the saved comparison model:

```text
HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration
```

## Direct CLI usage

After installing the package, you can run the evaluator directly:

```bash
./.venv/bin/python -m panda \
  --comparison-preset panda \
  --mode sanity \
  --save-results \
  --results-dir results/dev/panda_truthfulqa_sanity10
```

Example: run a larger TruthfulQA subset:

```bash
./.venv/bin/python -m panda \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --mode subset \
  --save-results \
  --results-dir results/dev/truthfulqa_subset_run
```

Useful flags:

- `--comparison-preset panda` to apply the public PAnDa preset
- `--model-name ...` to choose a different Hugging Face causal LM
- `--local-files-only` to avoid network fetches
- `--truthfulqa-limit 50` to override the seeded sample size

The fixed-alpha DoLa comparison set is hardcoded to `0.1`, `0.5`, and `0.95`. PAnDa uses the `0.1` and `0.95` regimes internally.

## Benchmarks and outputs

The current public CLI evaluates `TruthfulQA` multiple choice and reports:

- `mc1`
- `mc2`
- `mc3`

Fresh runs written by the CLI use a decoder-specific artifact prefix. For example, the PAnDa preset writes:

- `panda_full_eval_raw_predictions.csv`
- `panda_full_eval_summary.csv`
- `panda_full_eval_pairwise_summary.csv`
- `panda_full_eval_metadata.json`

The checked-in historical artifacts under `results/dev/stage*/` use the older unprefixed naming:

- `raw_predictions.csv`
- `summary.csv`
- `pairwise_summary.csv`
- `metadata.json`

## Plotting results

Plot an overview from a freshly generated PAnDa summary:

```bash
./.venv/bin/python scripts/plot_results.py \
  results/dev/panda_truthfulqa_sanity10/panda_full_eval_summary.csv \
  --output results/dev/panda_truthfulqa_sanity10/overview.png \
  --title "PAnDa TruthfulQA sanity run"
```

Plot the checked-in historical artifact instead:

```bash
./.venv/bin/python scripts/plot_results.py \
  results/dev/stage12_panda_truthfulqa_sanity10/summary.csv \
  --output results/dev/stage12_panda_truthfulqa_sanity10/overview.png \
  --title "Historical Stage 12 PAnDa artifact"
```

## Repository layout

```text
PAnDa/
  paper/          Current paper draft and bibliography
  results/        Checked-in development artifacts and new run outputs
  scripts/        Preset runners and plotting helper
  src/panda/      Evaluator, benchmark loaders, CLI, and decoder wrappers
```

Key implementation files:

- `src/panda/evaluator.py`: model loading and decoder logic
- `src/panda/evaluation.py`: benchmark scoring, summaries, and pairwise comparison helpers
- `src/panda/entrypoint.py`: command-line driver
- `src/panda/benchmarks.py`: dataset loaders
- `src/panda/args.py`: CLI flags and public preset mapping

## Reproducibility notes

- The evaluator reads `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` for remote Hugging Face access.
- `--mode sanity` uses small seeded subsets; `--mode subset` uses larger seeded subsets; `--mode full` removes the default limit.
- The repo tracks compact artifacts and provenance metadata, not every exploratory run or cache file from earlier internal work.
- Exact reproduction can still depend on model availability, Hugging Face authentication, local dataset copies, and hardware comparable to the original environment.

## Paper

Build the paper from `paper/` with:

```bash
cd paper
latexmk -pdf panda_note.tex
```

Further paper notes live in [`paper/README.md`](paper/README.md).
