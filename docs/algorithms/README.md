# Decoder Algorithms in This Repo

This folder explains the current decoder implementations in `src/panda/` as they exist in the public repo.

Files:

- [greedy.md](greedy.md)
- [dola.md](dola.md)
- [fixed-alpha-dola.md](fixed-alpha-dola.md)
- [panda.md](panda.md)

## Shared Runtime

All decoders in this repo use the same local Hugging Face causal LM loaded by `Stage4Evaluator`:

- tokenizer load: `src/panda/evaluator.py`
- model load: `src/panda/evaluator.py`

The public evaluator exposes these decoder names:

- `greedy`
- `dola`
- `fixed_alpha_dola_low`
- `fixed_alpha_dola`
- `fixed_alpha_dola_high`
- `panda`

See `src/panda/utils.py`.

## Repo-Wide Decoder Defaults

These are the current code defaults, not necessarily the settings used by every historical artifact under `results/dev/stage*/`.

| Setting | Current repo default | Where it comes from |
| --- | --- | --- |
| CLI model default | `Qwen/Qwen2.5-3B-Instruct` | `src/panda/args.py` |
| Public preset model override | `HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration` | `src/panda/args.py` |
| Comparison presets | `panda` | `src/panda/args.py` |
| Default decoder set | `greedy`, `dola`, `fixed_alpha_dola_low`, `fixed_alpha_dola`, `fixed_alpha_dola_high`, `panda` | `src/panda/utils.py` |
| Default shallow bucket | `range(0, max(1, num_layers // 4), 2)` | `src/panda/evaluator.py` |
| `tau` | `0.5` | `src/panda/config.py` |
| `alpha_min` | `0.1` | `src/panda/config.py` |
| `alpha_max` | `0.95` | `src/panda/config.py` |
| `update_every` | `4` | `src/panda/config.py` |
| `dola_relative_top` | `0.1` | `src/panda/args.py` |
| `dola_relative_top_value` | `-1000.0` | `src/panda/args.py` |
| Fixed-alpha DoLa presets | `0.1`, `0.5`, `0.95` | `src/panda/config.py` |
| PAnDa low/high regimes | `0.1`, `0.95` | `src/panda/config.py` |
| `jacobi_window_size` | `4` | `src/panda/args.py` |
| `jacobi_max_iters` | `2` | `src/panda/args.py` |
| `panda_divergence_threshold` | `0.05` | `src/panda/args.py` |
| `panda_truth_bias` | `0.02` | `src/panda/args.py` |
| `panda_early_agreement_shortcut` | `False` | `src/panda/args.py` |

## Public Preset Overrides

When you run `--comparison-preset panda`, the repo changes a few defaults if you did not override them manually:

- `model_name`: `Qwen/Qwen2.5-3B-Instruct -> HINT-lab/DeepSeek-R1-Distill-Qwen-1.5B-Self-Calibration`
- `strict_eval = True`

See `src/panda/args.py`.

## Notes

- `dola` in this repo does **not** use an explicit adaptive `alpha_t` controller.
- `panda` is a block-local arbitration method that keeps one evolving trajectory instead of reranking two completed responses.
