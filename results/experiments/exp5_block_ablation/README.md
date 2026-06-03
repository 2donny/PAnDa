# Experiment 5: Block-Refined Always Contrast

This repurposed `exp5` asks the cleaner follow-up question after the oracle
diagnosis:

- does block refinement still help if we stop relying on greedy arbitration?
- the default speculative block window for this experiment is now `2`

Compared decoders:

- `always_contrast`
  - single-token stateful fixed contrast endpoint
  - always uses `final_logits - shallow_logits`

- `panda_switch`
  - original speculative block refinement with greedy/contrast arbitration

- `panda_switch_update4`
  - keeps the same block refinement and greedy/contrast arbitration as `panda_switch`
  - but refreshes one shared JSD-selected shallow layer only every `4` scored tokens
  - reuses that same shallow layer across every speculative position inside the block

- `panda_always_contrasts`
  - uses the same Jacobi block refinement scaffold as `panda_switch`
  - selects the best shallow layer per speculative position
  - but always decodes from the contrast view at every position
  - does not use greedy as the final decision rule inside the block

Interpretation:

- if `panda_always_contrasts` beats `always_contrast`, then block
  refinement still adds value beyond plain fixed contrast
- if `panda_switch_update4` stays competitive with `panda_switch`, then
  per-position layer reselection may be unnecessary inside the block
- if it also beats `panda_switch`, then the old greedy arbitration inside the block was
  probably unnecessary or harmful
- if it loses to `panda_switch`, then the block may still benefit from the original
  mixed-view arbitration rule

Run example:

```bash
./.venv/bin/python results/experiments/exp5_block_ablation/run_experiment.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --local-files-only \
  --mode subset \
  --truthfulqa-limit 50 \
  --run-id run_02_update4_default
```
