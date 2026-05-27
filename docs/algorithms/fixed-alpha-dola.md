# Fixed-Alpha DoLa

## What It Does

`fixed_alpha_dola_low`, `fixed_alpha_dola`, and `fixed_alpha_dola_high` keep DoLa-style dynamic shallow-layer selection, but replace the standard DoLa contrast rule with direct fixed-alpha shallow-logit subtraction:

```text
new_logits = final_logits - alpha * shallow_logits
```

Relevant code:

- `src/panda/evaluator.py:259-323`
- `src/panda/evaluator.py:659-679`
- `src/panda/evaluator.py:731-783`

## Generation Pseudocode

```text
# Initialize the dynamic shallow-layer state once.
state.selected_layer = shallow_bucket[0]
state.step = 0

# Prepare the prompt tokens.
generated = prepare_prompt(prompt)

repeat until max_new_tokens or EOS:
    # Get final-layer logits plus candidate shallow-layer logits.
    layer_logits, final_logits = forward_with_layer_logits(generated)

    # Turn the final logits into probabilities for layer selection.
    p_final = softmax(final_logits / tau)

    if step % update_every == 0:
        # Periodically refresh the shallow layer by choosing the one
        # that disagrees most with the final distribution.
        state.selected_layer = argmax over shallow_bucket of JSD(p_final, softmax(layer_logits[layer] / tau))

    # Read the currently selected shallow layer.
    shallow_logits = layer_logits[state.selected_layer]

    # Pick the fixed alpha attached to this decoder preset and clamp it
    # to the allowed range.
    alpha = clamp(decoder_fixed_alpha, alpha_min, alpha_max)

    # Direct fixed-alpha contrast rule used by this repo.
    scores = final_logits - alpha * shallow_logits

    # Decode greedily from the adjusted scores.
    next_token = argmax(scores)

    # Append the selected token and advance the step counter.
    append next_token to generated
    step += 1

return decoded continuation
```

## Candidate-Scoring Pseudocode

```text
# Reuse the same dynamic-layer state used during generation.
state.selected_layer = shallow_bucket[0]
state.step = 0

# Start from the prompt.
generated = prepare_prompt(prompt)
total_logprob = 0

for token in tokenize(choice_text):
    # Build the fixed-alpha scores for the current prefix.
    scores = fixed_alpha_step(generated, state)

    # Convert those scores into log-probabilities so a whole
    # candidate answer can be scored token by token.
    total_logprob += log_softmax(scores)[token]

    # Teacher-force the current candidate token into the prefix.
    append token to generated

return total_logprob
```

## Hyperparameters Used Here

| Setting | Value in this repo | Notes |
| --- | --- | --- |
| Fixed-alpha presets | `0.1`, `0.5`, `0.95` | repo-defined decoder set |
| `alpha_min` | `0.1` | lower clamp |
| `alpha_max` | `0.95` | upper clamp for the fixed-alpha presets |
| `tau` | `0.5` | used for layer selection and instability terms |
| `update_every` | `4` | dynamic shallow layer is refreshed every 4 steps |
| Shallow bucket | default `range(0, max(1, num_layers // 4), 2)` | early even-numbered layers |

## Extra Trace Quantities Recorded

The repo also logs these diagnostics for this decoder:

- `selected_layer`
- `divergence`
- `margin`
- `instability`
- `alpha`

Those come from `compute_fixed_alpha_step()`.
