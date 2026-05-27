# Standard DoLa

## What It Does

`dola` implements the repo's standard DoLa-style decoder:

1. run a forward pass with hidden states
2. score candidate shallow layers by disagreement with the final layer
3. choose the shallow layer with the largest JSD
4. build contrast scores from mature vs shallow log-probabilities
5. optionally apply the `relative_top` mask
6. choose the next token from those contrast scores

Relevant code:

- `src/panda/evaluator.py:599-640`
- `src/panda/evaluator.py:659-668`
- `src/panda/evaluator.py:731-783`

## Generation Pseudocode

```text
# Tokenize and prepare the prompt first.
generated = prepare_prompt(prompt)

repeat until max_new_tokens or EOS:
    # Run a forward pass that returns both final logits and
    # earlier-layer logits for the same current prefix.
    layer_logits, final_logits = forward_with_layer_logits(generated)

    # Search the allowed shallow layers and keep the one that
    # disagrees most with the final layer under JSD.
    best_layer = argmax over shallow_bucket of JSD(final_logits, layer_logits[layer])
    premature_logits = layer_logits[best_layer]

    # Convert both views into log-probability form.
    mature_log_probs = log_softmax(final_logits)
    premature_log_probs = log_softmax(premature_logits)

    # Standard DoLa contrast: boost tokens favored by the final layer
    # relative to the selected shallow layer.
    contrast_scores = log_softmax(mature_log_probs - premature_log_probs)

    if dola_relative_top > 0:
        # Mask very implausible tail tokens using the mature-layer view.
        mask tokens outside relative_top filter
        set masked scores to dola_relative_top_value

    # Choose the next token from the DoLa contrast scores.
    next_token = argmax(contrast_scores)

    # Extend the running prefix and continue.
    append next_token to generated

return decoded continuation
```

## Candidate-Scoring Pseudocode

The repo uses the same DoLa step rule when it scores a fixed candidate:

```text
# Start from the prompt tokens.
generated = prepare_prompt(prompt)
total_logprob = 0

for token in tokenize(choice_text):
    # Recompute the DoLa contrast scores for the current prefix.
    contrast_scores = dola_step(generated)

    # In this implementation, DoLa already returns log-prob-style scores,
    # so the scorer adds the chosen token's score directly.
    total_logprob += contrast_scores[token]

    # Teacher-force the candidate token into the prefix.
    append token to generated

return total_logprob
```

Because `contrast_scores` are already log-prob style values here, the scorer adds them directly.

## Hyperparameters Used Here

| Setting | Value in this repo | Notes |
| --- | --- | --- |
| Shallow bucket | dynamic, default `range(0, max(1, num_layers // 4), 2)` | early even-numbered layers |
| Mature layer | final hidden layer | `num_layers - 1` |
| Layer-selection rule | max JSD between final and candidate shallow layer | recomputed every decoding step |
| `dola_relative_top` | `0.1` | default CLI value |
| `dola_relative_top_value` | `-1000.0` | masked token score |

## Important Repo-Specific Note

This public repo's `dola` path does **not** use an explicit adaptive `alpha_t` parameter.

The contrast rule is:

```text
contrast_scores = log_softmax(log p_final - log p_shallow)
```

plus `relative_top` masking.
