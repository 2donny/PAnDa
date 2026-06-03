# PAnDa

## What It Does

`panda` is the repo's block-local arbitration decoder.

It keeps one evolving trajectory and works inside a speculative block:

1. build a greedy view and a contrast-subtracted view for the same block
2. detect the first meaningful local disagreement
3. arbitrate locally from that point onward
4. refine the block with Jacobi-style updates
5. commit the stable prefix and continue

Relevant code:

- `src/panda/evaluator.py:421-575`
- `src/panda/evaluator.py:684-729`
- `src/panda/evaluator.py:793-826`

## Generation Pseudocode

```text
# Start from the prompt tokens.
generated = prepare_prompt(prompt)

repeat until max_new_tokens or EOS:
    # The block can shrink near the end if little token budget remains.
    block_window_size = min(jacobi_window_size, remaining_budget)

    # Build and refine one speculative block under PAnDa.
    block_result = run_panda_block(generated, block_window_size)

    # Commit only the stable prefix of that refined block.
    commit_len = stable-prefix commit from block_result

    # Append the committed tokens and continue from the updated prefix.
    append committed tokens to generated

return decoded continuation
```

## `run_panda_block()` Pseudocode

```text
# Initialize the speculative block by repeating the last generated token.
buffer = repeat last generated token across the whole block
previous_buffer = buffer

for iteration in 0 .. jacobi_max_iters - 1:
    # Run one shared forward pass for the whole current block hypothesis.
    input_ids = generated + buffer
    layer_logits, final_logits = forward_with_window_layer_logits(input_ids, window_size)

    # Choose one shallow layer per block position using maximum JSD.
    selected_layers = choose one shallow layer per position by max JSD

    for each position in the block:
        # Read the shallow view chosen for this position.
        shallow_logits = layer_logits[selected_layer][position]

        # Build the two local views for the same position.
        greedy_scores = final_logits[position]
        contrast_scores = final_logits[position] - shallow_logits

        # Each regime proposes its own top token.
        greedy_token = argmax(greedy_scores)
        contrast_token = argmax(contrast_scores)

        # Only mark a real disagreement if the top tokens differ
        # and the two regime distributions are far enough apart.
        if greedy_token != contrast_token and JSD(greedy_scores, contrast_scores) >= panda_divergence_threshold:
            mark this position as a disagreement

    # Arbitration starts only from the earliest meaningful disagreement.
    first_divergence = earliest marked disagreement position

    for each position in the block:
        # Prefix positions before the first divergence just keep the greedy view.
        arbitration_active = position >= first_divergence

        # After arbitration activates, switch to the contrast-subtracted view only if
        # its top-1 confidence beats the greedy view by the truth-bias rule.
        if arbitration_active and top1_confidence(contrast_scores) > top1_confidence(greedy_scores) - panda_truth_bias:
            choose contrast_scores / contrast_token
        else:
            choose greedy_scores / greedy_token

    # These chosen tokens become the next block proposal.
    new_buffer = chosen block tokens

    if new_buffer == buffer:
        # Stop early if the block stopped changing.
        converged = True
        break

    # Otherwise keep iterating Jacobi-style on the updated block.
    previous_buffer = buffer
    buffer = new_buffer

# After refinement, commit only the stable prefix, not necessarily the whole block.
stable_prefix_len = full window if converged else common_prefix_length(previous_buffer, buffer)
commit_len = stable_prefix_len if stable_prefix_len > 0 else 1

# Return the refined block, first-position scores, and diagnostics.
return chosen block, first-position scores, diagnostics, and commit length
```

## Candidate-Scoring Pseudocode

When the repo scores a fixed candidate answer with `panda`, it does not score the full block at once. It repeatedly uses the block's first-position scores:

```text
# Start from the prompt.
generated = prepare_prompt(prompt)
total_logprob = 0

for token in tokenize(choice_text):
    # Rebuild a PAnDa block for the current prefix.
    block_result = run_panda_block(generated, min(jacobi_window_size, remaining_tokens))

    # Only the first-position scores of the block are used to score
    # the next candidate token under teacher forcing.
    total_logprob += log_softmax(block_result.first_scores)[token]

    # Teacher-force that candidate token into the prefix.
    append token to generated

return total_logprob
```

## Hyperparameters Used Here

### Current Code Defaults

| Setting | Value in this repo |
| --- | --- |
| `jacobi_window_size` | `4` |
| `jacobi_max_iters` | `2` |
| `panda_divergence_threshold` | `0.05` |
| `panda_truth_bias` | `0.02` |
| `panda_early_agreement_shortcut` | `False` |

### Binary Views Used Inside PAnDa

PAnDa uses a fixed binary rule rather than a tuned alpha sweep:

| Setting | Value |
| --- | --- |
| Greedy view | `final_logits` |
| Contrast-subtracted view | `final_logits - shallow_logits` |

### Shared Layer-Selection Settings

| Setting | Value |
| --- | --- |
| Shallow bucket | default `range(0, max(1, num_layers // 4), 2)` |
| Per-position layer rule | max JSD between final logits and candidate shallow layer logits |
| Buffer init | repeat last token across the block |
| Commit fallback | commit at least 1 token |

## Short Summary

PAnDa is best described as:

```text
shared greedy/contrast block views + local disagreement detection + truth-biased arbitration + stable-prefix commit
```
