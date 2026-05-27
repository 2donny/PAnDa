# Greedy Decoder

## What It Does

`greedy` is the simplest decoder ihRelevant code:

- `src/panda/evaluator.py:659-662`
- `src/panda/evaluator.py:731-783`
- `src/panda/evaluator.py:793-873`

## Generation Pseudocode

```text
# Start from the tokenized prompt / current prefix.
generated = prepare_prompt(prompt)

repeat until max_new_tokens or EOS:
    # Run one forward pass and look only at the last position,
    # which is the model's score for the next token.
    final_logits = model(generated).logits[-1]

    # Greedy decoding always picks the highest-scoring token.
    next_token = argmax(final_logits)

    # Extend the prefix with that chosen token.
    append next_token to generated

# Decode only the continuation part, not the original prompt.
return decoded continuation
```

## Candidate-Scoring Pseudocode

For multiple-choice style evaluation, the repo scores a fixed candidate answer token by token:

```text
# Start from the prompt tokens.
generated = prepare_prompt(prompt)
total_logprob = 0

for token in tokenize(choice_text):
    # Force the model to score the next token under the current prefix.
    final_logits = model(generated).logits[-1]

    # Convert raw scores into log-probabilities so we can accumulate
    # a full sequence score for this fixed candidate answer.
    logprobs = log_softmax(final_logits)

    # Add the score of the ground-truth candidate token.
    total_logprob += logprobs[token]

    # Teacher-force the candidate token into the prefix.
    append token to generated

# The final number is the sequence log-probability of this answer.
return total_logprob
```

## Hyperparameters Used Here

Greedy has no decoder-specific contrast hyperparameters.

Shared runtime behavior still applies:

| Setting            | Value in this repo                                       |
| ------------------ | -------------------------------------------------------- |
| Step rule          | `argmax(final_logits)`                                 |
| EOS handling       | stop when EOS is produced if `stop_on_eos=True`        |
| Generation default | `max_new_tokens=96` inside `generate_with_decoder()` |
| Prompt handling    | `prepare_prompt()` with chat template when available   |

## What It Is Not

- No shallow-layer contrast
- No dynamic layer selection
- No pairwise reranking
- No block-local arbitration
