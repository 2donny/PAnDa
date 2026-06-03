# Fixed-Alpha DoLa: Worked Example

This is a companion note to [fixed-alpha-dola.md](fixed-alpha-dola.md).

The main decoder file explains the repo-accurate algorithm. This note explains the same decoder more intuitively with a single-step numeric example.

## Core Idea

Fixed-alpha DoLa in this repo does:

```text
new_scores = final_logits - alpha * shallow_logits
```

So it takes:

- the model's final-layer next-token scores
- the selected shallow-layer next-token scores

and subtracts a scaled version of the shallow scores from the final scores.

## Why This Exists

The shallow layer can push some tokens strongly even when the final layer is more refined.

Fixed-alpha DoLa says:

- keep the final layer as the main signal
- suppress shallow influence by a chosen amount `alpha`

If `alpha` is:

- smaller: weaker correction
- larger: stronger correction

## One-Step Example

Suppose the prompt is:

```text
What is the capital of Australia?
```

At the current next-token position, assume the selected shallow layer and the final layer give these raw logits:

| Token | Final logit | Shallow logit |
| --- | ---: | ---: |
| `Canberra` | 5.0 | 0.5 |
| `Sydney` | 5.2 | 2.0 |
| `Melbourne` | 2.0 | 0.7 |

Assume:

```text
alpha = 0.5
```

Then the fixed-alpha rule gives:

| Token | Calculation | New score |
| --- | --- | ---: |
| `Canberra` | `5.0 - 0.5 * 0.5` | `4.75` |
| `Sydney` | `5.2 - 0.5 * 2.0` | `4.20` |
| `Melbourne` | `2.0 - 0.5 * 0.7` | `1.65` |

So the decoder changes from:

- final-layer top token: `Sydney` (`5.2`)

to:

- fixed-alpha top token: `Canberra` (`4.75`)

This is the whole mechanism:

- `Sydney` started slightly ahead
- but its shallow-layer support was much stronger
- subtracting `alpha * shallow_logits` penalized it more
- `Canberra` became the new top token

## Step-by-Step Generation Logic

```text
1. Start from the current generated prefix.
2. Run a forward pass and collect:
   - final logits
   - candidate shallow-layer logits
3. Choose one shallow layer by maximum JSD disagreement.
4. Compute:
   new_scores = final_logits - alpha * shallow_logits
5. Pick the argmax token from new_scores.
6. Append that token to the prefix.
7. Repeat.
```

## What Is Dynamic and What Is Fixed

### Dynamic

- the current prefix
- the selected shallow layer
- the final logits
- the shallow logits

### Fixed

- the formula
- the chosen `alpha` value for this decoder run

That is why it is called **fixed-alpha** DoLa:

- the shallow layer can still change
- but the contrast strength does not change token by token

## Repo-Specific Hyperparameters

These are the relevant current repo settings:

| Setting | Value |
| --- | --- |
| Fixed-alpha presets | `0.0`, `0.5`, `1.0` |
| `alpha_min` | `0.0` |
| `alpha_max` | `1.0` |
| `tau` | `0.5` |
| `update_every` | `4` |
| Default shallow bucket | `range(0, max(1, num_layers // 4), 2)` |

See:

- `src/panda/args.py`
- `src/panda/config.py`
- `src/panda/evaluator.py`

## Difference From Standard DoLa

Standard `dola` in this repo uses:

```text
contrast_scores = log_softmax(log p_final - log p_shallow)
```

plus `relative_top` masking.

Fixed-alpha DoLa instead uses:

```text
new_scores = final_logits - alpha * shallow_logits
```

So the main difference is:

- standard DoLa: indirect normalized contrast rule
- fixed-alpha DoLa: direct tunable shallow-logit penalty

## When To Use This Note

Use [fixed-alpha-dola.md](fixed-alpha-dola.md) when you want the repo-accurate pseudocode.

Use this file when you want:

- the intuition
- one numeric example
- a simpler explanation for slides or speaking
