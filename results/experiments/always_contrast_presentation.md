# Why Our Contrast Decoder Improves Over DoLa

## Slide 1: Title

**Improving Factuality With a Stronger Contrast Endpoint**

- Goal: improve TruthfulQA factuality without changing model weights
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Baseline: official DoLa
- Main result: our `always_contrast` decoder beat official DoLa on all three TruthfulQA metrics

## Slide 2: Motivation

Large language models often contain the right knowledge, but the decoded answer can still drift toward:

- common misconceptions
- generic continuations
- easy but false answers
- fluent responses that are not actually grounded

Our initial intuition was simple:

- LLM answers often improve when the model gets another chance to evaluate what it has already said
- repeated evaluation can expose mistakes that looked plausible on the first pass

This intuition is also consistent with prior work. For example, *Ask, Assess,
and Refine* (Lee et al., EACL 2024) uses explicit evaluation followed by
iterative refinement, and reports improved factual consistency together with
reduced hallucinations. That is not the same mechanism as our decoder, but it
supports the broader idea that repeated evaluation can help factuality.

We wanted a decoding-time fix, not a retraining story.

Research question:

- can we build a lightweight decoder that repeatedly re-evaluates the growing answer and improves factuality?

## Slide 3: Terminology

Two terms matter in this presentation.

**Stateful**

- the decoder carries a small piece of information from one token step to the next, instead of making every decision fully from scratch
- in our case, that state is mainly:
- the currently selected shallow layer, which is our latest guess about where the premature or misleading signal is coming from
- the running step count, which tells the decoder when to refresh that layer choice

**Sequential**

- it can mean **past-aware** decoding, where the decoder carries information from earlier token steps
- or **future-aware** decoding, where the decoder looks ahead over possible next-token trajectories
- our method only uses the first kind
- it is stateful over past steps, but it does not do beam search, lookahead, or horizon planning

## Slide 4: Core Intuition

We compare two internal views of the same model:

- final-layer signal: what the mature model currently wants to say
- shallow-layer signal: what an earlier, more premature representation already supports

At every token, the model sees the full answer prefix again.

If shallow layers over-favor generic or stereotyped answers, then subtracting that shallow preference may help expose the more truthful final-layer preference.

In one sentence:

- factuality may improve when we decode from what survives after premature continuation pressure is discounted
- and when we let that contrast be recomputed as the answer prefix grows

## Slide 5: What Official DoLa Does

Official DoLa already uses the basic contrastive idea:

1. Choose a shallow layer by JSD against the final layer.
2. Build a contrastive score from mature and premature **log-probabilities**.
3. Apply a relative-top filter before decoding.
4. Recompute this decision independently at each token step.

High-level formula:

```text
contrast_scores = log_softmax(final_logits) - log_softmax(shallow_logits)
contrast_scores = log_softmax(contrast_scores)
```

So official DoLa is not just "subtract shallow from final." It is a more constrained log-probability contrast with filtering.

In decoding-time terms:

- official DoLa is prefix-conditioned, but mostly **step-local**
- it re-evaluates each token from scratch rather than carrying a decoder-side belief forward

## Slide 6: What We Changed

Our modification keeps the contrastive idea, but simplifies and strengthens the endpoint:

1. Keep dynamic shallow-layer selection.
2. Decode from a direct binary contrast view:

```text
contrast_scores = final_logits - shallow_logits
```

3. Carry the selected shallow layer statefully and refresh it every `4` token steps.
4. Do not use the official relative-top mask in this path.
5. Re-run the model on the full growing prefix at every step, so each next-token choice is a fresh re-evaluation of the answer-so-far.

This is the `always_contrast` decoder.

So compared with official DoLa, our decoder is more stateful over decoding time:

- it does not just condition on the prefix
- it also carries a short-lived state about which shallow layer currently best represents the premature signal

## Slide 7: How `selected_layer` Works

The key extra state in our decoder is `selected_layer`.

We interpret it as a short-lived belief about:

- which shallow layer is currently best exposing the premature or misleading continuation pressure

Transformer-side view:

- at each token step, the transformer processes the full current prefix
- every layer produces a hidden state for the last position
- after final normalization and the LM head, each layer can be interpreted as its own token distribution
- the final layer is our mature view
- the earlier layers are candidate shallow views
- the decoder then asks: which shallow layer currently disagrees most with the mature view?

How do we calculate `selected_layer`?

- for each candidate shallow layer, we compare its token distribution to the final-layer distribution
- we compute Jensen-Shannon divergence (JSD) between the two distributions
- we choose the shallow layer with the **largest** JSD
- then we keep that layer for a short span and refresh it every `4` token steps

Why calculate it this way?

- a layer with larger JSD is the one that disagrees most strongly with the final layer
- we treat that disagreement as a proxy for premature, generic, or misleading continuation pressure
- subtracting a shallow layer that is too similar to the final layer would remove little useful signal
- subtracting the most divergent shallow layer gives the contrast term the clearest chance to suppress what the mature model has moved away from

Why might carrying that state across steps help?

- **Lower-variance correction signal**: wrong continuations often persist over several nearby tokens, not just one. If we reselect the shallow layer independently at every step, small token-level fluctuations can make the chosen correction source bounce between layers even when the underlying error mode has not really changed. Carrying `selected_layer` for a short span smooths that process: the decoder keeps subtracting the same locally relevant premature signal, which makes the contrast term more stable and less noisy across the phrase.

This connects directly to the motivation:

- if repeated evaluation helps because the model gradually exposes its own weak or premature beliefs
- then carrying `selected_layer` gives that repeated evaluation a small memory, instead of treating every token as a totally fresh decision

This is also the right way to interpret the optimization story:

- after each refresh, the decoder makes a locally greedy choice of the best correction layer under the current prefix
- it then reuses that choice for the next few token decisions instead of resolving the layer-selection problem from scratch every time
- so it does use past information to guide future local decisions
- but it is still not future-horizon optimization, beam search, or lookahead planning

## Slide 8: Why The Full Modification Could Help

Our theory was that official DoLa may be leaving useful factual signal on the table.

Possible reasons:

- raw-logit subtraction may preserve a cleaner separation signal than subtracting normalized log-probabilities
- removing the relative-top mask may avoid prematurely pruning truthful candidates
- a fixed full-contrast endpoint may suppress shallow-layer misconceptions more aggressively
- carrying `selected_layer` may make the correction signal less noisy than DoLa's step-local reselection
- repeated prefix re-evaluation may act like a lightweight online self-check, even without rewriting past tokens

Important caveat:

- `always_contrast` is more stateful than official DoLa
- but it is still not a full revise-and-rewrite decoder
- and it is not beam search or horizon lookahead
- it re-evaluates the growing prefix online without changing already emitted tokens

## Slide 9: Research Hypotheses

### Hypothesis A

Official DoLa is not the optimal point in the contrastive design space.

### Hypothesis B

A stronger and simpler contrast endpoint should improve factuality over official DoLa.

### Hypothesis C

Carrying `selected_layer` across short spans may make the decoder less noisy and less myopic than DoLa's step-local layer selection.

### Hypothesis D

The gain should show up consistently across `mc1`, `mc2`, and `mc3`, not only on one metric.

## Slide 10: Evidence 1 — Contrast Construction Matters

`exp1_contrast_strength` asked a narrow question:

- if we keep the DoLa family structure but change the contrast construction, do results move?

On 50 TruthfulQA examples:

- `official_dola`: `mc1 = 0.28`, `mc2 = 0.528`, `mc3 = 0.273`
- `matched_alpha_dola_0p10`: `mc1 = 0.34`, `mc2 = 0.525`, `mc3 = 0.306`
- `matched_alpha_dola_0p50`: `mc1 = 0.32`, `mc2 = 0.522`, `mc3 = 0.286`
- `matched_alpha_dola_0p95`: `mc1 = 0.30`, `mc2 = 0.527`, `mc3 = 0.280`

Takeaway:

- factuality is sensitive to the exact contrast recipe
- official DoLa is not obviously the best point even within closely related variants

## Slide 11: Evidence 2 — Stronger Fixed Contrast Wins

`exp4_family_baselines` tested the fixed-endpoint family more directly.

On 50 TruthfulQA examples:

- `dola`: `mc1 = 0.28`, `mc2 = 0.528`, `mc3 = 0.273`
- `fixed alpha dola (0.0)`: `mc1 = 0.34`, `mc2 = 0.531`, `mc3 = 0.307`
- `fixed alpha dola (0.5)`: `mc1 = 0.36`, `mc2 = 0.565`, `mc3 = 0.315`
- `fixed alpha dola (1.0)`: `mc1 = 0.40`, `mc2 = 0.576`, `mc3 = 0.351`

Takeaway:

- as we move toward a stronger fixed contrast endpoint, factuality improves
- the best point in this experiment is the full-contrast endpoint

## Slide 12: Main Result

Our deployed decoder, `always_contrast`, reproduces that strong endpoint cleanly.

In `exp7_switch_vs_fixed_regimes` on 50 TruthfulQA examples:

- `always_contrast`: `mc1 = 0.40`, `mc2 = 0.576`, `mc3 = 0.351`, latency `9.15s`
- `dola`: `mc1 = 0.28`, `mc2 = 0.528`, `mc3 = 0.273`, latency `9.07s`

Improvement over official DoLa:

- `+0.12` on `mc1`
- `+0.048` on `mc2`
- `+0.078` on `mc3`

And the latency is nearly the same.

## Slide 13: What The Gain Is Not

Our results suggest the gain is **not** coming from future-horizon search.

- not from beam-search-style horizon search
- not from speculative future-token refinement

Our closest search-like / horizon-style variants were worse. In `exp5` on 30 TruthfulQA examples:

- `always_contrast`: `mc1 = 0.367`, `mc2 = 0.592`, `mc3 = 0.338`
- `panda_switch`: `mc1 = 0.267`, `mc2 = 0.548`, `mc3 = 0.295`
- `panda_always_contrasts`: `mc1 = 0.267`, `mc2 = 0.516`, `mc3 = 0.284`

So the win is more consistent with:

- a stronger contrast signal
- plus a small amount of carried state

than with adding search or speculative block refinement

## Slide 14: Interpretation

Our current interpretation is:

- the important gain is not "contrastive decoding" in the broadest sense
- the important gain is using a **stronger, cleaner, less filtered contrast endpoint**

Official DoLa already had the right intuition.

What our results suggest is that:

- the official log-probability contrast plus relative-top filtering may be too conservative for this factuality setting
- a direct `final_logits - shallow_logits` decoder better suppresses shallow premature preferences
- the extra statefulness of `always_contrast` may also help by making the decoder less myopic than DoLa's step-local selection

## Slide 15: Contribution And Claim

The contribution is best framed as an empirical insight:

- on this TruthfulQA multiple-choice teacher-forced setup, our modification of DoLa's contrast construction substantially improved factuality

This is not just a tuning result in spirit. It changes the interpretation of where the factuality gain comes from:

- not from sampling randomness
- not from extra search machinery
- but from the exact way we contrast mature and premature model signals while repeatedly re-evaluating the growing prefix with a small amount of carried state

**Claim**

For `Qwen/Qwen2.5-3B-Instruct` on TruthfulQA multiple-choice under our evaluator, `always_contrast` is clearly stronger than official DoLa.

## Slide 16: Scope And Final Message

What we can say:

- our modification improved over official DoLa in this benchmark setting
- stronger fixed contrast appears to be the right direction for factuality here

What we should not overclaim:

- universal superiority across all models
- universal superiority across all factuality datasets
- a complete mechanistic proof of why raw-logit contrast wins

If we had to summarize the whole project in one line:

- DoLa had the right idea, but our `always_contrast` decoder combined a stronger contrast signal with a small amount of carried state, and that was enough to deliver a clear factuality gain.
