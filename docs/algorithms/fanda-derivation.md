# FAnDa: Derivation and Provable Claims

This note is about the experiment-local `fanda` family, not the public `src/panda` decoder preset.

In this repo, `fanda` means:

- raw-logit contrast
- no relative-top filter
- one shallow layer chosen by max-JSD
- then held for a chosen refresh schedule

Relevant code:

- `results/experiments/local_evaluator.py:68-82`
- `results/experiments/local_evaluator.py:730-767`
- `results/experiments/local_evaluator.py:1890-1967`
- `src/panda/evaluation.py:129-171`

## Scope

This note is intentionally narrower than a paper-style claim like:

```text
FAnDa is more factual than DoLa.
```

That broad statement is not something this codebase can prove by derivation alone.

What we *can* prove cleanly is:

- when anchor persistence leaves the chosen token unchanged
- when anchor persistence leaves multiple-choice candidate rankings unchanged
- why an interior refresh interval can be better than both `update1` and `frozen`

So the mathematically defensible claim is a stability-and-tradeoff claim, not a universal factuality claim.

## Setup

At decoding step `t`, let:

- `V` = vocabulary
- `f_t(v)` = final-layer logit for token `v`
- `s_t(l, v)` = shallow-layer logit for token `v` at candidate layer `l`
- `p_t = softmax(f_t / tau)`
- `q_t(l) = softmax(s_t(l) / tau)`

The repo's step-local layer rule is:

```text
l*_t = argmax over l in shallow_bucket of JSD(p_t, q_t(l))
```

The raw-logit contrast score attached to layer `l` is:

```text
g_t(l, v) = f_t(v) - s_t(l, v)
```

The next-token rule is:

```text
v_t(l) = argmax over v in V of g_t(l, v)
```

For a refresh interval `k`:

- `k = 1` means `update1`
- `k = 2` means `update2`
- `k = 4` means `update4`
- `k = infinity` means the repo's practical `fanda_frozen` behavior
  - one refresh at step `0`, then hold the layer for the rest of the answer

Let `l^k_t` be the layer actually carried by the schedule with interval `k`.

Then the carried decoder uses:

```text
g^k_t(v) = g_t(l^k_t, v)
```

while the step-local oracle version uses:

```text
g*_t(v) = g_t(l*_t, v)
```

Define the anchor-error vector:

```text
e^k_t(v) = g^k_t(v) - g*_t(v) = s_t(l*_t, v) - s_t(l^k_t, v)
```

So the entire effect of stale persistence enters through `e^k_t`.

## Result 1: Exact Score Perturbation Identity

The carried schedule and the step-local oracle differ by:

```text
g^k_t = g*_t + e^k_t
```

with

```text
||g^k_t - g*_t||_inf = ||e^k_t||_inf
```

This is immediate from the definition above, but it is the key reduction:

- the final-layer logits cancel
- only the shallow-layer mismatch matters

That means any theorem about FAnDa persistence reduces to bounding the shallow-anchor error.

## Result 2: Token Invariance Under Bounded Anchor Error

Let the oracle top token at step `t` be:

```text
a*_t = argmax over v of g*_t(v)
```

Assume the top token is unique, and define its oracle margin:

```text
m_t = g*_t(a*_t) - max over v != a*_t of g*_t(v)
```

### Lemma

If

```text
2 ||e^k_t||_inf < m_t
```

then the carried schedule chooses the same token:

```text
argmax_v g^k_t(v) = a*_t
```

### Proof

For the oracle winner:

```text
g^k_t(a*_t) >= g*_t(a*_t) - ||e^k_t||_inf
```

For any competing token `v != a*_t`:

```text
g^k_t(v) <= g*_t(v) + ||e^k_t||_inf
```

Therefore:

```text
g^k_t(a*_t) - g^k_t(v)
>= g*_t(a*_t) - g*_t(v) - 2 ||e^k_t||_inf
>= m_t - 2 ||e^k_t||_inf
> 0
```

So every competitor stays below the oracle winner, and the argmax is unchanged.

### Interpretation

This is the cleanest exact statement in the whole theory:

- persistence is harmless whenever the carried layer perturbs scores by less than half the oracle margin
- token flips happen only after stale-anchor error becomes comparable to the local decision margin

That matches the exp12 intuition that persistence helps until it becomes stale enough to change rankings.

## Result 3: Sequence-Score Stability For MC Evaluation

The repo's multiple-choice evaluator scores a fixed candidate answer token by token under teacher forcing, then ranks candidates by total sequence log-probability:

```text
S_k(y_1:T) = sum over t of log_softmax(g^k_t(prefix = y_<t))[y_t]
```

This is exactly the structure used in:

- `results/experiments/local_evaluator.py:730-767`
- `src/panda/evaluation.py:129-171`

### A useful inequality

For any two score vectors `x` and `z`, and any token index `i`:

```text
|log_softmax(x)_i - log_softmax(z)_i| <= 2 ||x - z||_inf
```

Reason:

```text
log_softmax(x)_i = x_i - logsumexp(x)
```

and both terms are `1`-Lipschitz under `||.||_inf`, so the difference is at most `2 ||x - z||_inf`.

### Proposition

For any fixed candidate `y_1:T`:

```text
|S_k(y_1:T) - S_*(y_1:T)|
<= 2 sum over t of E^k_t(y_<t)
```

where

```text
E^k_t(prefix) = ||e^k_t(prefix)||_inf
```

and `S_*` is the step-local oracle score.

### Proof sketch

Apply the per-token Lipschitz bound at each teacher-forced step, then sum over `t`.

### Ranking corollary

For two candidates `y` and `z`, define the oracle margin:

```text
Gamma_*(y, z) = S_*(y) - S_*(z)
```

If

```text
Gamma_*(y, z)
> 2 sum_t E^k_t(y_<t) + 2 sum_t E^k_t(z_<t)
```

then the carried schedule preserves the ranking:

```text
S_k(y) > S_k(z)
```

### Why this matters here

This is the most practical theorem for this repo.

It does **not** prove full-sequence factuality in open-ended generation.
But it *does* speak directly to TruthfulQA MC, because MC1/MC2/MC3 are derived from candidate sequence scores after teacher-forced evaluation.

So if your theory target is:

```text
selected-layer persistence can preserve or improve MC ranking stability
```

then yes, a mathematical derivation is realistic.

## Result 4: Why An Interior Refresh Interval Can Be Optimal

The theorem above tells us when a given schedule is safe.
It does not yet explain why `update4` can beat both `update1` and `frozen`.

For that we need one extra modeling step.

## Assumptions

Assume two effects contribute to total decoding disturbance:

### 1. Refresh noise

Every time we refresh the layer selector, we give the decoder another chance to change the scoring function for reasons unrelated to genuine context change.

Model that normalized cost as:

```text
noise(k) = A / k
```

because a run of length `T` has roughly `T / k` refresh opportunities.

### 2. Staleness

Between refreshes, the carried layer drifts away from the current step-local best layer.

If that drift grows roughly linearly with time since refresh, then the average within-interval error scales like:

```text
staleness(k) = B k
```

for some problem-dependent constant `B > 0`.

### Combined perturbation model

Then the total normalized risk is bounded by:

```text
R(k) <= A / k + B k
```

### Minimizer

Differentiate the right-hand side:

```text
d/dk (A / k + B k) = -A / k^2 + B
```

Setting this to `0` gives:

```text
k* = sqrt(A / B)
```

So the model predicts:

- `update1` is too jittery when `A` is large
- `frozen` is too stale when `B` is large
- an intermediate schedule wins when both effects are present

This is exactly the kind of claim the exp12 README already frames:

- `update1` = fresh but jittery
- `frozen` = stable but stale
- `update4` = compromise point

See:

- `results/experiments/exp12_state_persistence_diagnostics/README.md`

## A more specific persistence reading

If your verbal theory is:

```text
hallucination-prone local modes persist for a few nearby tokens,
so reselecting the layer every token is often unnecessary
```

then the math above supports that story in the following way:

- if nearby steps share the same useful correction layer, then `l*_t` changes slowly
- when `l*_t` changes slowly, `E^k_t` stays small for moderate `k`
- when `E^k_t` stays below local margins, token choices stay unchanged
- therefore a moderate `k` can remove selector thrash without damaging token ranking

That is the clean derivation target for FAnDa.

## How This Maps To `exp12`

The exp12 mechanism metrics line up naturally with the derivation:

- `switch_rate`
  - practical proxy for the refresh-noise term

- `selected_layer_match_rate`
  - practical proxy for how often `l^k_t` still equals `l*_t`

- `avg_oracle_jsd_gap`
  - practical proxy for stale-anchor error

In the stored matched run, the pattern is:

- `update1`
  - `mc2 = 0.5344`
  - `switch_rate = 0.6920`
  - `selected_layer_match_rate = 1.0000`
  - `avg_oracle_jsd_gap = 0.0000`

- `update4`
  - `mc2 = 0.5758`
  - `switch_rate = 0.1353`
  - `selected_layer_match_rate = 0.4612`
  - `avg_oracle_jsd_gap = 0.0025`

- `frozen`
  - `mc2 = 0.5602`
  - `switch_rate = 0.0000`
  - `selected_layer_match_rate = 0.2584`
  - `avg_oracle_jsd_gap = 0.0034`

Source:

- `results/experiments/exp12_state_persistence_diagnostics/runs/run_01_default/run_01_default_summary.csv`

This does not prove the theory by itself, but it is consistent with the derived tradeoff:

- `update1` pays high jitter cost
- `frozen` pays higher staleness cost
- `update4` sits between them and wins on this matched MC run

## What The Derivation Does *Not* Prove

It does **not** prove:

- that `fanda_frozen` is always better than `update1`
- that the best schedule is universally `4`
- that JSD-best shallow layers are causally tied to truth
- that open-ended factuality must improve on arbitrary prompts

Those require empirical evidence.

In particular, open-ended generation adds feedback effects:

- a token change alters the future prefix
- the future layer-selection problem changes
- the theorem's clean teacher-forced MC bounds no longer give a full factuality guarantee

So for open-ended claims, the theory should be presented as a mechanism explanation plus empirical support, not as a full proof.

## Recommended Claim Language

If this becomes paper or slide material, the safest theorem-shaped claim is:

```text
Under bounded stale-anchor error, persistent-layer contrast preserves token decisions
and multiple-choice candidate rankings; under a noise-vs-staleness tradeoff,
the optimal refresh interval is generically interior rather than at either extreme.
```

That claim is:

- mathematically derivable
- closely tied to the repo's actual scoring rule
- compatible with the exp12 mechanism metrics
- honest about what remains empirical

## Bottom Line

If your theory is:

```text
short-horizon selected-layer persistence can outperform per-token reselection
because it reduces selector thrash before staleness becomes too large
```

then yes, this repo supports a real mathematical derivation.

If your theory is:

```text
frozen FAnDa is universally more factual than the alternatives
```

then no, that is still an empirical claim.
