# What the published methods score, and why none of them clears `setoracle`

After the demand-signal negative the remaining question was whether any
published method uses a signal *outside* what was measured — something other
than attention mass, which perfect knowledge of does not beat recency. Three
were read in full: TriAttention, SnapKV, R-KV. None clears the bar, and the
survey turned up a shared blind spot that is more useful than any of them.

## `setoracle` is the general bound, and it is granularity-conditional

Worth stating precisely, because it decides every case below. `setoracle` picks
the optimal *set* under the exact joint cost, greedily. Any per-token or
per-block ranking function is a strictly weaker instrument than set selection,
so **every method that ranks and takes top-k is bounded above by it** — and it
ties recency.

The bound is conditional on one thing only: `setoracle` was run at a fixed block
size, so it bounds selections over units of that size and says nothing about
finer granularity, where the feasible set is strictly larger. That is the entire
residual, and `granularity.md` records why it stays unmeasured.

## TriAttention: an analytic attention-mass estimator

With `k` the pre-RoPE key recovered by inverting RoPE off the cache and `q̄` the
calibrated per-head mean pre-RoPE query, per frequency pair *f*:

    amp   = |q̄|·|k|     phi = arg(q̄ · conj(k))     extra = (E|q| − |q̄|)·|k|
    score(d) = Σ_f amp_f · scale_f · cos(ω_f·d + φ_f) + Σ_f extra_f · scale_f

That is the closed-form **expectation of the RoPE'd q·k logit at relative
distance d** under the calibration query distribution, split into a coherent
term (mean query direction, oscillatory in distance — the "trigonometric
series") and an incoherent term (residual query magnitude, flat in distance).

So it is not a different signal. It estimates the quantity `massoracle`
measures exactly, and it is *strictly weaker* on the one workload where that
oracle wins: substituting `q̄` for the live query cannot find a needle, because
a needle query is by construction atypical.

Two properties are worth keeping regardless:

- **The summary is recomputable, not stored.** Nothing per-token is kept; `k` is
  recovered on demand by inverting RoPE and `amp`/`phi` are transient. Only the
  calibration file persists — `q_mean_complex` + `q_abs_mean` per sampled
  (layer, head), context-independent. That is a real advance over Quest's stored
  min/max bounds, which `granularity.md` shows stop paying below block 72.
- **The cost is compute, and it forces the cadence.** Scoring is
  [keys × offsets × head_dim/2] cosines per head per layer, so it compacts every
  128 steps. Caching `amp`/`phi` to avoid that would cost about as much as the
  keys. **A score affordable every 128 steps cannot be a residency policy**,
  which re-decides every step.

## SnapKV: attention mass, plus an unacknowledged contiguity prior

`C = Σ_i W_obs[:,i,:]` — attention from the last 32 prompt tokens summed over
the query axis, top-5% per head, prefill only. The observation window is a proxy
for future queries; the signal is attention mass.

The interesting part is step 14 of their pseudocode: **1D max-pooling, kernel
5-13, applied to the scores before top-k**, which retains the neighbours of
every high-scoring position. Their Figure 7 ablates it: without pooling,
LongEval-Lines retrieval fails; with kernel 5 it holds to 16K. Same scores, same
budget — the difference is whether the kept set is contiguous.

That is independent corroboration of this project's contiguity finding from a
different codebase, and the two read better together: SnapKV shows the
contiguity term is what makes mass-based selection survivable, and our oracles
show the mass term adds nothing on top of contiguity.

## R-KV: redundancy is a genuinely different quantity, wired into a ranking

    I_i = (1/α) Σ_j Ã_{j,i}        Ã = attention from α observation tokens, max-pooled over 2W
    S   = K̄ K̄ᵀ                    cosine similarity of normalised KEY vectors
    R_i = softmax(mean_j S_{i,j})
    Z_i = λ·I_i − (1−λ)·R_i        top-k, every 128 decode steps, β recent forced

Redundancy is not attention mass, and that was the one reason to look. Two
things defuse it:

- **It is not set-aware.** `R_i` is mean similarity to *everything*, computed
  once — not similarity to what has already been selected. There is no MMR loop
  and no re-scoring after each pick, so it is not a coverage objective at all;
  it is an anti-centrality score, "prefer keys unlike the average key".
- **Which makes the whole thing a per-token ranking**, and therefore bounded by
  `setoracle` above.

The one condition under which it would escape: if `setoracle`'s joint cost were
defined over mass coverage rather than output effect, a diversity criterion
would not be in its span. It is output-level, so it is covered.

## The shared blind spot

| | importance signal | contiguity component | recency component | recency baseline |
|---|---|---|---|---|
| SnapKV | obs-window attention | max-pool k=5-13 | — | **none** |
| R-KV | obs-window attention | max-pool 2W | β recent forced | **none** |
| TriAttention | expected logit vs `q̄` | — | `WINDOW_SIZE=128` | **none** |

Every one carries an unablated recency or contiguity term, every one benchmarks
against the others, and **not one compares against sink+recent alone.** R-KV's
tables have SnapKV and FullKV; TriAttention's have SnapKV and R-KV; SnapKV's
have neither.

That explains why the literature reads as though there is headroom while our
measurements say there is not: the published comparisons are scorer-against-
scorer on top of a shared contiguity substrate nobody isolates. Our negative
tested the thing all three left out, which is also why it disagrees with all of
them at once — and why that disagreement is not evidence of a measurement
error.

Related: TriAttention's own AIME table shows 42.1 against Full Attention's 57.1
at budget 2048; the near-lossless rows are at 3072-4096. The compression headline
and the no-loss headline come from different budgets.

## What would reopen it

Not a better ranking — anything of that shape is bounded by `setoracle`. It
would take a signal that is not a function of the current cache state: task
structure known before the query (tool-call boundaries, explicit retrieval
intent), or changing *what is in* the cache rather than which parts survive
(merge, or recompute from source tokens). All three methods here are permanent
eviction and none can bring anything back, so the one property this plugin has
is untested against them — and our own data says nothing wants to come back.

The full account of why TriAttention's vLLM path could not be run truthfully,
and the calibration work that survived it, is in `vllm-exl3-plugin`
`docs/triattention.md`.
