# The demand signal, and why it is closed

This project began with a plan: a *demand signal* — something resident that
says "you will want this block that is not here" — because that is the one
capability a pager has and an eviction method cannot. Without it, residency
can only slide forward and the restore path is dead weight.

The plan does not work. This is the record of how thoroughly, because the
result was arrived at against sustained scepticism and four separate
confounds had to be cleared before it could be trusted.

## The finding

A scored policy lost **every one of fourteen configurations** — two model
architectures, three budgets, five block sizes, two model families — while
paying 1.5–2.5× the copy-out traffic and thousands of fetches recency never
makes.

More decisively, at a fixed budget on an 84k-token agent session:

| policy | ranks on | agreement |
|---|---|---|
| `recency` | position only | **0.2374** |
| `massoracle` | *true* attention mass | 0.2372 |
| `impactoracle` | *true* marginal output shift | 0.2052 |
| `quest` | bounds estimate of mass | 0.2256 |

**Perfect knowledge of attention mass ties a policy that ignores it to four
decimal places.** There is no headroom in that quantity, so no estimator of
it could ever have won — which retires the bounds, the 2-bit summary and
quest together, since all three estimate the same empty target.

## Mass is not importance

The natural response is that mass is the wrong target: what matters is
whether dropping a block *changes the output*, and a block can carry large
mass while sitting near the output's centroid. That quantity is available in
closed form. For `o = Σ w_k v_k`, dropping block *b* renormalises the rest
and moves the output by exactly

    Δo = m_b · (o − v̄_b) / (1 − m_b)

(verified against brute-force block removal to 5×10⁻⁷ relative). Ranking by
it — `impactoracle` — is **worse** than recency, at 0.2052.

That indicts the objective rather than the idea. `Δo` is the leave-*one*-out
shift, used to choose a set that drops ~85% of blocks, and those effects do
not compose. Dropping a set D costs

    ‖Σ_{b∈D} d_b‖ / (1 − Σ_{b∈D} m_b),   d_b = m_b·o − s_b

which is a norm of a *sum*, not a sum of norms: blocks whose error vectors
oppose each other cancel, and dropping them together is nearly free.
Magnitude ranking is blind to that. On a small case, greedy joint selection
reaches the brute-force optimum while marginal ranking is 16% worse.

`setoracle` selects against the joint objective, and it is the true ceiling.
It ties recency too:

| policy, block 1056 | agreement | copied out | fetched |
|---|---|---|---|
| `recency` | 0.2102 | 2,645 | **0** |
| `quest` | 0.2087 | 6,454 | 3,807 |
| **`setoracle`** | **0.2121** | **27,572** | **25,986** |

A 0.0019 difference, deep inside a ~0.03 noise floor, for ten times the
copy-out traffic and 26k fetches recency never makes.

**That is the closing result.** The previous two oracles could each be
dismissed on objective — mass is the wrong quantity, and a leave-one-out
shift is the wrong way to choose a set. This one uses the exact joint cost,
greedily minimised, verified against brute force. It is the best set of
blocks obtainable with perfect knowledge, and it buys nothing.

So contiguity is not a heuristic that happened to beat some bad estimators.
**There is no set of blocks materially better than the most recent ones at
these budgets** — which is why every signal failed, and why a better signal
would not have helped.

## The proxy ran backwards

Mass capture — the objective every signal here estimated — ordered the
policies *backwards* end to end:

| arm | mass captured | token agreement |
|---|---|---|
| recency | 0.274 | 0.1427 |
| quest | 0.664 | 0.0982 |
| massoracle | 0.872 | 0.0621 |

The better the mass capture, the worse the output. That was the first hard
evidence that the target was wrong, and it took a while to believe because
mass capture is such a natural thing to optimise.

## Four confounds, all cleared

The conclusion is trustworthy because each objection was measured rather
than argued away.

**The baseline was a strawman.** Every mass comparison ran against a recency
selector keeping *no* sinks, while the shipped policy keeps two. Two blocks
of 112 hold **0.4580 of all attention mass**, so the real comparison is
sinks+recency 0.7907 against an oracle's 0.8812 — 11% relative, not the 220%
a sinkless baseline implies.

**Decode volume was unrepresentative.** Non-thinking generation produced 138
tokens/turn against 617 with `--thinking`, and that difference *reversed* a
result: paging appeared to beat truncation by 59% at low decode volume and
lost to it at realistic volume.

**Context was too short.** Every mass number was taken below ~5k tokens,
which is the regime least favourable to scoring: the gap between what
recency gets free and what an oracle can reach roughly doubles from 112
blocks to 346 (+0.049 → +0.114 at 5% budget). No mass number taken below ~5k
tokens should be generalised.

**Granularity was the last objection**, and it is flat. On a uniform-RoPE
full-attention model where block size is a free parameter:

| block | recency | quest |
|---|---|---|
| 64 | 0.2147 | 0.2029 |
| 1056 | 0.2102 | 0.2087 |

A 16.5× change moves nothing outside noise. Earlier selection results sat at
~40 units of freedom, where "selection does not help" and "selection cannot
be expressed" are the same measurement; they are separable now and the answer
does not change.

## What was learned that outlives the negative

**A fixed-size per-block summary does not scale down.** Quest-style bounds
cost `layers × kv_heads × 2 × head_size` whatever the block holds — 288 KiB
on a 36-layer model — while a block is `B × kv_heads × head_size × 2 × 2`
bytes:

| block size | block | bounds as % of block |
|---|---|---|
| 16 | 64 KiB | **450%** |
| 72 | 288 KiB | **100% — break-even** |
| 528 | 2112 KiB | 14% |

Below block 72 the summary exceeds the block it describes, so bounds are
pointless at fine granularity however well they rank. A quantized key
summary is `B × 256` bytes — a flat **6.25%** at any block size — so it
scales correctly. If a per-block summary is ever wanted again, that is the
shape it has to have.

**Quantized keys rank at the oracle ceiling.** Against a mass oracle at
0.8756, 2-bit keys reach 0.8755, and one step of staleness costs 0.55%
relative. Ranking never needed accuracy, only order. The result stands; it
is simply aimed at a target with nothing in it.

**Provable skipping is impossible, and not because of the layer union.**
Exact output and evicting are incompatible outright: full attention at step
*t* attends to all *t* keys, so bit-exactness means streaming the whole
context per token. And skipping provably-negligible blocks does not rescue
it — at ε=1e-4 the *minimum* any single layer demands alone is 89.3 of 90
blocks. Every layer has a long thin tail; there are no sparse layers to
weight toward.

**Contiguity beats selection against exact knowledge**, which is a stronger
statement than "our estimator was bad". Both oracles knew everything and
neither won.

## Dead ends recorded so they are not retried

- **Quest-style min/max bounds.** Rank at 0.6466 against 0.8811 for 2-bit
  keys, cannot prove anything (they overstate true mass by 9.4 orders of
  magnitude), and do not scale below block 72.
- **A U-shaped positional prior on the scores.** Deltas of ±0.006 against a
  ~0.03 noise floor across three block sizes: no measurable effect. The
  standalone version of the idea is just sink+recency, which the sink sweep
  already covers.
- **Layer-0 queries as a pre-forward signal.** Computable before any
  attention runs, ranks at 0.3473 against recency's 0.2743 — real but not
  competitive. Only interesting for a prefill cold start, where nothing else
  can exist.

## The published methods are bounded by `setoracle`

Surveyed after the fact, to check whether anything scores a quantity that was
not measured here. TriAttention, SnapKV and R-KV all reduce to per-token
rankings, and every ranking is a weaker instrument than optimal set selection.
Formulas, the one genuinely novel quantity (R-KV's redundancy term, which is not
set-aware and therefore does not escape), and the shared blind spot — none of
the three carries a recency baseline — are in `eviction-survey.md`.

## What this result is and is not scoped to

Recorded because the configuration was never varied, and it should not be
rediscovered as a surprise.

**Every arm here ran with the prompt attended in full during prefill**, and with
prefix caching on, so per-turn prefill covered only the new suffix. In that
regime the resident tail is a full-context encoding: recency already has
indirect access to everything, and there is nothing in the offloaded blocks that
is not also represented in what is resident.

**The ranking survives that; the level probably does not.** Mass concentrating
on recent blocks is RoPE decay plus learned locality, neither of which windowed
prefill touches — under a worse tail encoding, attention still decays toward
recency and `massoracle` still lands where `recency` already is. What is
unmeasured is the *absolute* number under paged prefill, which can drop while
the curve stays just as flat. That is a capacity result, not a policy one, and
`capacity.md` is where it goes.

The flatness itself is the strongest evidence against a suppressed signal: a
methodological confound compresses a margin, it does not push perfect knowledge
*below* a positional heuristic. `impactoracle` at 0.2052 against recency's
0.2374 is an oracle paying a real churn cost for a benefit that is not there.
