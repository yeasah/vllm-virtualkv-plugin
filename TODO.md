# TODO

Open work only. What is settled lives in `README.md`.

## `serving-knobs` — partly done

A budget can now be written as blocks (`64`), tokens (`1024t`) or a share of
`max_model_len` (`25%`), resolved once the engine can say how big a block is.
That was the part accuracy work needed: a sweep written in blocks is not a
sweep of the same quantity across models or block sizes, so results taken at
one geometry cannot be compared with another.

Two pieces remain, neither blocking:

1. **Per-request is still the wrong scope.** The quantity an operator controls
   is the *pool*, shared across whatever is running; a per-request budget
   becomes a global one multiplied by concurrency. Wants cross-request
   coordination, which is a design rather than a knob.
2. **The fetch ceiling is not expressed at all.** The transport measurement
   says the budget that matters is *absolute* — roughly 543 tokens per decode
   step at 5% added latency on a 27B at fp8, whatever the context length — so
   neither a block count nor a percentage says the thing a policy has to bound.
   That knob only becomes meaningful with a policy that fetches, so it belongs
   with `demand-signal`.

## `undersized-cache` — you cannot declare a context larger than VRAM

**Do not do this before `prefill-residency`.** On its own it converts a startup
error into an OOM during the first prefill, since prefill allocates the whole
prompt before any residency decision is made. The two are one piece of work.


`check_enough_kv_cache_memory` (`vllm/v1/core/kv_cache_utils.py`) raises at
startup unless **one** request at `max_model_len` fits entirely in the KV cache.
So the configuration this plugin exists to enable — declared context larger than
the GPU can hold — is refused before any of it runs. TriAttention monkeypatched
exactly this check, which is a fair signal about the shape of the fix rather
than an endorsement of it.

Patching the check alone is not enough; see below.

## `prefill-residency` — the peak this does not touch

**Prefix hits shorten this considerably**, which was not obvious. Cache-hit
tokens never go through a prefill forward — `num_computed_tokens` jumps to the
hit length — so the guard here (`processed < num_prompt_tokens`) clears almost
at once and paging starts nearly immediately. Over a multi-turn session with a
high hit rate the genuinely-prefilled portion is small, and the peak is set by
the first cold turn rather than by the running context. Keeping any single
prompt under the KV capacity therefore buys most of what prefill residency
would, without building it.


Paging is decode-only (README, "Limits"). That bounds the *steady-state* decode
footprint, and the blocks genuinely return to the pool, so a workload of many
long generations really does hold less. What it does not do is let a request
declare a longer context than VRAM, because prefill allocates the whole prompt
before any residency decision is made.

**These two are one barrier, not two.** Lifting `check_enough_kv_cache_memory`
without paging prefill just moves the failure from a startup error to an OOM
during the first prefill. Long context needs both, and prefill paging needs its
own correctness argument: the write span is a run of blocks rather than one,
block order becomes load-bearing where at decode it carries none, and
prefix-cache hashing is live so an evicted block can be handed to another
request by hash.

Order: prefill residency first, then the startup check, then the knobs.

## `demand-signal` — measured, and the premise was wrong

**Quest-style min/max bounds are not the demand signal.** Carried here as a
premise since before any of this was built, and now measured twice against
alternatives, failing both times. Same budget for every selector, scored on
the share of true attention mass its pick holds, summed over all 36 layers
and 32 heads. 23 of 90 blocks, 137 decode steps of a real session,
Qwen3-8B-AWQ:

| selector | mass captured |
|---|---|
| oracle (ceiling, impossible timing) | 0.8756 |
| 8-bit keys | 0.8756 |
| 4-bit keys | 0.8756 |
| 2-bit keys | 0.8755 |
| oracle, 1 step stale | 0.8708 |
| **2-bit keys, 1 step stale — deployable** | **0.8708** |
| min/max bound | 0.6396 |
| layer-0 queries | 0.3473 |
| recency (floor) | 0.2743 |

**A quantized copy of the keys is the signal**, at two bits, ranking as well
as the true keys do against a ceiling nothing can exceed. Ranking never
needed accuracy, only order, and 2-bit codes destroy the values while
preserving the order. About 25% of key bytes with the per-block scales, so
~12.5% of KV to carry an essentially-oracle demand signal for the whole
context — against 4 KiB per block per layer for bounds that rank at 0.64.

**Staleness is nearly free, which was the gating unknown.** Residency for
step N is settled before step N's forward, so the freshest query a policy
can hold is N-1's. That costs 0.55% relative (0.8756 -> 0.8708) with a
perfect summary, and the 2-bit summary costs nothing on top of it. The
autocorrelation the accumulated-standing scorer was built on is now measured
rather than assumed.

**Provable skipping is dead, and not because of the union.** Hypothesis: the
flat union over 1152 head-layer pairs is set by a sensitive few, so weighting
layers would defuse it. Wrong — at epsilon=1e-4 the *minimum* any single
layer demands alone is 89.3 of 90 blocks. There are no sparse layers. Every
layer has a long thin tail, so a per-head threshold that strict is
unsatisfiable by anyone and no weighting recovers it. Exact output and
evicting are incompatible anyway: full attention at step t attends to all t
keys, so bit-exactness requires streaming the whole context per token. The
target is statistical fidelity, which puts this on the same footing as every
other KV scheme — it has to beat the alternatives or combine with them.

| epsilon | bound needs | oracle needs |
|---|---|---|
| 0.1 | 100.0% | 28.5% |
| 0.01 | 100.0% | 99.1% |
| 0.001 | 100.0% | 100.0% |

**Early layers are the real asymmetry.** Under a globally optimal 25% pick
the median layer keeps 0.8857 of its mass and the worst keeps 0.5306 —
nearly half gone — and the worst is essentially always layer 1. Two layers of
thirty-six is a cheap carve-out, so this is a targeted fix rather than a
structural problem. (`worst_layer_idx` is a mean of per-step argmins, not a
mode; the full per-layer distribution is unmeasured.)

**Layer-0 queries are computable before any forward** — embed, norm, q_proj,
no attention — and rank at 0.3473. Not competitive as a general signal, but
the prefill cold start is the one place where nothing else can exist, and
there the only bar is recency.

What is not yet known:

1. **Whether mass capture translates into output quality.** It is a proxy,
   and this repo has already seen a proxy improve while the outcome got
   worse. `tools/session_turns.py` is the instrument; the question is whether
   0.87 capture beats recency's 0.5937 token agreement by a matching margin.
2. **Whether it beats eviction at equal residency**, which is the claim the
   design rests on: paging can do everything eviction does and then fetch
   back what it got wrong. Untested against H2O/SnapKV-style baselines.
3. **Longer contexts.** All of this is ~1600 keys. Sparsity may improve with
   length, which is where a pager earns its keep.
4. **A fetch ceiling**, still unexpressed: the transport measurement says the
   budget that matters is absolute, ~543 tokens per decode step at 5% added
   latency, which neither a block count nor a percentage says.

**The instrumentation was wrong twice in one session**, in ways that changed
conclusions, while the plugin passed every exact check it was given:
`audit.py` never applied the attention scale (softmaxing logits ~11x too
large, off a far sharper distribution than the model's), and the working-set
softmax excluded the partial tail block, handing its mass to older blocks —
that one alone moved the oracle from 99.5% to 26.4% at epsilon=0.1. Every
`missed_mass` figure recorded before those fixes is off; ordering may
survive, magnitudes should not be quoted. **Before more decisions rest on
that path it needs a test pinning its reconstructed attention against the
model's own attention output.**

## `auto-context` — an upstream idea, noted here so it is not lost

Sizing a context by hand against available memory is tedious and gets redone
every time the model, the quantization or the card changes. `max_model_len =
auto:N` would ask for the largest context that leaves room for `N` concurrent
requests.

Most of it already exists upstream: `estimate_max_model_len(vllm_config,
kv_cache_spec, available_memory)` in `vllm/v1/core/kv_cache_utils.py` binary
searches for exactly this and restores the config it borrowed, and today it is
called only to make the "doesn't fit" error message friendlier. `auto:N` is
close to parsing the suffix and calling it with `available_memory // N`.

Not this plugin's business — it belongs upstream, and it is worth more there
than here — but it is adjacent enough to record: this plugin's own knobs have
the same problem, which is `serving-knobs` above.

## `prefix-caching` — first evidence, and it is better than expected

Exercised 2026-09-06 across a six-turn conversation whose prompt grows each
turn, which is the shape that actually creates shared prefixes:

- `churn` with prefix caching **on** is bit-identical to no plugin, in tokens
  and logprobs, over all six turns. 1287 blocks out, 1152 back, no restore
  unbacked, no guard violation.
- `recency` at a real budget with caching on runs clean too — 375 evictions,
  no violations — and drops context, which is what it is for.

That is consistent with the argument the design made and never tested: a pager
*relocates* blocks rather than modifying them, so a block's hash stays truthful
whoever holds it. A request that evicts a block still owns those tokens; the
pool may hand the block to someone else under its hash, and both are right,
because our copy comes back from the host rather than from that block.

One consequence of caching turned out to be a rule rather than a caveat, and
is now implemented: **a block another live request holds is never evicted.**
`free_blocks` only queues a block once its ref count reaches zero, so freeing a
shared one returns no memory at all, while still costing a host slot — and the
restore allocates a fresh block, so a block that was shared between two
requests comes back as a private copy and the total goes *up*. Keeping it is
free, because someone else is paying for it either way. It needs genuine
prefix sharing to arise, which is to say it needs real traffic.

That also sharpens what a budget means: a request holding a shared block is not
spending its budget on it in any sense that matters, so residency accounting
should eventually be per-block-refcount rather than per-request. Not done.

Not settled. Churn only hides a block for a single step, and the recency arm
has no exact reference to be judged against — it changes the output on purpose.
What would settle it is a run where a *second* request demonstrably hits the
hash of a block the *first* has evicted, with the hit verified rather than
hoped for. Nothing here proves that case occurred.

## `hybrid-attention` — required, and it gates several retracted conclusions

**Next.** Not a compatibility item: a plugin that cannot page a hybrid model
is not useful, because hybrids are what gets deployed. And the architecture
happens to attack the findings this repo gave up on, so the re-tests it
enables matter as much as the support does.

`~/ckpt/Qwen3.5-9B-exl3-4.00bpw-bq` is local and vLLM registers
`Qwen3_5ForConditionalGeneration`. What changes:

| | Qwen3-8B (everything measured so far) | Qwen3.5-9B |
|---|---|---|
| layers | 36, all full attention | 32, **8 full attention** (every 4th) |
| query heads | 32 | 16 |
| head-layer pairs sharing a block table | **1152** | **128** |
| KV per token | 144 KiB | **32 KiB** |
| max context | 32k | 262144 |

**What this may exonerate.** `provable skipping is dead` rests on a union over
1152 head-layer pairs -- a block is undroppable if any one of them wants it.
At 128 pairs that is a different proposition and the measurement should be
redone before the conclusion stands. Likewise every mass number here was
taken below 5k tokens, which `sink-mass` shows is the regime least
favourable to scoring; 32 KiB/token puts 128k within 4 GiB, so the regime
where the recency-vs-oracle gap actually opens becomes reachable on one card.

**The work, which is not a one-liner.** The plugin assumes a single KV cache
group in six places, and a hybrid has two -- full attention, and linear
attention state:

- `block_tables[0]`, `slot_mappings[0]`, `kernel_block_sizes[0]` index group
  *zero*, which for a hybrid may be the linear group. The pager would rewrite
  the wrong table.
- `kv_cache_groups[0].kv_cache_spec` reads `head_size`/`head_size_v` from
  whichever group is first.
- `runner.kv_caches` is handed wholesale to the host tier and to every
  measurement, so linear-attention state would be copied and reconstructed as
  though it were K/V.

Index everything by the group whose spec is the paged one. **Assert it rather
than find it quietly**: reading the wrong tensor as keys yields plausible
numbers instead of a crash, which is the failure class that has cost this
repo the most. `block_keys` already refuses to slice hopefully; the group
identity deserves the same treatment.

Passing non-full-attention specs through untouched is already correct for a
hybrid, so that part needs nothing.

**Do it on EXL3 with an fp8 cache.** The AWQ checkpoint pays ~28% in
embeddings, and every tier comparison here used `base_bits=16` because that
is what AWQ gave -- fp8 is the defensible baseline and halves each degraded
tier's apparent advantage. Both are fixed by the same move.

## `sink-mass` — the baseline was a strawman, and that explains the rest

**Two blocks of 112 hold 0.4580 of all attention mass.** The shipped
`recency` policy keeps `sink=2`; the selector every mass number in this repo
was compared against kept none. So:

| selector | mass captured |
|---|---|
| oracle | 0.8812 |
| 2-bit keys, one step stale | 0.8811 |
| **sinks + recency (what ships)** | **0.7907** |
| recency, no sinks (what was quoted) | 0.3384 |

The reported "2-bit captures 0.8708 against recency's 0.2743, 3.2x" is wrong:
the sinkless baseline flattered it by 46 points of sink mass.

**But the corrected gap is not small either, and that correction was itself
taken at the wrong context length.** Those numbers came from 112 blocks --
about 1800 tokens. Repeating the sweep at 346 blocks, the gap roughly doubles
at every budget:

| budget | gap @112 blocks | gap @346 blocks |
|---|---|---|
| 5% | +0.0492 | +0.1137 |
| 10% | +0.0646 | +0.1501 |
| 25% | +0.0969 | +0.1770 |
| 50% | +0.0968 | +0.1642 |

The components say why: `sinks+recency` decays with length (0.7907 -> 0.7121)
while the oracle holds (0.8875 -> 0.8892). A recency window is a fixed
fraction of a growing context so it covers proportionally less of what
matters; a scored policy follows the content. Sink mass is flat at ~0.45, so
sinks dominate only at short context.

At 5.5k tokens the prize is already 25% relative and the trend is steep and
not flattening. Every KV-compression method is evaluated at 32k-128k, which
is where scoring earns its keep -- and it answers the obvious objection
("why does TriAttention work, then?"): it operates where the gap is large.
Both results hold; the short-context one simply cannot show it.

**No mass number in this repo taken below ~5k tokens of context should be
generalised.** That is a sharper rule than the sink one and it invalidates
more.

It is also the mechanism behind mass and importance diverging: 46% of mass in
two blocks carrying no information is what the sink literature describes.

The 2-bit summary still ranks at the oracle ceiling. That result stands; the
value of ranking well is what shrank.

## `proxy-is-broken` — mass capture ordered the policies backwards

The demand-signal work optimises attention mass captured. End to end, on a
real session at budget 64, more mass capture produced *worse* output:

| arm | mass captured | token agreement |
|---|---|---|
| recency | 0.274 | 0.1427 |
| quest (bounds) | 0.664 | 0.0982 |
| massoracle (measured mass, the ceiling) | 0.872 | 0.0621 |

`massoracle` reconstructs every block's true attention mass each step and
ranks on it, so no estimator can do better. It was the worst arm. A recency
floor sweep under it confirms the direction: agreement climbs as the floor
grows, and the best mixture (0.1368) still loses to pure recency (0.1427).
Every block spent on mass-ranked selection would have been better spent
extending the recent window.

Two candidate explanations, and they want different fixes:

1. **Holes are out of distribution.** A contiguous recent window is a shorter
   conversation; a mass-ranked set has gaps at positions matching nothing the
   model was trained on. If so, sparse residency carries an intrinsic penalty
   no selector avoids, and this is a problem for every non-contiguous scheme.
2. **The workload never needs distant context**, so keeping it cannot pay and
   only the damage is visible. The `truncate` result below supports this one.

Not settled. Distinguishing them needs a task that genuinely requires distant
content -- see `capability-suite`, which this now blocks on.

## `truncation-is-the-baseline` — recency is not a policy

`recency` matches a plugin-free `truncate` arm that cuts the prompt to the
same tokens (0.4606 vs 0.4565 agreement) and *loses* on drift (0.0045 vs
0.0016) and exact turns (2/10 vs 4/10). So the host tier, block-table
rewriting and guard buy nothing over a shorter prompt at 33% residency on
this session.

Everything this repo has compared against recency has therefore been
comparing against truncation without saying so. The bar is truncation, and
only the needle test clears it.

## `capability-suite` — the harness that could actually judge a policy

Neither existing harness can. The needle needs distant retrieval but answers in
the first token or two, before a query-aware policy has seen a query. GSM8K as
turns has long generations but independent questions -- the history is ballast,
nothing in turn eight depends on turn three, and a policy that drops the middle
loses almost nothing. That is why one is policy-sensitive and the other
mechanism-sensitive, and lengthening either does not convert it.

What is wanted is genuine cross-turn dependency *with* long generations: a
conversation whose later turns require what earlier ones established. That is
a benchmark question rather than a flag, and it is the thing standing between
here and an answer on whether scoring beats recency.

Two pieces of it now exist in `gsm8k_turns.py` and are worth keeping whatever
replaces it. `--chat` with `--thinking` puts the decode-token volume where a
real request has it: Qwen3 emits 1000-6700 characters of reasoning per turn
against a ~300-character gold answer, and that trace is most of what a pager
has to serve. `--replay` feeds a previous run's generations back as the shared
history, so the conversation is model-shaped and still identical across arms --
which is the only way to have both length and pairing.

## `serving-exposure` — point something real at it

`tools/gsm8k_turns.py` is the halfway house: a growing conversation with a high
prefix hit rate, scored against ground truth, replayable across arms. It found
a config-ordering bug on its first run (below), which is the argument for it.
It is not a substitute for real traffic — the request shapes are still ones we
chose.


The argument for doing this earlier than its dependencies suggest: benchmarks
cover the request shapes we thought of, and real traffic covers the ones we did
not. Multi-turn conversations, aborts, preemption, ragged lengths and the
attention patterns they produce are where architectural problems surface, and
architectural problems are the expensive kind to find late.

The lifecycle leak found on 2026-09-06 is the argument in miniature: nothing
released a finished request's host slots, so a server would have lost one per
evicted block per request until the tier filled, then refused every eviction
and quietly stopped honouring the budget. No test of a handful of requests
could see it; a morning of real use would have.

It implies `prefill-residency`, which is not the tidy next step. That is the
trade, and it is a real one rather than an argument for postponing.

Still untested and reachable this way: preemption and resumption (a preempted
request has all its blocks freed while the tier still holds copies keyed to
it), request aborts mid-generation, and contexts past a few thousand tokens.
