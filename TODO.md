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

## `demand-signal` — scoring exists, and selects the right block

`bounds.py` holds the per-block key ranges, `quest.py` scores them against the
previous step's queries, and `policy.py` exposes `quest`. The decision travels
worker to scheduler through the shared state, one step stale, which is the
protocol step the design predicted from the moment block order was measured: a
query only exists inside the forward, and residency must be settled before it.

**First result.** At a budget of 16 of 129 blocks, with a needle planted in a
known block, the scored policy *selects* that block and recency does not. It is
selection that is measured, not the answer: a needle answer is emitted in the
first token or two, before a query-aware policy has seen a single query, so no
such policy can affect it. Any harness meant to judge a scored policy on
answers has to ask its question late.

**The aggregation was the whole result**, and the default was wrong on import.
With `head_agg="mean"` the policy missed the block; with `"max"` it found it,
same budget, same everything. The mean default came from an earlier measurement
that mean captures more attention *mass* across a GQA group — a different
question from retrieval, where the signal sits in a few query heads and
averaging dilutes it. The layer-wise choice made no difference either way,
which leaves the all-layer union looking less fatal than feared.

**The metric to refine against now exists.** `audit.py` recomputes attention
each step from the true keys -- resident ones from the GPU, evicted ones from
the host tier, which an eviction method could never do -- and reports the mass
that sat in blocks the policy did not have. Transfer counters cannot
distinguish a policy that fetches well from one that fetches constantly;
`missed_mass`, `fetched_mass` and `evicted_mass` can.

**Ranking on accumulated standing rather than the current step.** A step's
scores become a distribution before they are accumulated -- their scale rides
on the query's norm, so mixing raw scores would let one step's magnitude
outvote another's shape -- and a block's standing decays instead of being
replaced. A block seen for the first time starts at what it is worth now, so a
new signal is still actionable immediately; an old one fades rather than
vanishing, which is the pressure to keep a block that has stopped being
indicated. GSM8K as 8 turns, budget 12 blocks:

| decay | missed mass | worst layer | set churn | moved out/in |
|---|---|---|---|---|
| 1.0 (per-step) | 0.0097 | 0.427 | 9.65 blk/step | 4747 / 4289 |
| 0.5 | 0.0098 | 0.386 | 4.45 | 2761 / 2293 |
| 0.2 | 0.0076 | 0.417 | 2.00 | 1347 / 886 |
| **0.05** | **0.0070** | **0.297** | **0.66** | 670 / **217** |

Better on every axis at once, which is not what a smoothing knob usually does:
28% less missed mass, a third off the worst layer, 15x less set movement and a
twentieth of the fetches. Against recency (missed 0.0121, 460 out, 0 in) the
scored policy is now 42% better on mass at a transport cost in the same order
rather than ten times it. The trend had not turned at 0.05, so the optimum may
be lower and is untested.

**On a model that can do the task, recency wins.** Qwen3-8B-AWQ, GSM8K as 12
turns, budget 12 of ~90 blocks (13% resident), prefix caching on:

| arm | correct | moved |
|---|---|---|
| no plugin | 11/12 | — |
| churn | 11/12, bit-identical | 41335 / 41020 |
| **recency** | **10/12** | 1136 / 0 |
| quest | 8/12 | 1589 / 441 |

At 13% residency both policies keep nearly all of it, and the scored one is
two answers *worse*. That is a real negative result for the current scoring,
and it is also close to the least informative task for the question: GSM8K
needs the exemplars at the start and the question and reasoning at the end, and
nothing in between, which is precisely what recency keeps for free. The harness
was chosen for being mechanism-sensitive and policy-insensitive; it is
behaving as designed and cannot show a scored policy's advantage.

What can, and does, is the needle: quest selects the block holding a planted
answer and recency does not. **The task that would settle it needs both** --
distant retrieval *and* generations long enough for a query-aware policy to
act. Neither harness here is that, and building one is the next thing.

**A recency floor under the scored policy is not optional.** Without one quest
scored 0/12; with half the budget reserved for the newest blocks it scored
8/12. Attention is heavily recency-weighted and a generation has to see what it
just wrote; a pure ranking has no floor under that. `recent` now defaults to
half the budget.

**The measurement bug that produced two rounds of wrong conclusions.** The
answer extractor matched a trailing period, so `"18."` was scored wrong against
a gold `"18"`. It was not uniform across arms -- it penalised whichever arm's
output happened to end with a period -- which is exactly the shape of a
difference that reads as a result. Under it, recency read 6/12 instead of
10/12, quest read 0/12 instead of 8/12, and the recency-floor sweep read
0/0/1/1 instead of 0/0/6/8. Conclusions drawn and then withdrawn: that quest
was catastrophically worse than recency, and that a recency floor barely
helped. Both were artifacts. Nothing was wrong with the plugin.

**The earlier comparison, kept because it is what the smoothing fixed.**
GSM8K as 8 turns, budget 12 blocks:

| policy | missed mass | worst layer | fetched mass | moved | correct |
|---|---|---|---|---|---|
| recency | 0.0121 | 0.415 | 0.0000 | 460 out, 0 in | 3/8 |
| quest | **0.0097** | 0.427 | 0.0118 | 4747 out, **4289 in** | **0/8** |

The scored policy captures about 20% more attention mass and does *worse* on
the task, at ten times the transport. Three things to take from that, none of
them "quest is bad":

- **The proxy moved the right way and the outcome moved the wrong way.** Mass
  captured has always been a proxy for quality rather than a measurement of it;
  this is the first time the two have been observed disagreeing here, and it is
  the reason the end-to-end harness exists.
- **n is 8, on a model that scores 3/8 at full context.** The accuracy column
  is not evidence of much. The mass column, over 911 audited steps, is.
- **Nothing bounds the fetching**, and it shows: 4289 restores over 911 steps
  is ~4.7 blocks per step, which at the measured transport cost is real latency
  spent for a proxy improvement of 0.0024. The fetch ceiling stops being a
  future refinement here and becomes the next thing to build.

What is not yet known:

1. **Whether it beats recency on anything but a planted needle.** One block,
   one prompt, one model. Selection accuracy against real attention mass is
   unmeasured.
2. **What the bounds cost in practice.** Held in fp32 at
   `num_kv_heads * head_size * 2` per block per layer, which is not a
   deployable format, and the memory comes out of the same budget the blocks
   do.
3. **Whether one step of staleness is enough** at real generation lengths.
   `ranked` counts how often a ranking was available and it is low on short
   runs.
4. **A fetch ceiling**, now measured as the pressing gap rather than a
   theoretical one -- see the churn figures above.
5. ~~Whether the instability is the problem.~~ **Measured, and it was.**



`recency` never fetches; its window only slides forward. At 12% residency an
oracle reproduces the full-context answer token for token and recency loses it
(README, "Status"), so the mechanism is not the limit.

A policy can only fetch if something resident tells it a non-resident block is
wanted. Quest-style per-block key bounds are the shape: min/max per channel,
kept when the block leaves, giving an upper bound on that block's attention for
the current query without its keys. Not free — about 4 KiB per block per layer
in fp16 against a 32 KiB block, out of the same budget — and worth measuring at
the intended geometry before committing to it.

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
