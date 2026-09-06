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

## `demand-signal` — the policy that makes any of this pay

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

Not settled. Churn only hides a block for a single step, and the recency arm
has no exact reference to be judged against — it changes the output on purpose.
What would settle it is a run where a *second* request demonstrably hits the
hash of a block the *first* has evicted, with the hit verified rather than
hoped for. Nothing here proves that case occurred.

## `serving-exposure` — point something real at it

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
