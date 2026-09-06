# TODO

Open work only. What is settled lives in `README.md`.

## `serving-knobs` — residency is specified in the wrong units

`VLLM_VIRTUALKV_BUDGET` is *full blocks resident per request*, which is a
convenient number for the code and a useless one for an operator. Nobody sizes
a deployment in blocks per request: they have a card, a model, a concurrency
target, and a context length.

Three problems, and the second is the one with a measurement behind it.

1. **Per-request is the wrong scope.** The quantity an operator controls is the
   *pool*, shared across whatever is running. A per-request budget silently
   becomes a global one multiplied by concurrency.
2. **A fraction of context would also be wrong**, which is worth stating
   because it is the obvious fix. The transport measurement says the fetch
   budget is *absolute* — roughly 543 tokens per decode step at 5% added
   latency on a 27B at fp8, whatever the context length — so "keep 10%
   resident" is cheap at 32K and unaffordable at 300K. What a policy has to
   bound is tokens fetched per step, and neither blocks-per-request nor a
   percentage expresses that.
3. **Tokens, not blocks.** Block size is an engine detail the operator did not
   choose.

Probably: a pool-level target in bytes or tokens, a per-step fetch ceiling, and
let the policy derive per-request budgets from what is running.

## `undersized-cache` — you cannot declare a context larger than VRAM

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

## `prefix-caching` — untested, and not merely untested

Every measurement so far ran with prefix caching off. Blocks are hashed and
registered as they fill, so an evicted block can be handed to another request
by hash. The interaction is genuinely open, not just unexercised.
