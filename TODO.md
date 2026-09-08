# TODO

Open work only. Findings live in `docs/`; what is settled and shipped lives in
`README.md`.

**Scope, since it narrowed.** The demand-signal direction is closed — a scored
policy lost fourteen of fourteen configurations and an oracle with perfect
knowledge of attention mass ties a policy that ignores it. That is written up
in `docs/demand-signal.md` and is not worth reopening without a new idea about
*what quantity to score*. What remains is the capacity argument, which never
depended on it: **let a request declare a context larger than VRAM, and pay
for it in stalls rather than quality.**

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
   Only meaningful with a policy that fetches, and there is no longer one to
   build it for — see `docs/demand-signal.md`. Park it.
3. **Three knobs are denominated in blocks, and a block is not a fixed size.**
   `sink`, `recent` and `host_slots` take block counts, so each means
   something ~500x different on a hybrid (16.5 MiB per block) than on a dense
   model (32 KiB). That has already caused an OOM kill, a confounded sweep,
   and a budget entirely consumed by sinks. They should take the `1024t` and
   `25%` forms `budget` accepts. **`sink` should also default near a third of
   the budget rather than 2 blocks** — the measured optimum, worth 13-31% and
   more than any policy change (`docs/sink.md`).

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

## `hybrid-support` — V1 runner, or an honest refusal

Most of this works: the spec patch reaches the right layers, group resolution
handles a four-group model, the guard validates, `full` is bit-identical and
an oracle finds the needle. Details in `docs/hybrid.md`.

The blocker is that vLLM has two GPU model runners with the same class name
and picks between them per model — and a hybrid whose architecture is not in
`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (8 entries, no Qwen) gets the one this
plugin does not patch. The hook installs, never fires, and the plugin loads
and silently does nothing. `fired_or_raise()` catches it now, but catching is
not supporting.

Either hook the V1 runner too, or refuse at startup with the
`VLLM_USE_V2_MODEL_RUNNER=1` workaround in the message. Hybrids are what gets
deployed, so one of the two is required.

## `pin-the-audit-path` — it has been wrong twice

`audit.py` and `workingset.py` reconstruct attention in Python, and both bugs
found in them changed conclusions: a missing `1/sqrt(head_size)` scale, and a
tail block excluded from the softmax. Neither crashed; both produced numbers.

Nothing pins that path against the model's own attention output. It should:
recompute one step both ways and require agreement. Cheap, and several
results now rest on it.
