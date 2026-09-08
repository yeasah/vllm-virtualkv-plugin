# TODO

**Suspended, 2026-09-08. There is no open work.** Both directions this was
built for are closed — see "Why it is suspended" in `README.md`, with the
accounts in `docs/demand-signal.md`, `docs/capacity.md`,
`docs/eviction-survey.md` and `docs/granularity.md`.

This file now records only what would justify reopening it, and what was left
unfinished so that a revisit does not start by rediscovering it.

## What would justify a revisit

None of these is a matter of finishing something here. Each is a change from
outside the project that would move a bound rather than an estimate.

- **A demand signal that is not a function of the current cache state.** Task
  structure known before the query — tool-call boundaries, explicit retrieval
  intent — is the only shape suggested so far. Everything scored from the cache
  is a ranking, and `setoracle` bounds rankings.
- **A mechanism that changes what is *in* the cache rather than which parts
  survive** — merging, or recomputation from source tokens. Selection is
  closed; construction was never tested.
- **Upstream decoupling of attention and mamba page sizes.** Block size on a
  hybrid is forced to 528/1056 by alignment. Sub-64 outcome data is the one
  unmeasured regime, and nothing on a hybrid can act on it while that floor
  stands. Note the second wall behind it: acting on a positive result needs
  intra-block compaction, which forfeits prefix caching inherently.
- **An interconnect that is not PCIe.** The capacity arithmetic is 86% of PCIe
  5.0 x16's theoretical ceiling, and the 16 GiB cards that create the use case
  are mostly x8. Nothing in the mechanism is the bottleneck.

## Left unfinished

Real work, parked rather than abandoned. Ordered by how much a revisit would
regret skipping it.

**`pin-the-audit-path` — it has been wrong twice.** `audit.py` and
`workingset.py` reconstruct attention in Python, and both bugs found in them
changed conclusions: a missing `1/sqrt(head_size)` scale, and a tail block
excluded from the softmax. Neither crashed; both produced numbers. Nothing pins
that path against the model's own attention output. It should: recompute one
step both ways and require agreement. Several recorded results rest on it.

**`hybrid-support` — V1 runner, or an honest refusal.** vLLM has two GPU model
runners with the same class name and picks between them per model; a hybrid
whose architecture is not in `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` gets the
one this plugin does not patch, and the plugin then loads and silently does
nothing. `fired_or_raise()` catches it, but catching is not supporting.
Hybrids are what gets deployed. Details in `docs/hybrid.md`.

**`serving-knobs` — two defects, both cheap.** `sink`, `recent` and
`host_slots` take *block* counts, so each means something ~500x different on a
hybrid (16.5 MiB per block) than on a dense model (32 KiB); that has already
caused an OOM kill, a confounded sweep, and a budget entirely consumed by
sinks. They should take the `1024t` and `25%` forms `budget` accepts. And
**`sink` should default near a third of the budget rather than 2 blocks** — the
measured optimum, worth 13-31% and more than any policy change
(`docs/sink.md`).

**`prefix-caching` — one case never demonstrated.** A pager relocates rather
than modifies, so a block's hash stays truthful; `churn` is bit-identical with
caching on across a growing conversation. What was never shown is a *second*
request hitting the hash of a block the *first* has evicted, with the hit
verified rather than hoped for. Also unimplemented: residency accounting should
be per-block-refcount rather than per-request, since a request holding a shared
block is not spending its budget on it in any sense that matters.

## Closed, so that they are not picked up again

**`undersized-cache` and `prefill-residency`.** These were the capacity path's
remaining build, and capacity is closed on arithmetic rather than on effort —
building them would not change the numbers in `docs/capacity.md`. Two findings
from that analysis are worth carrying anyway, since they are properties of the
configuration rather than of this plugin:

- Prefill residency is a *mechanical* consequence of declaring a context larger
  than the KV cache, not a design choice: an uncached prompt larger than VRAM
  arrives eventually and OOMs. Every benchmark in this project avoided it only
  by never running the configuration the plugin exists for.
- Under prefix caching a prefill residency error is **durable**, where a decode
  one is a stall. Tokens prefilled against a windowed view are encoded that way
  and cached, and every later turn reuses them. It inverts the property this
  design was built on, and it passes every guard, because nothing about it is
  illegal.
