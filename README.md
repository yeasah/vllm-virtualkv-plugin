# vllm-virtualkv-plugin

Virtual memory for a vLLM KV cache. A request's context may be larger than the
GPU can hold: the rest lives in host memory and comes back when it is wanted.

vLLM's PagedAttention virtualised the *allocation* of KV blocks. This
virtualises their *residency*. The block table is a page table, the per-step
view is address translation, and the host tier is swap.

**The point is what being wrong costs.** Every KV compression method either
shrinks the cache or discards from it, so a mistake is unrecoverable — a needle
at an evicted position is gone, and that is why eviction methods have to be
timid. Nothing here is discarded, only moved, so a mistaken residency decision
is a *stall* rather than a wrong answer. That is the only mechanism that
decouples declared context from VRAM without paying in accuracy, and it is what
lets a policy be aggressive.

## Status

The machinery works and is measured. **The shipped policy is not the one that
makes it pay**, and that is the honest headline.

On Llama-3.2-1B with 2048 tokens of context and a budget of 16 of 129 blocks —
**12.4% resident** — a needle planted at a known block:

| policy | needle | output |
|---|---|---|
| none (no plugin) | found | the reference |
| `full` (plugin, evicting nothing) | found | **bit-identical to the reference** |
| `recency` | **lost** | diverges at step 2 |
| `oracle` (told which block holds it) | found | **tokens identical to the reference** |

So at 12% residency the mechanism reproduces a full-context answer token for
token when the right blocks are kept, and loses it entirely when they are not.
The mechanism is not the limit; the policy is.

`recency` losing it is not a defect, it is what recency *is*: its window only
slides forward, so it never asks for a block back and its fetch rate is zero
after warm-up. That is StreamingLLM — useful, shippable, and the baseline a
scoring policy has to beat. It also means the restore path, the entire
difference between paging and eviction, is exercised by the tests and not by
the default policy.

The next real work is a demand signal: something resident that says "you will
want this block that is not here". Quest-style per-block key bounds are the
shape of it, and they are not free — roughly 4 KiB per block per layer in fp16
against a 32 KiB block, out of the same budget.

## Use

```bash
pip install -e .
export VLLM_VIRTUALKV=1
export VLLM_VIRTUALKV_BUDGET=64        # resident full blocks per request
export VLLM_VIRTUALKV_POLICY=recency   # recency | stress | full | oracle
vllm serve <model>
```

| variable | default | meaning |
|---|---|---|
| `VLLM_VIRTUALKV` | off | must be set, or the plugin does nothing at all |
| `VLLM_VIRTUALKV_BUDGET` | `0` | resident full blocks per request; `0` evicts nothing |
| `VLLM_VIRTUALKV_POLICY` | `recency` | which blocks to keep |
| `VLLM_VIRTUALKV_SINK` | `2` | leading blocks always kept |
| `VLLM_VIRTUALKV_HOST_SLOTS` | `1024` | host tier size, in blocks |
| `VLLM_VIRTUALKV_VERIFY` | on | run the residency guard |

Installed but unconfigured, this does nothing: it is loaded into every vLLM
process on the machine, and a plugin that patched attention because it happened
to be on the path would be a menace.

`BUDGET=0` with `POLICY=full` is not a no-op — it wires everything up and
evicts nothing. That is the control arm, and its output must be identical to
running without the plugin.

## Verifying it, which is not optional

The failure mode here does not crash. A block freed while something still
points at it means attention reads another request's KV: plausible text,
slightly wrong, for the rest of the generation. If that lands inside a quality
measurement, the damage becomes a design decision. So three layers, and a
result that does not say which were active is not evidence:

- **the residency guard** (`VLLM_VIRTUALKV_VERIFY`) checks, per step, that the
  view names nothing the request does not own, that this step's key lands in
  the last resident block, that the length matches what the policy intended,
  and that two requests' resident sets do not overlap. Three of the four need
  nothing but the step being executed and so run in a deployment; the fourth
  needs the scheduler and therefore an in-process engine.
  `tools/guard_selftest.py` injects each fault and shows it caught.
- **the control arm** (`tests/test_control_arm.py`) — the plugin evicting
  nothing must be bit-identical to not having it. Necessary and not sufficient:
  it cannot catch a bug that only appears once eviction happens.
- **transfer counters** — `copied_out`, `copied_in`, `missing_host_copy`. The
  guard checks that what happens is *legal*, and a plugin that silently does
  nothing is perfectly legal; twice during development a counter reading zero
  was what caught it.

```bash
pytest tests/                                       # no GPU needed for most
VIRTUALKV_TEST_MODEL=<model> pytest tests/          # adds the control arm
tools/smoke.py <model> --budget 16                  # end to end, three arms
tools/quality.py <model> --budget 16                # the needle table above
tools/guard_selftest.py <model>                     # the guard, faulted
```

## What it reaches past

Three pieces sit on documented extension points: the spec and manager via
`KVCacheSpecRegistry.register`, the view inside an attention metadata builder,
and startup via `vllm.general_plugins`. Two are patches, both in
`vllm_virtualkv/integration.py` so the answer to "what does this monkeypatch"
is one file:

- **`Attention.get_kv_cache_spec`**, to choose the paged spec per layer.
  `customize_spec` looks like the intended hook and is even called, but its
  result is only used to size a page while the layer builds a plain
  `FullAttentionSpec` anyway. Its docstring calls itself a temporary
  compatibility API and says the backend will build the spec directly
  (vllm#42449) — so this one has an upstream expiry date.
- **`GPUModelRunner.prepare_attn`**, to make the worker's block table equal the
  manager's mapping before `compute_slot_mappings` reads it positionally.
  Restored blocks reach the worker through the *append* channel, so its row
  outgrows the context. There is no seam: the protocol can say "here are more
  blocks" and cannot say "this block now lives at index `i`". Closing it means
  a field on `SchedulerOutput`.

## Limits

- **Decode only.** A request still prefilling is left alone. Compaction is
  mask-safe — `flash_attn_varlen_func` aligns the causal mask bottom-right, so
  dropping prefix keys moves both sides of the bound — but a prefill chunk's
  write span is a *run* of blocks rather than one, block order becomes
  load-bearing where at decode it carries no meaning, and prefix-cache hashing
  is live during prefill so an evicted block can be handed to another request
  by hash. None of that is measured. The cost: peak residency includes the
  whole prompt, so this bounds the decode footprint and not the prefill peak.
- **Prefix caching off.** Untested with it on; see above for why it is not
  merely untested but genuinely open.
- **Full-attention layers only.** Sliding-window and other specs are passed
  through untouched: their kernels rebuild key position from the block's index
  in the row, which compaction breaks. Measured, not assumed.
- **Single GPU, eager.** No TP, no MLA, and CUDA graphs are unexercised.

## Where this came from

The measurements that justify the design were taken in
[`vllm-exl3-plugin`](https://github.com/yeasah/vllm-exl3-plugin) and are written
up in its `docs/kv-pager.md`: that block order carries no positional meaning at
decode, that the engine will attend to a subset of a request's blocks, that a
block survives being destroyed on the GPU and restored from host memory into a
*different* physical block bit-identically, and that explicit DMA does not care
about locality — `copies x 1.30 us + bytes / 54 GB/s` — which is what makes
block-granular residency affordable.
