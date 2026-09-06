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
after warm-up. That is StreamingLLM — and **it is not the baseline a scoring
policy has to beat, because it is not a baseline at all**. Measured against a
`truncate` arm that runs no plugin and simply cuts the prompt to the same
tokens, on a real session at 33% residency:

| arm | agreement | drift | turns identical |
|---|---|---|---|
| truncate (no plugin, shorter prompt) | 0.4565 | **0.0016** | **4/10** |
| recency | 0.4606 | 0.0045 | 2/10 |
| quest | 0.4296 | 0.0054 | 2/10 |

Recency *is* truncation, and truncation is slightly better on the sensitive
measures. The whole apparatus buys nothing over a shorter prompt there. The
bar is beating truncation, and nothing here does it yet outside the needle.

That is a statement about the workload as much as the policy: the session is
chained four-turn conversations, so a 2048-token window holds the entire
current topic and nothing distant is required. A harness cannot reward keeping
context it never needs. The condition under which paging can pay is distant
retrieval — which is exactly what the needle has and this does not. It also means the restore path, the entire
difference between paging and eviction, is exercised by the tests and not by
the default policy.

The next real work is a demand signal: something resident that says "you will
want this block that is not here". **That signal is a 2-bit copy of the keys**,
and the shape of it has been measured rather than assumed. At a budget of 23 of
90 blocks on a real session, ranking blocks by the mass their pick holds:

| selector | mass captured |
|---|---|
| oracle (ceiling) | 0.8812 |
| **2-bit keys, one step stale — deployable** | **0.8811** |
| min/max bound | 0.6466 |
| **sinks + recency (what ships today)** | **0.7907** |
| recency without sinks | 0.3384 |

**Read the last two rows together.** Two blocks hold 46% of all attention
mass, and the shipped policy keeps them; a selector that does not looks 2.3x
worse than it is, and earlier versions of this table made that mistake.

**And read the whole table as context-length dependent**, because that turns
out to matter more. The gap between what recency gets free and what a scored
policy can reach grows with context: at 25% residency it is +0.097 over 112
blocks and +0.177 over 346, while recency's own capture decays (0.79 -> 0.71)
and the oracle holds (0.887 -> 0.889). A recency window is a fixed fraction
of a growing context. Every number above was taken at a few thousand tokens,
which is the regime least favourable to scoring and least like the one this
plugin exists for.

Three further results in that table. Quantized keys rank as well as the true keys, at
two bits, because ranking needs order and not accuracy. Quest-style min/max
bounds — which this project assumed were the answer — rank at 0.64 and cannot
prove anything either, overstating true mass by 9.4 orders of magnitude. And
the one-step staleness a policy is forced into by the protocol costs 0.55%
relative, so almost none of the advantage is lost to it.

The cost is ~12.5% of KV to carry that signal for the entire context, against
4 KiB per block per layer for the bounds it replaces.

**What it cannot do is guarantee.** Full attention at step t attends to all t
keys, so exact output plus eviction means streaming the whole context per
token. And skipping provably-negligible blocks does not rescue it: every layer
has a long thin tail, so at a per-head threshold of 1e-4 even a *single* layer
alone demands 89.3 of 90 blocks. Residency is therefore statistical, which puts
this on the same footing as every other KV scheme — it has to beat the
alternatives, or combine with them.

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
| `VLLM_VIRTUALKV_BUDGET` | `0` | residency per request: `64` blocks, `1024t` tokens, or `25%` of `max_model_len`; `0` evicts nothing |
| `VLLM_VIRTUALKV_POLICY` | `recency` | which blocks to keep |
| `VLLM_VIRTUALKV_SHOW_PENDING` | off | read blocks chosen for eviction but not yet freed |
| `VLLM_VIRTUALKV_SINK` | `2` | leading blocks always kept |
| `VLLM_VIRTUALKV_HOST_SLOTS` | `auto` | host tier size in blocks; derived from the engine unless set |
| `VLLM_VIRTUALKV_VERIFY` | on | run the residency guard |

Installed but unconfigured, this does nothing: it is loaded into every vLLM
process on the machine, and a plugin that patched attention because it happened
to be on the path would be a menace.

**`HOST_SLOTS` sizes itself, and that is a rule rather than a convenience.**
A budget is a promise that everything not resident is somewhere else, so the
tier has to hold the difference for every request that can be in flight:

    slots >= max_num_seqs * (ceil(max_model_len / block_size) - budget)

Every input is something the engine already knows, so the default is that
number rather than a figure the operator looks up — and re-looks-up whenever
the model, the card or the concurrency target changes.

*The general form, since it applies to knobs not yet written:* **when a setting
is checked against a hard threshold, its default should be derived from that
threshold.** Otherwise the arithmetic gets exported to whoever is running the
thing, who has strictly less information than the engine does, and it goes
stale silently. Setting `HOST_SLOTS` explicitly still works, and is warned
about only when it is too small.

Falling short degrades rather than corrupts — the worker *refuses* an eviction it
cannot back up and the block stays on the GPU, so the resident set quietly
exceeds the budget instead of a block being freed while its only copy is the
one being freed. That refusal heals: the policy re-chooses the block once the
tier has room. It stops being survivable once the startup context guard is
relaxed, because then the memory the relaxation was counting on is not there.

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
- **two control arms** (`tests/test_control_arm.py`). `full` evicts nothing and
  must be bit-identical to not having the plugin — necessary, and weak, since
  it cannot catch anything that only appears once eviction happens. **`churn`
  is the strong one**: it cycles blocks out and asks every one of them straight
  back, so each block is copied to the host, freed, reallocated elsewhere and
  copied back, while the model never loses sight of anything. It saves no
  memory and is not trying to. It is the only arrangement in which the
  manager's own evict-and-restore path can be held against an exact reference,
  because every policy that actually drops context changes the output on
  purpose and has nothing to be compared with. Both are bit-identical to no
  plugin, in tokens *and* logprobs.
- **the attention audit** (`VLLM_VIRTUALKV_AUDIT=1`) recomputes attention each
  step from the true keys — resident ones from the GPU, evicted ones from the
  host tier — and reports the mass that sat in blocks the policy did not have.
  It is the only honest way to refine a policy: transfer counters cannot tell a
  policy that fetches well from one that fetches constantly. Far slower than
  what it measures, and strictly for measurement runs.
- **transfer counters** — `copied_out`, `copied_in`, `missing_host_copy`,
  `evictions_refused`. The
  guard checks that what happens is *legal*, and a plugin that silently does
  nothing is perfectly legal; twice during development a counter reading zero
  was what caught it.

```bash
pytest tests/                                       # no GPU needed for most
VIRTUALKV_TEST_MODEL=<model> pytest tests/          # adds the control arm
tools/smoke.py <model> --budget 16                  # end to end, three arms
tools/quality.py <model> --budget 16                # the needle table above
tools/guard_selftest.py <model>                     # the guard, faulted
tools/gsm8k_turns.py <model> --budget 25%           # multi-turn, scored
```

`gsm8k_turns.py` is the multi-turn instrument, and it is sensitive to the
opposite thing from `quality.py`. Each GSM8K question needs the exemplars at
the start and the question at the end and nothing between, so no sane policy
loses accuracy on it — which makes a drop a *mechanism* signal rather than a
policy one, while the needle test is policy-sensitive by construction. Turns
are extended with the gold answer, never the model's own, so every arm sees one
conversation; feeding back what the model said would fork the context the first
time two arms disagreed and every later turn would compare different
conversations rather than different residency.

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

## Reading an accuracy number from this

Paging is decode-only, so a quality measurement taken now is an **optimistic
bound**, and it is worth being clear about that before producing one rather
than after. A request's prompt attends to itself in full during prefill; only
generation runs against a restricted set. Adding prefill residency can lower
quality and cannot raise it, so a policy that looks bad here will not look
better later.

That does not make the number uninteresting — it is the number for the shape
most real work has, a long prompt and a short generation, where what matters is
whether the few generated steps can reach the right blocks. Recency losing the
needle at 12% residency is a real finding under exactly this reading.

Four concurrent requests have been exercised (all four bit-identical under
`churn`, zero guard violations over 88 steps, including the exclusivity check
that catches two requests sharing a block). Contexts beyond a few thousand
tokens have not.

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
