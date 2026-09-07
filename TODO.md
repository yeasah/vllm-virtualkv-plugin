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

## `both-ceilings-tie-recency` — at 40 units of freedom

Same configuration throughout (block 2112, sink 2, budget 12672t, 163 turns,
~84k context):

| arm | agreement | ranks on |
|---|---|---|
| recency | **0.2374** | position only |
| massoracle | 0.2372 | true attention mass |
| impactoracle | 0.2052 | true marginal output shift |
| quest | 0.2256 | bounds estimate of mass |

**Perfect knowledge of attention mass exactly ties a policy that ignores it,
and perfect knowledge of the marginal output shift is worse.**

Mass is settled: there is no headroom in that quantity, so the bounds, the
2-bit summary and quest were all estimating an empty target.

The impact result is more interesting and probably indicts the objective
rather than the idea. `m*(o - v_b)/(1 - m)` is the leave-*one*-out shift, and
it is being used to pick a set that drops ~85% of blocks. Those effects do
not compose -- the renormalisation interacts -- so a set assembled from
individually high-impact blocks can be much worse than a contiguous window.
The identity extends to sets, `m_D*(o - v_D)/(1 - m_D)`, so a greedy forward
selection recomputing the residual after each pick would be a real set-oracle
at O(k*n) per step rather than O(n). Untried.

**The granularity caveat governs all of it.** At 2112 tokens a whole 84k
context is ~40 blocks. "No selection criterion beats contiguity" and
"selection cannot be expressed at 40 units of freedom" are the same
measurement here, and nothing in this table distinguishes them. The premise
question is open until it is asked at a granularity that can express an
answer -- see `granularity-rig`.

## `granularity-rig` — a full-attention model, to separate two conclusions

Every selection result so far is conditioned on an enormous block: 528 tokens
on the hybrid at bf16, 2112 in the runs that produced the oracle numbers. At
2112 a whole 84k context is ~40 blocks, so "which blocks you keep barely
matters" and "selection cannot be *expressed* at this granularity" are
indistinguishable. A full-attention model has no mamba page to match, so the
block size is a free parameter and the two can be separated.

Surveyed locally for pure full attention (no sliding window, no linear
layers) at long context:

| model | ctx | KV/token bf16 | KV at full ctx | note |
|---|---|---|---|---|
| **SmolLM3-3B** | 65536 | 72 KiB | 4.5 GiB | all 36 layers full_attention |
| MiniCPM5-1B | 131072 | 24 KiB | 3.0 GiB | 24 layers, 2 KV heads; exl3 local |
| Llama-3.2-1B | 131072 | 32 KiB | 4.0 GiB | the original needle model |
| AFM-4.5B | 65536 | 72 KiB | 4.5 GiB | 9 GiB of weights at bf16, too tight |
| Phi-4-mini | 131072 | 128 KiB | 16 GiB | KV too large |

Everything else long-context in the cache is sliding-window or
linear-hybrid and would reproduce the same problem.

**SmolLM3 has 9 NoPE layers and is therefore the wrong instrument here.**
`no_rope_layers` is 27 with RoPE and 9 without, every fourth. This plugin's
foundational result -- block order carries no positional meaning at decode --
holds *because* RoPE is baked into keys at write time. Layers with no
positional encoding respond to eviction differently, so a granularity study
there averages two behaviours and cannot attribute its result. Uniform RoPE
is a requirement, not a preference.

That leaves the Llama-3.2 line, which is uniform RoPE throughout:

- **Llama-3.2-3B** -- 28 layers, 8 KV heads, hd 128 = **112 KiB/token**, so
  64k costs 7 GiB of KV plus ~1.8 GiB of weights at 4bpw. 128k would be 14
  GiB and does not fit, making 64k the ceiling and **4096 blocks at block
  16**. Not cached: needs a download and a quant. This is the one to quote.
- **Llama-3.2-1B** -- already here as `turboderp/Llama-3.2-1B-Instruct-exl3`
  (3.0 GiB cached), 131072 context, 32 KiB/token, so 128k fits in 4 GiB and
  block 16 gives **8192 blocks**. It is also the original needle model, the
  one case where selection ever paid. Weak, but the granularity question asks
  whether damage varies with block size, which a small model still exhibits,
  so it can run the shape immediately.

Suggested split: 1B now for the shape, 3B for the number.

**Qwen/Qwen3-4B-Thinking-2507 is the pick.** `Qwen3ForCausalLM`,
`use_sliding_window: False`, no `layer_types`, no `no_rope_layers` -- uniform
full attention with uniform RoPE, and the same family as the original dense
work so earlier numbers stay comparable. 262144 declared, but 36 layers x 8
KV heads x 128 head_dim is 144 KiB/token, so the reachable ceiling here is
64k (9 GiB KV + ~2.4 GiB at 4bpw), giving **4096 blocks at block 16** --
a hundredfold more than the hybrid runs.

It also *thinks*, which matters: decode volume has already reversed a result
in this repo (138 tokens/turn against 617), and its benchmark standing blunts
the "the model could not do the task" objection to whatever comes out.

*Superseded:* SmolLM3-3B at 64k: ~6.2 GiB weights + 4.5 GiB KV, and
at block 16 that is **4096 blocks** -- a hundredfold more decision units than
the hybrid runs -- with enough model to blunt the capability objection.
MiniCPM5-1B is the cheap fallback for 128k or very fine blocks, at 1B.

Note the transcripts run ~84k tokens, past SmolLM3's 64k, so a run there
needs fewer turns or a shorter trajectory. The corpus has 2954 agent turns
across 53 files to choose from.

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

### The long-context rig, measured 2026-09-06

`~/ckpt/Qwen3.5-9B-exl3-4.00bpw-bq`, one 16 GiB card, `VLLM_USE_V2_MODEL_RUNNER=1`:

    max_model_len 131072, bf16 KV, util 0.85   ->  initialises
    weights 5.48 GiB, KV 5.57 GiB              ->  ~11 GiB of 15.5
    KV capacity 177,332 tokens                 ->  33 KiB/token, all groups
    block size 528                             ->  248 blocks at full context

**Do not use fp8.** It halves the attention page, so vLLM doubles the block
size to 1056 tokens to keep the mamba page equal -- buying memory at the cost
of granularity, which is the wrong trade for this design and the opposite of
what one expects. Every earlier estimate here assumed 528 and was wrong under
fp8.

**Three of the four groups are MambaSpec at the same page size**, so ~75% of
the KV allocation is linear-attention state. Mamba state is per-*request*, not
per-528-tokens, so most of that is likely never used -- attention alone would
be ~8 KiB/token. Worth checking whether vLLM populates those groups sparsely
before calling it waste, but it is where the memory is.

248 blocks is the first configuration with enough granularity for an
aggressive budget: 10% is 25 blocks, against the 1.5 blocks that made every
step violate the guard. It is the rig for the open questions -- the redundancy
threshold (many small holes vs few large at equal residency), the sink result
free-running, and the recency-vs-oracle gap which widens with context length
(+0.049 at 112 blocks, +0.114 at 346).

*Caveat on filling it:* the longest trajectory is ~121k raw and ~60k stripped,
so 128k needs chained trajectories, which reintroduces the topic-boundary
artifact that made the UltraChat session undemanding. Prefer one long
trajectory at whatever length it reaches.

### Broken under pressure on a hybrid, 2026-09-06

An aggressive free-running run -- Qwen3.5-9B-exl3, fp8 cache, 26 turns, 3-block
budget (1584t, ~13% residency), `sink=1` -- **violates the guard on every
step**: `guard 9932/9932`. `recency` and `quest` returned bit-identical output
(same agreement, same drift, same 112 transfers), so both degraded into the
same broken state. Its numbers are artifacts and must not be quoted.

The hybrid needle run was clean at `guard 0` over 22 steps, so this is a
configuration boundary rather than a general hybrid failure. What differs:
fp8 cache dtype, `sink=1`, a 3-block budget, and a much longer context.
Suspect first the `length` check -- `seq_len = (len(resident) - 1) *
block_size + tail_count` assumes an arithmetic that is far more sensitive at
528-token blocks than at 16 -- then fp8 changing the cache layout under
`block_keys`.

Re-run with `VLLM_VIRTUALKV_VERIFY=1` and print `by_check` to see which of the
four is firing; `session_turns.py` reports only counts, which is why this
needed a separate look and should be fixed.

**Forcing does not engage on this model at all.** The assertion added the same
day fired -- "forcing did not take on turn 0 ... the sampler seam is wrong or
a prefill chunk ate a step" -- so `--force` is dense-only until the tap's
`logits.shape[0] == 1` guard is reconciled with a model declaring MTP layers.

### Working on a hybrid, 2026-09-06

`tools/quality.py` on Qwen3.5-9B-exl3, 16k context, 31 blocks of 528 tokens,
budget 8 (~26% residency), needle at block 10, V2 runner forced:

| arm | needle | transfers |
|---|---|---|
| off | found | reference |
| `full` | found, **bit-identical** | — |
| `recency` | lost, diverges at step 3 | 23 out |
| `quest` | lost; **did not select the needle block** | 24 out, 2 in |
| **`oracle`** | **found** | 23 out |
| `oracle_late` | lost | 24 out, 1 in |

Zero guard violations on every arm.

**"Coarse blocks do not break selection" was wrong and is withdrawn.** The
needle showed only that *one* block can be retained, which is the single
question granularity does not affect; every other hole it makes is invisible
to a test whose output is one number.

Measured properly -- token budget fixed, oracle selection, only the block
size varying, by grouping native 16-token blocks into super-blocks:

| block size | blocks kept | mass captured |
|---|---|---|
| 16 | 86 | 0.8892 |
| 32 | 43 | 0.8711 |
| 64 | 21 | 0.8413 |
| 128 | 10 | 0.8103 |
| 256 | 5 | 0.7466 |

A monotonic 16% loss from 16 to 256 tokens per block, extrapolating to about
0.70 at 528. Coarse blocking costs roughly a fifth of what a fine-grained
pager could capture, and this is the *best* case: an oracle ranks super-blocks
by their true summed mass, where a real policy estimates from a summary.

It is a smooth decline rather than a cliff, and mass has repeatedly failed to
predict output damage here, so the reading is "meaningful, needs an end-to-end
check", not "fatal". But it does put the block-based design back in question
for hybrids, where 528 tokens is forced.

The scale argument is independent and does not rest on the mass numbers: at
256k context there are only ~485 blocks, fewer decision units than the 1152
head-layer pairs that must all agree on each one. A 528-token span is a torn
page, not a dropped vowel.

*Artifact to fix before quoting the 528 row:* `_summary_summary` intersects
keys across steps, and early steps have too few blocks to form super-blocks at
the largest factor, so that row is dropped. Re-run from a longer starting
context rather than extrapolating.

**And the whole stack works on a hybrid**: spec patch, group resolution
across four groups with mamba state interleaved in `kv_caches`, view
application, guard, host tier, transport, all against an EXL3 checkpoint.

`quest` failing to select the needle block is unchanged from the dense model,
so the policy gap is not an artifact of the earlier setup.

### First contact, 2026-09-06

Partly working, with three blockers found by running it.

**Done and verified on Qwen3.5-9B-exl3:** the spec patch reaches exactly the
8 full-attention layers; `groups.resolve` finds the paged group; the guard
validates against the right allocation table; 22 decode steps ran with **zero
guard violations, all four checks active**.

The real layout justified the resolver more than its design did: **four**
groups, not two — 24 linear layers split into three `MambaSpec` groups — with
the paged group at **index 3**. A hardcoded `[0]` pages mamba state silently.
The guard needed the same fix: it takes a `group` argument that the worker was
leaving at its default, so every paged block read as unowned.

**Blocker 1: vLLM picks between two GPU model runners per model, not per
version.** `use_v2_model_runner` is a config property (forced for PCP,
dspark, multi-KV-group DFlash; overridable by `VLLM_USE_V2_MODEL_RUNNER`) and
this hybrid defaults to **V1** — `vllm/v1/worker/gpu_model_runner.py`, not the
`vllm/v1/worker/gpu/model_runner.py` the plugin patches. So the hook installs
and never fires: the plugin loads, reports nothing, and pages nothing. That is
the worst available failure mode and it was caught only by `smoke.py` saying
"nothing moved". Either support V1 or **detect and refuse**.

**Blocker 2: a block is 528 tokens on this model.** vLLM sets the attention
block size so the attention page is >= the mamba page. That is 33x the block
size every measurement in this repo has used, and it changes the units the
plugin thinks in:

- a block-count budget means something 33x different, so `--budget 16` is 8448
  tokens here and can never bind. The token and percentage budget forms stop
  being a convenience and become the only safe way to express it.
- one block is 4 kv heads x 528 tokens x 256 dims x 2 x 2 bytes = 2 MiB per
  layer, **16.5 MiB across the 8 paged layers**, against 32 KiB on Qwen3-8B.
  Anything sized in blocks — `required_host_slots` above all — is now sizing
  in 16.5 MiB units. **The autosize rule needs a byte dimension**: the
  threshold it derives from is a block count, but the resource it spends is
  bytes.
- residency granularity coarsens by the same factor, which cuts against the
  9x improvement in the union. Which dominates is unmeasured.

**Blocker 3: a reproducible OOM kill (exit 137) when the plugin is enabled.**
`tools/quality.py` on the hybrid dies with ~17.9 GB of `shmem-rss` (kernel
log), and it reproduces with 20 GB free and no swap pressure — an earlier
guess that this was environmental is **wrong** and was withdrawn.

What is known:

- GPU KV sizing is byte-identical with and without the plugin (0.98 GiB,
  16,102 tokens), and the engine finishes initialising. So it is host-side.
- The host tier is *not* the cause: instrumented at **1 slot, 16.5 MiB,
  0.02 GiB total**. Its `pin_memory=True` is page-locked and so unswappable,
  which makes it an aggravator under pressure but nowhere near 17 GB.
- A minimal script with the plugin enabled on the same model runs clean.

What differs between the clean run and the failing one, none yet isolated:
`gpu_memory_utilization` 0.85 vs **0.55**, prefix caching on vs **off**, and
`max_model_len` fixed 2048 vs derived from the prompt. Bisect these first.

The suspicion worth testing: the paged spec subclass may perturb the
attention/mamba page-size equalisation (`interface.py` pads the mamba page to
match), and `PagedAttentionSpec.merge` has already once dropped fields because
the base rebuilds from an explicit list. A wrong page size would not change
the reported GPU total while still making something host-side enormous.

**A reporting weakness that cost time here**: `quality.py` prints
`stdout[-1500:]` and `stderr[-2500:]` on failure, which are blank when a run
ends in progress-bar output, so it reports "arm failed" and shows nothing.

**Still to do regardless: make the hook verify it *fired*.** `install()`
checks that it patched something; nothing checks that it ran. Three separate
bugs in one session share that shape — the forcer on a superseded `Sampler`
class, the pager on a superseded runner class, and a budget that never bound.
A counter asserted at first request completion catches all three at once.

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

## `sink-allowance` — the largest single effect measured, and undertuned

Block size swept end to end on a dense model at a fixed *token* budget
(2048t), forced decoding, entropy-banded. The first sweep had `sink=2`
**blocks**, which is 32 tokens at block size 16 and 1024 at block size 512 --
so it compared budget allocations, not granularities. Equalising the sink at
1024 tokens:

| block size | KL | flip | `certain` flip |
|---|---|---|---|
| 16 (sink 64) | 0.1270 | 0.0504 | 0.0210 |
| 128 (sink 8) | 0.1231 | 0.0444 | 0.0171 |
| 512 (sink 2) | 0.0946 | 0.0377 | 0.0139 |

**Raising the sink from 32 to 1024 tokens cut KL from 0.3772 to 0.1270 at
block size 16** -- a 3x reduction in output damage from one knob at the same
budget, larger than any policy effect measured to date. `sink` defaults to 2
blocks and is denominated in blocks, so it silently means 32 tokens on one
model and 1056 on a hybrid. It should take a token form like `budget` does,
default far higher, and warn when it eats a large share of the budget.

**Granularity showed no penalty and no cliff.** With allocation matched, 16
and 128 are indistinguishable and 512 is slightly *better*. The mass curve
predicted a fifth of capturable mass lost by 528 tokens and none of it
appeared in the outcome -- the proxy misled again, in the opposite direction
this time. The residual coarse advantage is likely an artifact: the partial
tail block is always resident and is not charged against the budget, so
bs=512 gets up to 512 free recent tokens against bs=16's 16.

Not a test of the redundancy hypothesis (that small holes are papered over by
surrounding context while large ones destroy the context that would repair
them). That predicts a *threshold*, and this run sits at ~50% residency on a
~4000-token context, comfortably on the safe side of any threshold. Re-run at
10% or less before concluding anything about coarse blocks under pressure.

**Audit every knob denominated in blocks.** `budget` accepts `1024t` and
`25%`; `sink`, `recent` and `host_slots` do not, and all three have now
caused a bug: host_slots OOM-killed the process on a hybrid, sink confounded
this sweep and consumed the entire budget in the hybrid needle run.

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

## `sink-optimum` — about a third of the budget, and the default is far off

Swept on the long rig (Qwen3.5-9B-exl3, druid-15402, 163 turns, ~84k context,
12672t budget, 528-token blocks, `recency`):

| sink | tokens | % of budget | agreement | drift | turns identical |
|---|---|---|---|---|---|
| 1 | 528 | 4% | 0.2312 | 0.026007 | 24 |
| 2 | 1056 | 8% | 0.1993 | 0.026666 | 15 |
| 4 | 2112 | 17% | 0.2563 | 0.025229 | 30 |
| **8** | **4224** | **33%** | **0.2604** | **0.023756** | **35** |
| 16 | 8448 | 67% | 0.2074 | 0.028295 | 25 |

Peak at a third of the budget on the front of the context. Drift bottoms and
identical turns peak at the same point.

**The 3x from the dense sweep is corrected to ~13%.** That comparison ran 32
tokens against 1024 -- almost all of it was escaping a near-zero default, not
a large effect. Here sink=1 is already 528 tokens and the remaining headroom
over it is 13% (31% over the worst setting).

**Do not theorise the sink=2 dip.** Agreement is chaotic in the
configuration: a turn scores its identical *prefix*, so one early flipped
token discards that whole turn, and a one-block change can trigger it. All
three columns dip together because all three are driven by the same few
divergences -- one cause, not three confirmations. Wants a finer grid (3, 5,
6) before it means anything.

**Actionable:** `sink` defaults to 2 blocks, which is 8% of budget here and
32 tokens on Qwen3-8B. It should be a fraction of the budget expressed in
tokens, defaulting near a third, with the same `1024t` / `25%` forms `budget`
already takes. Same fix `recent` and `host_slots` need.

## `quantisation-fights-granularity` — no escape hatch downward

On a hybrid the attention page must be at least the mamba page, so shrinking
the attention page makes vLLM *grow* the block:

| KV dtype | block size |
|---|---|
| bf16 | 528 |
| fp8 | 1056 |
| tq4 | 2048 |

Quantising to buy KV headroom coarsens residency, which is the opposite of
what one wants and runs the wrong way to escape. The only routes to finer
granularity are decoupling the two page sizes upstream, or sub-block
residency underneath vLLM's block.

## `positional-prior` — measured, no effect

Swept over block size with sink pinned at 4224 tokens (the measured optimum)
by moving it inversely: 528/8, 1056/4, 2112/2.

| block | recency | quest | quest+prior | prior delta |
|---|---|---|---|---|
| 528 | **0.2604** | 0.2429 | 0.2397 | -0.0032 |
| 1056 | **0.2405** | 0.2232 | 0.2226 | -0.0006 |
| 2112 | **0.2374** | 0.2256 | 0.2312 | +0.0056 |

All three deltas are +/-0.006 against a noise floor of ~0.03 (the size of the
unexplained sink=2 dip), so this resolves nothing. The prediction on record --
that it would help most at 528 where there are the most blocks to reshape, and
least at 2112 -- was wrong in direction as well as magnitude.

**Recency is 12 for 12** against the scored policy: nine across budgets and
block sizes, three here.

**Granularity, done properly this time.** Holding sink in *tokens* rather than
blocks removes the confound that made the earlier sweep unusable and reverses
its answer -- recency is monotonic in fineness: 0.2604 / 0.2405 / 0.2374 for
528 / 1056 / 2112. That agrees with the mass-capture curve and with the
original intuition that 528-token holes are too coarse. Hold it loosely: the
528->1056 gap is 0.020 and 1056->2112 is 0.003 against a ~0.03 floor, and the
baselines differ (22129 / 22146 / 22420 generated).

**The instrument's resolution is now the binding constraint.** At 163 turns
agreement cannot resolve below ~0.03, because a turn scores its identical
prefix and one early flip discards the whole turn. Anything smaller needs
per-step scoring, which means forcing -- and forcing does not engage on this
model (see the MTP note). `massoracle` OOM'd, so the ceiling that would
calibrate all of these numbers is still missing.

## `positional-prior` — the original idea, for reference

`sink` and `recent` express the U-shape of attention as two hard
reservations counted in blocks. `policy.positional_prior` expresses it as a
scale-free weight in fractions of the context, and the worker can multiply a
scored ranking by it (`VLLM_VIRTUALKV_PRIOR`, 0 disables, off by default).

The difference is what it makes possible rather than what it forbids: a fence
never offers a middle block at any price, while a multiplier lets a strong
enough score win one. `quest` lost nine for nine partly by spending budget
through the middle, so making the middle expensive rather than forbidden is
the cheapest thing left to try on the scored side.

Note the standalone version is already being swept: with `recency` the budget
is exactly sink + recent, so sweeping sink at fixed budget *is* sweeping the
front/back split of a discrete U-curve. The prior only adds something in
combination with a demand signal.

Untested. It is a positional function, so like recency it needs no fetches of
its own.

## `quest-is-a-net-negative` — nine for nine at long context

Qwen3.5-9B-exl3, `apache__druid-15402` at 163 turns, context to 334,718 chars
(~84k tokens), 22k generated, `max_model_len` 262144, util 0.95, sink 1 block:

| budget | block | recency | quest | quest out/in | recency out/in |
|---|---|---|---|---|---|
| 6336t | 528 | **0.1708** | 0.1419 | 20981/7539 | 13437/0 |
| 6336t | 1056 | **0.1377** | 0.1202 | 9554/2884 | 6669/0 |
| 6336t | 1584 | **0.1830** | 0.1679 | 5490/1080 | 4410/0 |
| 12672t | 528 | **0.2312** | 0.2052 | 21475/9914 | 11559/0 |
| 12672t | 1056 | **0.1880** | 0.1576 | 10611/4876 | 5734/0 |
| 12672t | 1584 | **0.2607** | 0.2224 | 6375/2585 | 3791/0 |
| 25344t | 528 | **0.3618** | 0.3473 | 17448/9360 | 8080/0 |
| 25344t | 1056 | **0.3654** | 0.3262 | 9734/5728 | 4007/0 |
| 25344t | 1584 | **0.4267** | 0.3921 | 6454/3807 | 2645/0 |

**The scored policy loses every time, at 1.5-2.5x the copy-out traffic plus
thousands of fetches recency never makes.** This is the regime the mass work
predicted would favour scoring -- long context, genuine distant dependency,
budgets from 7.5% to 30% -- and it is a net negative there, not merely
neutral. Three budgets x three block sizes, no exceptions.

Taken with `proxy-is-broken` below, the demand-signal direction as built is
finished: the objective it optimises does not predict outcome, and pursuing it
costs transport and quality together.

**The mechanism is solid at this scale**: zero guard violations over ~230k
guarded steps, 163 turns, 84k-token contexts, on a hybrid with an EXL3
checkpoint at 262144 max-len. Far harder than any earlier run.

**Do not read the block-size axis**, which is confounded twice. `sink=1` is one
*block*, so the sink is 528/1056/1584 tokens as the block grows -- and the sink
dominates everything else measured here. And the baselines differ: `off`
generated 22129/22146/22410 tokens at the three block sizes, so block size
changes the output through kernel numerics and each column is scored against a
different reference. That also explains 1056 coming out worst, which has no
mechanical story.

**Sobering absolute:** at 30% residency on a real long session, agreement is
0.36-0.43. Most tokens diverge from the full-context baseline even at the
loosest budget tested. Whether that costs task success is a separate question
agreement cannot answer.

*Environment for max-context runs on the hybrid:* language model only, util
0.95, host tier cap raised to 50%.

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
