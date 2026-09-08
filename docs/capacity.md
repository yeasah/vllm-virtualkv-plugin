# Capacity, and the arithmetic that closes it

The direction that outlived the demand-signal negative: *let a request declare
a context larger than VRAM and pay for it in stalls rather than quality.* It
does not survive its own transport arithmetic. Nothing here is a policy
question — the numbers below are PCIe, and PCIe is already at the consumer
ceiling.

## At a hybrid's block size the transport is bandwidth-bound

`cost = copies x 1.30 us + bytes / 54 GB/s`, measured in `vllm-exl3-plugin`
`docs/kv-pager.md`. At block 528 the second term dominates and every
plugin-specific parameter — scatter, coalescing, block size itself —
disappears into a 3% rounding error. Geometry is Qwen3.8-27B: 16 full-attention
layers of 64, 4 KV heads x 256.

| | bf16 KV, block 528 | fp8 KV, block 1056 |
|---|---|---|
| bytes/token | 64 KiB | 32 KiB |
| bytes/block | **33.0 MiB** | **33.0 MiB** |
| copies/block | 16 | 16 |
| submission tax | 21 us = **3.1%** | 3.1% |
| cost/block | 662 us | 662 us |
| effective rate | 52.3 GB/s | 52.3 GB/s |
| **in 20 ms** | **30 blocks / 16.0k tokens** | **30 blocks / 31.9k tokens** |
| in 1 ms (5% of a 20.7 ms step) | 830 tokens | 1,660 tokens |

The two columns agree because the page-alignment rule (`granularity.md`) doubles
the block as the dtype halves, so **a block is 33 MiB whatever the KV dtype**.
Cross-check: this reproduces the contiguous 5% figure of 1,718 tokens from the
original transport sweep at 1,660, the difference being the submission tax.

**So 20 ms buys 1.05 GB, and no engineering changes that.** 54 GB/s is 86% of
PCIe 5.0 x16's theoretical 63 GB/s. There is no headroom in the mechanism and
none in the hardware.

## Exact capacity is bounded by that figure, not by host RAM

Decode attention is all-to-all, so "stalls rather than quality" means every
non-resident byte crosses the bus every step:

    offloaded set <= 1.05 GB ~= 16k tokens (bf16) at 20 ms/step

Against a 24 GiB card holding a 27B at 4 bpw with ~9.5 GiB free for KV — 156k
tokens — that is **+10% context for 2x step latency**, and this is the
*favorable* geometry. Qwen3-8B at bf16 (144 KiB/token) gets 7.1k tokens per
20 ms.

Host RAM is a red herring throughout. 128 GB holds ~2M tokens and takes 2.4 s
per step to sweep. **The tier's useful size is set by its bandwidth, never by
its capacity.** Past the ceiling you must choose which 1 GB to fetch, which is
the selection problem `demand-signal.md` closed — so capacity above ~1 GB
silently re-inherits the demand signal rather than being independent of it.

## Overlap is worth 2x and no more

A decode step is an HBM-bound read of the weights; the copy engines are idle.
For *exact* capacity the whole offloaded set is wanted, so it is known before
the step begins and prefetches perfectly: the step is `max(compute, transfer)`,
not the sum. That is the one real lever, and it is a factor of two.

Note this only works for the exact case. A Quest-style selector cannot overlap
at all, because it needs the query first — which inverts the usual intuition
about which is cheaper.

## On a small card it is a bpw trade, not a context trade

The deal improves as VRAM shrinks, because the offloaded-to-resident ratio
improves. It does not improve enough. 27B on 16 GiB, 4-bit KV, with the
embed+head tax present:

| | KV space | context |
|---|---|---|
| 3.0 bpw | 3.57 GiB | 234k |
| 3.5 bpw | 2.00 GiB | 131k |
| 4.0 bpw | 0.43 GiB | 28k |

Paging never wins on context: at 3 bpw the KV budget already is not the binding
constraint. The only trade it can offer is **spend the VRAM on weight quality
and buy the context back with bandwidth**:

| | context | step | t/s |
|---|---|---|---|
| 3.0 bpw, no paging | 234k | 10.5 ms | 95 |
| 4.0 bpw + 1 GB paged, 5.0 x16 | 89k | 18.5 ms | **54** |
| 4.0 bpw + 1 GB paged, 4.0 x16 | 89k | 38.5 ms | 26 |
| 4.0 bpw + 1 GB paged, 4.0 x8 | 89k | 76.9 ms | 13 |

So: +1.0 bpw for 43% of throughput and *less* context than you started with.
Arguable on one card, dead on the others.

**And the premise does not hold on the cards that create the scenario.** Among
16 GiB consumer cards only the RTX 5080 is PCIe 5.0 x16. The 5060 Ti and
4060 Ti are x8, and the 4060 Ti is 4.0 x8 — 13 GB/s, where 1 GB costs 77 ms and
there is no per-step story at all. The modal 16 GiB card is a quarter of the
ceiling this arithmetic assumes.

## Per-step is the wrong denominator, and it does not save it

The workload paging helps most — long prompt, short generation — levies the tax
fewest times, against a prefill that dominates the request:

| workload | baseline | paged | overhead |
|---|---|---|---|
| needle: 120k prompt, 200 out | 32.8 s | 33.7 s | **2.7%** |
| agent turn: 84k prompt, 617 out | 29.7 s | 32.4 s | 9.2% |
| reasoning: 8k prompt, 8k out | 114.8 s | 150.1 s | 30.8% |

(5080 class, overlapped, ~4k tok/s prefill assumed; the ratio is the point, not
the rate.) This is the strongest form of the case and it is still bounded by the
+10% context above — a small overhead on a capacity gain that was never large.

## Prefill paging is mechanically required, and has two variants

Not a design choice: if declared context exceeds the KV cache, an uncached
prompt larger than VRAM arrives eventually and OOMs. Every benchmark in this
project avoided it only by never running the configuration the plugin exists
for.

- **Windowed prefill** (evict as you go, never fetch back) is the free floor,
  and it is *not* truncation. Each prompt token attends to its own preceding
  window, and on a hybrid the 48 GDN layers are passed through untouched, so
  their recurrent state sees the entire prompt with unbounded receptive field.
  A hybrid with windowed full-attention layers is a known-good architecture.
- **Streaming prefill** (each chunk fetches back what it needs) is exact and
  costs one N^2/2 sweep: ~39 GB at 100k, ~159 GB at 200k with 4-bit KV. Against
  a compute-bound prefill that is **3-6% on 5.0 x16**, 12-24% on 4.0 x8. Both
  the traffic and the attention compute scale as N^2/2, so unlike the per-step
  decode tax that ratio does not degrade with context.

Prefill reach is therefore cheap. Decode reach is the expensive half, and by
`demand-signal.md` it is also the half nothing wants.

**One risk that does not exist at decode.** Under prefix caching, tokens
prefilled while part of the context is evicted are encoded against a windowed
view and *that encoding is cached and reused by every later turn*. A prefill
residency error is durable where a decode one is a stall — it inverts the
property this plugin was built on, and it passes every guard, because nothing
about it is illegal. On the 84k/163-turn session with a 16k budget, roughly 80%
of the final context would have been written under a windowed view against the
~0% the tests actually ran.

## Under recency the host tier is write-only

The corollary that decides what would ship. With no policy that fetches, the
window only slides forward, so no evicted block is ever wanted again:
`copied_in` goes to zero and the host copy is never read. That configuration is
not paging — it is a sliding window with a backup nobody reads, and the
copy-out is pure overhead that could be deleted by freeing the blocks instead.

So if declared-context-at-N ever beats truncation-to-W, the artifact is **a
serving-time sliding window imposed on the full-attention layers of a hybrid** —
a spec-and-mask change, adjacent to the in-tree R-SWA machinery — needing none
of the host tier, the guard, the relocation or the restore path. The parts of
this plugin that were hard are the parts that only matter if something gets
fetched back.

## Verdict

Not viable at any operating point examined. On a 24 GiB card it is +10% context
for 2x latency; on 16 GiB it becomes a bpw-for-bandwidth trade that is arguable
on exactly one card and dead on the rest; the modal card is at a quarter of the
assumed bandwidth; and the configuration that survives all of that is one the
mechanism is not needed to deliver.
