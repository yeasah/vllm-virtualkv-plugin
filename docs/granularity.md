# Block size, and what it costs

Residency is per block, so the block size is the unit of every decision this
plugin makes. It is not a free parameter on every model, and where it is
forced it is forced *large*.

## Where the block size comes from

On a dense full-attention model it is vLLM's `block_size`, default 16, and
settable. On a **hybrid** — linear/GDN attention in most layers, full
attention in a few — it is not:

    Setting attention block size to 528 tokens to ensure that
    attention page size is >= mamba page size.

The attention page must be at least the mamba page, and vLLM grows the
attention block until it is. Worse, the constraint runs the wrong way when
you try to escape it:

| KV dtype | block size on Qwen3.5-9B |
|---|---|
| bf16 | 528 |
| fp8 | 1056 |
| tq4 | 2048 |

**Quantising the KV cache to buy headroom coarsens residency**, because a
smaller attention page needs more tokens to match the mamba page. There is
no configuration that escapes downward. The only routes to finer granularity
on a hybrid are decoupling the two page sizes upstream, or sub-block
residency underneath vLLM's block.

## What coarse blocks cost — in mass

At a fixed *token* budget with oracle selection, grouping native 16-token
blocks into super-blocks:

| block size | blocks kept | mass captured |
|---|---|---|
| 16 | 86 | 0.8892 |
| 32 | 43 | 0.8711 |
| 64 | 21 | 0.8413 |
| 128 | 10 | 0.8103 |
| 256 | 5 | 0.7466 |

A monotonic 16% loss from 16 to 256 tokens per block, extrapolating to ~0.70
at 528. This is the *best* case: an oracle ranks super-blocks by their true
summed mass, where a real policy estimates from a summary.

## What coarse blocks cost — in output

Nothing measurable. On a uniform-RoPE full-attention model where block size
is a free parameter, at a fixed 12672t budget with the sink pinned at 4224
tokens:

| block | recency | quest |
|---|---|---|
| 64 | 0.2147 | 0.2029 |
| 1056 | 0.2102 | 0.2087 |

A 16.5× change moves nothing outside the ~0.03 noise floor. **The mass curve
predicted a fifth of capturable mass lost by 528 tokens and none of it
reached the outcome** — the third time in this project that mass failed to
predict output damage.

That matters beyond granularity: it means the earlier selection results,
which all sat at ~40 units of freedom, were not being masked by their
coarseness. "Selection does not help" and "selection cannot be expressed"
are separable now, and the answer is the same.

## Measuring it correctly

Two confounds ruined earlier attempts and will ruin the next one:

**Hold every block-denominated knob in tokens.** `sink` is a *block* count,
so `sink=2` is 32 tokens at block 16 and 1056 at block 528. A sweep that
leaves it fixed compares budget *allocations*, not granularities — and gives
the opposite answer, because a coarse sink accidentally retains far more of
the prefix, which is where the mass is.

**Each block size needs its own baseline.** `off` generated 22129 / 22146 /
22410 tokens at three block sizes on the same input. Block size should not
change output at all, so that is kernel-level numerical variation — and it
means cross-block-size comparisons carry baseline drift that small
differences cannot be resolved against.
