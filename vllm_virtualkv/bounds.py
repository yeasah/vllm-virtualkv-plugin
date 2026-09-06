"""A resident summary of a block that is not, so a policy can want it back.

Recency cannot fetch. Its window only slides forward, so it never asks for a
block again, and no policy can ask for one without something resident telling
it that a *non*-resident block matters. That something is this: per-channel
minimum and maximum of the keys in a block, kept when the block leaves.

Given the current query `q`, the largest score any key in a block could
achieve is bounded channel by channel --

    upper(block) = sum_d max(q_d * min_d, q_d * max_d)

-- because each channel independently contributes at most whichever end of its
range pairs better with `q_d`. It is loose (no single key need attain it) and
it is an upper bound, which is the direction that matters: a block whose bound
is low cannot contain a key that scores high, so it is safe to leave away.
This is Quest's mechanism (arXiv 2406.10774), used here to decide residency
rather than sparse attention.

**What it costs.** `num_kv_heads * head_size * 2` values per block per layer:
4 KiB in fp16 against a 32 KiB block on a 27B at fp8, so about 12.5%, and it
comes out of the same budget the blocks do. Worth measuring at the intended
geometry rather than assuming, since it scales with layers and heads while the
saving scales with context.

**Two aggregations this file deliberately does not decide.** A bound is per
layer and per KV head; residency is per *block*, shared by every layer, because
one block table serves them all. So the per-head and per-layer bounds have to
be collapsed into one number per block, and the two choices pull apart: `max`
is the sound upper bound, while an earlier measurement on this stack found mean
aggregation captured noticeably more attention mass than max when selecting a
GQA group's shared set (76.7% vs 69.6% at a 5% budget). Both are exposed;
neither is measured *here*. Measured elsewhere, since: with `head_agg="mean"`
a scored policy failed to select the block holding a planted answer, and with
`"max"` it selected it, at the same budget -- so the earlier mass-capture
finding does not transfer to retrieval, where the signal sits in a few query
heads and averaging over the group dilutes it. The layer-wise choice made no
difference to that outcome.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class LayoutError(RuntimeError):
    """The KV cache is not shaped the way key extraction assumes.

    Raised rather than guessed at. A wrong slice produces bounds that are
    merely wrong, which shows up as a policy that chooses badly -- a quality
    regression with no error attached, and the hardest kind to attribute.
    """


def block_keys(cache: torch.Tensor, block_id: int, head_size: int,
               head_size_v: int | None = None) -> torch.Tensor:
    """The keys held in one block of one layer, as [heads, block_size, dim].

    The single place a KV cache layout is assumed. FlashAttention packs K and V
    into the content dimension as `(num_blocks, num_kv_heads, block_size,
    head_size + head_size_v)` with K first; anything else has to be taught
    here, and is refused rather than sliced hopefully.
    """
    head_size_v = head_size if head_size_v is None else head_size_v
    block = cache[block_id]
    if block.ndim != 3 or block.shape[-1] != head_size + head_size_v:
        raise LayoutError(
            f"expected a block shaped [heads, block_size, "
            f"{head_size} + {head_size_v}] but got {tuple(block.shape)}; "
            f"key extraction has to be taught this layout before bounds mean "
            f"anything"
        )
    return block[..., :head_size]


def compute_bounds(caches: Sequence[torch.Tensor], block_id: int,
                   head_size: int, head_size_v: int | None = None
                   ) -> torch.Tensor:
    """Per-channel key range for one block across every layer.

    Returns [layers, heads, 2, head_size], the 2 being (min, max). Taken in
    fp32 and returned in fp32: these are compared against a query, and a bound
    computed in the cache's own dtype would round the wrong way for the low
    end. Storage precision is a separate decision from arithmetic precision.
    """
    per_layer = []
    for cache in caches:
        keys = block_keys(cache, block_id, head_size, head_size_v).float()
        per_layer.append(torch.stack([keys.amin(dim=1), keys.amax(dim=1)], dim=1))
    return torch.stack(per_layer)


def bound_scores(queries: Sequence[torch.Tensor], bounds: torch.Tensor,
                 head_agg: str = "max", layer_agg: str = "max") -> torch.Tensor:
    """Upper bound on each block's best key score, for this step's queries.

    `queries` is one tensor per layer, [num_query_heads, head_size]; `bounds`
    is [blocks, layers, kv_heads, 2, head_size]. Query heads are folded onto
    their KV head by GQA grouping, since residency is decided per KV head at
    best and per block in fact.
    """
    if bounds.ndim != 5:
        raise ValueError(f"bounds should be 5-d, got {tuple(bounds.shape)}")
    n_blocks, n_layers, n_kv, _, dim = bounds.shape
    if len(queries) != n_layers:
        raise ValueError(
            f"{len(queries)} layers of queries against {n_layers} of bounds")

    per_layer = []
    for layer, q in enumerate(queries):
        q = q.float()
        group = q.shape[0] // n_kv
        # [kv_heads, group, dim] -> the queries sharing each KV head
        q = q.reshape(n_kv, group, dim)
        lo = bounds[:, layer, :, 0, :].unsqueeze(1)      # [blocks, 1, kv, dim]
        hi = bounds[:, layer, :, 1, :].unsqueeze(1)
        qq = q.permute(1, 0, 2).unsqueeze(0)             # [1, group, kv, dim]
        best = torch.maximum(qq * lo, qq * hi).sum(dim=-1)   # [blocks, group, kv]
        per_layer.append(_reduce(best, dim=1, how=head_agg))  # [blocks, kv]
    stacked = torch.stack([_reduce(x, dim=1, how=head_agg) for x in per_layer])
    return _reduce(stacked.permute(1, 0), dim=1, how=layer_agg)


def _reduce(x: torch.Tensor, dim: int, how: str) -> torch.Tensor:
    if how == "max":
        return x.amax(dim=dim)
    if how == "mean":
        return x.mean(dim=dim)
    raise ValueError(f"unknown aggregation {how!r}; use 'max' or 'mean'")
