"""The bound has to bound. Everything else about it is tuning.

These check the property directly against scores computed the long way.

The layout is pinned by a *different* pair of tests, and the distinction is
worth stating because it was found by mutating the slice rather than by
reasoning. Comparing the bound against scores computed the long way cannot
catch a wrong slice at all: both sides extract keys through the same function,
so slicing V instead of K is self-consistent and the bound still bounds. What
catches it is the two tests that reach into the cache independently -- one
writes a uniform K half by raw indexing and requires the bound to become tight,
the other plants a key aligned with the query and requires that block to rank
first. A test whose reference comes from the code under test proves less than
it appears to.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from vllm_virtualkv.bounds import (  # noqa: E402
    LayoutError,
    block_keys,
    bound_scores,
    compute_bounds,
)

LAYERS, BLOCKS, KV_HEADS, BLOCK_SIZE, DIM = 3, 8, 2, 16, 32


def make_caches(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    # [blocks, kv_heads, block_size, K dim + V dim], K first.
    return [torch.randn((BLOCKS, KV_HEADS, BLOCK_SIZE, 2 * DIM),
                        generator=g, dtype=dtype) for _ in range(LAYERS)]


def test_the_bound_is_never_below_the_best_real_score():
    caches = make_caches()
    q_heads = KV_HEADS * 2
    g = torch.Generator().manual_seed(7)
    queries = [torch.randn((q_heads, DIM), generator=g) for _ in range(LAYERS)]

    bounds = torch.stack([compute_bounds(caches, b, DIM) for b in range(BLOCKS)])
    scores = bound_scores(queries, bounds, head_agg="max", layer_agg="max")

    # The long way: every query head against every key in the block.
    for b in range(BLOCKS):
        best = -float("inf")
        for layer, cache in enumerate(caches):
            keys = block_keys(cache, b, DIM).float()          # [kv, n, dim]
            q = queries[layer].reshape(KV_HEADS, -1, DIM)      # [kv, group, dim]
            got = torch.einsum("kgd,knd->kgn", q, keys).max().item()
            best = max(best, got)
        assert scores[b].item() >= best - 1e-4, (
            f"block {b}: bound {scores[b].item():.4f} is below a real score "
            f"of {best:.4f}, so a policy could leave away a block it needs"
        )


def test_a_block_of_one_repeated_key_is_bounded_tightly():
    """With no spread, the bound is the score -- it cannot be vacuous."""
    caches = make_caches()
    for cache in caches:
        cache[3, :, :, :DIM] = cache[3, :, :1, :DIM]     # every key identical
    q = [torch.ones((KV_HEADS, DIM)) for _ in range(LAYERS)]
    bounds = torch.stack([compute_bounds(caches, b, DIM) for b in range(BLOCKS)])
    score = bound_scores(q, bounds, head_agg="max", layer_agg="max")[3].item()

    direct = max(
        (block_keys(cache, 3, DIM).float() @ torch.ones(DIM)).max().item()
        for cache in caches
    )
    assert abs(score - direct) < 1e-3, (
        f"bound {score:.4f} against an attainable {direct:.4f}")


def test_a_block_holding_the_query_outranks_noise():
    """The signal has to be usable, not merely valid: a bound that is correct
    and uninformative would rank blocks arbitrarily."""
    caches = make_caches(seed=3)
    g = torch.Generator().manual_seed(11)
    queries = [torch.randn((KV_HEADS, DIM), generator=g) for _ in range(LAYERS)]
    planted = 5
    for cache, q in zip(caches, queries):
        cache[planted, :, 0, :DIM] = q * 4                # a key aimed at it

    bounds = torch.stack([compute_bounds(caches, b, DIM) for b in range(BLOCKS)])
    scores = bound_scores(queries, bounds)
    assert int(scores.argmax()) == planted, (
        f"the block containing a key aligned with the query ranked "
        f"{int(scores.argmax())}, not {planted}: {scores.tolist()}")


def test_mean_and_max_aggregation_both_run_and_differ():
    caches = make_caches(seed=5)
    g = torch.Generator().manual_seed(13)
    queries = [torch.randn((KV_HEADS * 2, DIM), generator=g) for _ in range(LAYERS)]
    bounds = torch.stack([compute_bounds(caches, b, DIM) for b in range(BLOCKS)])
    hard = bound_scores(queries, bounds, head_agg="max", layer_agg="max")
    soft = bound_scores(queries, bounds, head_agg="mean", layer_agg="mean")
    assert hard.shape == soft.shape == (BLOCKS,)
    assert not torch.allclose(hard, soft), (
        "the two aggregations agree exactly, so one of them is not wired in")
    assert (hard >= soft - 1e-4).all(), "max should not fall below mean"


def test_an_unexpected_layout_is_refused_not_sliced():
    bad = torch.randn((BLOCKS, KV_HEADS, BLOCK_SIZE, DIM + 1))
    with pytest.raises(LayoutError):
        block_keys(bad, 0, DIM)


def test_bounds_are_taken_in_fp32_even_from_a_low_precision_cache():
    caches = make_caches(dtype=torch.bfloat16)
    bounds = compute_bounds(caches, 0, DIM)
    assert bounds.dtype == torch.float32
    keys = block_keys(caches[0], 0, DIM).float()
    assert torch.equal(bounds[0, :, 0, :], keys.amin(dim=1))
    assert torch.equal(bounds[0, :, 1, :], keys.amax(dim=1))
