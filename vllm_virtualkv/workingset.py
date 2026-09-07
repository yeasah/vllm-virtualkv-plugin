"""How many blocks a step genuinely cannot do without.

The question this answers is prior to any policy, and decides whether the
design works at all. If a decode step's attention can be reproduced from a
small fraction of the blocks, residency is a cache and a mispredict is a
stall. If it needs most of them, no policy helps and no amount of prefetching
hides it.

**Exact and evicting are incompatible**, which is what makes a threshold
necessary rather than a compromise. Full attention at step t attends to all t
keys, so bit-exact output requires every block resident at every step -- that
is not paging, it is streaming the whole context per token. So the invariant
has to be *bounded* deviation: skip a block only when its contribution is
provably below epsilon.

**The bound is one-sided, which is the whole point.** `max(q.lo, q.hi)`
summed over channels is an upper bound on the best key score in a block, so
it can *prove* a block irrelevant; it can never wrongly claim one is. A block
is skippable when

    block_size * exp(scale * (upper_b - s_max)) < epsilon

since the block's numerator is at most `block_size * exp(scale * upper_b)`
and the partition function is at least `exp(scale * s_max)`. It has to hold
for every query head, every KV head and every layer at once, because
residency is per block and shared across all of them -- the union is what
makes this hard, and reporting a per-layer number would flatter it.

Two working sets are reported, and the gap between them is the finding:

    bound    blocks the bound cannot rule out -- what an implementation
             would have to fetch, and therefore the real stall rate
    oracle   blocks whose *true* mass is above epsilon -- what a perfect
             predictor would need

`oracle` is the floor imposed by the model and the data. `bound - oracle` is
slack in the bound, and is recoverable by tightening it; if that gap is most
of the number, sharpening the bound beats any policy work.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .audit import _keys_for
from .bounds import LayoutError, block_keys

#: Mass below which a block is treated as not worth fetching. Spread over
#: several orders of magnitude because the shape of the curve is the result:
#: a working set that barely moves across it means the blocks are sharply
#: separated into wanted and not, and a threshold is safe to pick.
EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4)


#: Bit widths for the quantized-key summary. The question is not whether a
#: quantized key is accurate -- it is whether it *ranks* blocks the way the
#: true keys do, which is a far weaker requirement and the only one a
#: residency decision needs.
BITS = (8, 4, 2)

#: Super-block factors over the native block size, so one run covers a
#: range of granularities including the 528 tokens a hybrid forces.
GROUPINGS = (1, 2, 4, 8, 16, 33)


def _quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-block, per-channel asymmetric quantization, dequantized in place.

    The same footprint story as the min/max bound -- two values per channel
    per block -- plus `bits` per key per channel for the codes. It buys an
    estimate of where each key actually sits instead of the corner of the box
    that contains them all.
    """
    lo = x.amin(dim=2, keepdim=True)
    hi = x.amax(dim=2, keepdim=True)
    step = (hi - lo).clamp(min=1e-9) / (2 ** bits - 1)
    return lo + torch.round((x - lo) / step) * step


def summary_step(caches: Sequence[torch.Tensor], tier, req_id: str,
                 row: Sequence[int], resident, n_full: int,
                 queries: Sequence[torch.Tensor], block_size: int,
                 head_size: int, scale: float,
                 head_size_v: int | None = None, tail: int = 0,
                 share: float = 0.25, shares: Sequence[float] = (),
                 stale: Sequence[torch.Tensor] | None = None) -> dict | None:
    """Which resident summary picks the blocks that actually carry the mass.

    Every selector gets the same budget and is scored on the same thing: the
    share of true attention mass its chosen blocks hold, summed over every
    layer and head. `oracle` is the ceiling -- it ranks on the true mass
    itself, so no summary can beat it and the gap to it is what a summary
    costs. `recency` is the floor, because it is what shipping today already
    does and a summary that cannot beat it is not worth its memory.
    """
    if not queries or n_full <= 0 or not scale:
        return None
    device = caches[0].device
    keys = _keys_for(caches, tier, req_id, row, n_full, head_size,
                     head_size_v, set(resident), device)
    if keys is None:
        return None

    n_keys = n_full * block_size
    true_sum = torch.zeros(n_full, device=device)
    bound_max = torch.full((n_full,), -float("inf"), device=device)
    est_sum = {b: torch.zeros(n_full, device=device) for b in BITS}
    stale_true = torch.zeros(n_full, device=device)
    stale_q2 = torch.zeros(n_full, device=device)
    #: Per-layer, kept rather than folded in. Aggregating over layers is what
    #: hides whether a flat union is being driven by a sensitive few.
    per_layer: list[torch.Tensor] = []

    for layer, (q, layer_keys) in enumerate(zip(queries, keys)):
        kv_heads = layer_keys.shape[0]
        if layer_keys.shape[1] < n_keys:
            return None
        layer_keys = layer_keys[:, :n_keys, :]
        q = q.float().reshape(kv_heads, -1, q.shape[-1])
        edge = None
        if tail > 0 and len(row) > n_full:
            edge = block_keys(caches[layer], row[n_full], head_size,
                              head_size_v)[:, :tail, :].float()

        def mass_of(k: torch.Tensor) -> torch.Tensor:
            full = k if edge is None else torch.cat([k, edge], dim=1)
            w = torch.softmax(torch.einsum("kgd,knd->kgn", q, full) * scale,
                              dim=-1)
            return w[..., :n_keys].reshape(
                *w.shape[:2], n_full, block_size).sum(-1)

        if stale:
            sq = stale[layer].float().reshape(kv_heads, -1, q.shape[-1])
            keep_q, q = q, sq
            stale_true += mass_of(layer_keys).sum(dim=(0, 1))
            stale_q2 += mass_of(
                _quantize(layer_keys.reshape(kv_heads, n_full, block_size, -1),
                          2).reshape(kv_heads, n_keys, -1)).sum(dim=(0, 1))
            q = keep_q
        layer_mass = mass_of(layer_keys).sum(dim=(0, 1))
        per_layer.append(layer_mass)
        true_sum += layer_mass
        blocked = layer_keys.reshape(kv_heads, n_full, block_size, -1)
        for bits in BITS:
            est_sum[bits] += mass_of(
                _quantize(blocked, bits).reshape(kv_heads, n_keys, -1)
            ).sum(dim=(0, 1))
        lo = blocked.min(dim=2).values.unsqueeze(1)
        hi = blocked.max(dim=2).values.unsqueeze(1)
        upper = torch.maximum(q.unsqueeze(2) * lo,
                              q.unsqueeze(2) * hi).sum(-1)
        bound_max = torch.maximum(bound_max, upper.amax(dim=(0, 1)))

    budget = max(1, int(round(share * n_full)))
    total = float(true_sum.sum()) or 1.0

    def captured(rank: torch.Tensor, b: int | None = None) -> float:
        pick = torch.topk(rank, min(b or budget, n_full)).indices
        return float(true_sum[pick].sum()) / total

    recency = torch.arange(n_full, device=device, dtype=torch.float32)
    #: Recency *with sinks*, which is what actually ships. The pure-recency
    #: floor above keeps none, and StreamingLLM's whole finding is that the
    #: first tokens absorb large attention mass while carrying no
    #: information -- so the gap between these two rows is how much of the
    #: "missed" mass is sink mass that no policy should want.
    sink_recency = recency.clone()
    sink_recency[:2] = float(n_full + 10)
    out = {"n_full": n_full, "budget": budget,
           "oracle": captured(true_sum),
           "bound": captured(bound_max),
           "recency": captured(recency),
           # Rank on layer 0 alone, score on every layer's mass: what a
           # signal computable before the forward would actually buy.
           "layer0": captured(per_layer[0]),
           "sink_recency": captured(sink_recency),
           # Mass sitting in the first two blocks alone. If this is most of
           # what recency misses, mass and importance come apart exactly
           # where the sink literature says they do.
           "sink_mass": float(true_sum[:2].sum()) / total}
    for bits in BITS:
        out[f"q{bits}"] = captured(est_sum[bits])
    if stale:
        out["oracle_stale"] = captured(stale_true)
        out["q2_stale"] = captured(stale_q2)

    # How the prize changes with the budget. A demand signal is worth
    # whatever separates it from what recency gets for free, and at a
    # generous budget a recency window already covers much of the context
    # -- so measuring only there understates scoring exactly where every
    # compression method actually operates.
    for sh in shares:
        b = max(1, int(round(sh * n_full)))
        sr = recency.clone()
        sr[:2] = float(n_full + 10)
        out[f"oracle@{sh:g}"] = captured(true_sum, b)
        out[f"q2@{sh:g}"] = captured(est_sum[2], b) if 2 in est_sum else 0.0
        out[f"sinkrec@{sh:g}"] = captured(sr, b)
        out[f"budget@{sh:g}"] = b

    # Granularity, at a *fixed token budget*. Block size is the unit of
    # every residency decision, and on a hybrid vLLM raises it to 528 tokens
    # to match the mamba page. A pager that can only cut in 528-token spans
    # makes enormous holes: the difference between dropping vowels and
    # tearing out a page. Grouping the native blocks into super-blocks
    # measures that directly, on one run, with the token budget held equal
    # so only the granularity changes.
    for g in GROUPINGS:
        wide = (n_full + g - 1) // g
        if wide < 4:
            continue
        pad = wide * g - n_full
        padded = torch.cat([true_sum, true_sum.new_zeros(pad)]) if pad \
            else true_sum
        coarse = padded.reshape(wide, g).sum(dim=1)
        keep = max(1, budget // g)          # same tokens, fewer, bigger units
        pick = torch.topk(coarse, min(keep, wide)).indices
        out[f"grain@{g * block_size}"] = float(coarse[pick].sum()) / total
        out[f"grainblocks@{g * block_size}"] = keep

    # The union question in mass terms: the globally-best pick still has to
    # serve every layer, and the worst-served layer is what a flat union is
    # really reporting.
    pick = torch.topk(true_sum, min(budget, n_full)).indices
    shares = torch.stack([m[pick].sum() / m.sum().clamp(min=1e-9)
                          for m in per_layer])
    out["worst_layer"] = float(shares.min())
    out["worst_layer_idx"] = int(shares.argmin())
    out["median_layer"] = float(shares.median())
    return out


def working_set_step(caches: Sequence[torch.Tensor], tier, req_id: str,
                     row: Sequence[int], resident, n_full: int,
                     queries: Sequence[torch.Tensor], block_size: int,
                     head_size: int, scale: float,
                     head_size_v: int | None = None,
                     epsilons: Sequence[float] = EPSILONS,
                     tail: int = 0) -> dict | None:
    """Blocks that cannot be ruled out for one decode step.

    `tail` is how many keys of the *partial* current block to include. They
    are not candidates for eviction and get no mass reported, but they have
    to be in the softmax: they are the newest tokens and take the largest
    share of attention, and leaving them out of the denominator hands their
    mass to the older blocks and makes every one of them look load-bearing.
    """
    if not queries or n_full <= 0 or not scale:
        return None
    device = caches[0].device
    keys = _keys_for(caches, tier, req_id, row, n_full, head_size,
                     head_size_v, set(resident), device)
    if keys is None:
        return None

    n_keys = n_full * block_size
    # A block is only skippable if it is skippable everywhere: start from
    # "yes" and let any layer or head veto it.
    skip_bound = {e: torch.ones(n_full, dtype=torch.bool, device=device)
                  for e in epsilons}
    skip_true = {e: torch.ones(n_full, dtype=torch.bool, device=device)
                 for e in epsilons}
    true_mass_max = torch.zeros(n_full, device=device)
    slack = []
    #: Blocks each layer *alone* cannot rule out, at the tightest epsilon.
    #: If the union is being set by a handful of layers this is where it
    #: shows, and a flat union is then the wrong criterion rather than a
    #: fatal result.
    alone: list[int] = []

    for layer, (q, layer_keys) in enumerate(zip(queries, keys)):
        kv_heads = layer_keys.shape[0]
        if layer_keys.shape[1] < n_keys:
            return None
        layer_keys = layer_keys[:, :n_keys, :]
        q = q.float().reshape(kv_heads, -1, q.shape[-1])       # [kv, grp, dim]

        full = layer_keys
        if tail > 0 and len(row) > n_full:
            edge = block_keys(caches[layer], row[n_full], head_size,
                              head_size_v)[:, :tail, :].float()
            full = torch.cat([layer_keys, edge], dim=1)
        scores = torch.einsum("kgd,knd->kgn", q, full) * scale
        weights = torch.softmax(scores, dim=-1)
        # [kv, grp, blocks] -- over the full blocks only; the tail is in the
        # denominator but is not a candidate.
        mass = weights[..., :n_keys].reshape(
            *weights.shape[:2], n_full, block_size).sum(-1)
        s_max = scores.max(dim=-1).values                       # [kv, grp]

        # Per-block channel ranges, then the upper bound on the best score in
        # each block for each query head.
        blocked = layer_keys.reshape(kv_heads, n_full, block_size, -1)
        lo = blocked.min(dim=2).values.unsqueeze(1)             # [kv,1,blk,dim]
        hi = blocked.max(dim=2).values.unsqueeze(1)
        qq = q.unsqueeze(2)                                     # [kv,grp,1,dim]
        upper = torch.maximum(qq * lo, qq * hi).sum(-1) * scale  # [kv,grp,blk]

        ceiling = block_size * torch.exp(upper - s_max.unsqueeze(-1))
        for e in epsilons:
            skip_bound[e] &= (ceiling < e).all(dim=1).all(dim=0)
            keep = (mass < e).all(dim=1).all(dim=0)
            skip_true[e] &= keep
            if e == min(epsilons):
                alone.append(n_full - int(keep.sum()))
        true_mass_max = torch.maximum(true_mass_max,
                                      mass.amax(dim=(0, 1)))
        # How many orders of magnitude the bound overstates the truth. A
        # linear mean is meaningless here: the bound can exceed the real mass
        # by 1e14, and one such block would swamp any average.
        slack.append(float((ceiling.amax(dim=(0, 1)).clamp(min=1e-30).log10()
                            - mass.amax(dim=(0, 1)).clamp(min=1e-30).log10()
                            ).mean()))

    out = {"n_full": n_full, "layers": len(keys),
           "slack": sum(slack) / len(slack),
           "layer_alone_mean": sum(alone) / max(len(alone), 1),
           "layer_alone_max": max(alone) if alone else 0,
           "layer_alone_min": min(alone) if alone else 0}
    for e in epsilons:
        out[f"bound@{e:g}"] = n_full - int(skip_bound[e].sum())
        out[f"oracle@{e:g}"] = n_full - int(skip_true[e].sum())
        # A bound that ever rules out a block the truth wanted is not an upper
        # bound, and the whole guarantee rests on that never happening.
        out[f"unsound@{e:g}"] = int((skip_bound[e] & ~skip_true[e]).sum())
    return out


def block_values(cache: torch.Tensor, block_id: int, head_size: int,
                 head_size_v: int | None = None) -> torch.Tensor:
    """The values of one block, the other half of what `block_keys` slices."""
    head_size_v = head_size if head_size_v is None else head_size_v
    block = cache[block_id]
    if block.ndim != 3 or block.shape[-1] != head_size + head_size_v:
        raise LayoutError(
            f"expected a block shaped [heads, block_size, {head_size} + "
            f"{head_size_v}] but got {tuple(block.shape)}")
    return block[..., head_size:]


def tier_step(caches: Sequence[torch.Tensor], req_id: str,
              row: Sequence[int], n_full: int,
              queries: Sequence[torch.Tensor], block_size: int,
              head_size: int, scale: float, head_size_v: int | None = None,
              tail: int = 0, share: float = 0.25, base_bits: int = 16,
              bits: Sequence[int] = (2, 4, 8)) -> dict | None:
    """Is a degraded block better than no block, at equal VRAM?

    Dropping a block is not "saving memory instead of being approximate" --
    it is the *worst* available quantization, zero bits, and it zeroes that
    block's entire contribution rather than perturbing it. Keeping it badly
    costs mass times a relative error; dropping it costs mass outright.

    So the arms are given the same bytes, not the same block count. A budget
    of `share` exact blocks buys either

        drop      `B` blocks exact, the rest absent
        degrade   `B'` exact plus every remaining block at `n` bits,
                  where B' = (B - n_full*r) / (1 - r) and r = n/base_bits

    At 4 bits against an fp16 cache the arithmetic is striking on its own:
    r = 0.25, so a 25% exact budget buys either a quarter of the blocks
    exactly or *all* of them at 4 bits, for identical memory.

    Scored as relative L2 error of the attention output against exact
    attention -- not mass captured. Mass is a proxy, and the whole point here
    is that two arms holding the same mass can be very differently wrong.
    Both arms select with the same oracle ranking, so this isolates the tier
    question from the selection question.
    """
    if not queries or n_full <= 0 or not scale:
        return None
    device = caches[0].device
    n_keys = n_full * block_size
    head_v = head_size if head_size_v is None else head_size_v

    def parts(layer: int):
        """Keys and values for every full block of this layer, plus the tail."""
        ks, vs = [], []
        for i in range(n_full):
            ks.append(block_keys(caches[layer], row[i], head_size,
                                 head_size_v).float())
            vs.append(block_values(caches[layer], row[i], head_size,
                                   head_size_v).float())
        k = torch.cat(ks, dim=1)
        v = torch.cat(vs, dim=1)
        ek = ev = None
        if tail > 0 and len(row) > n_full:
            ek = block_keys(caches[layer], row[n_full], head_size,
                            head_size_v)[:, :tail, :].float()
            ev = block_values(caches[layer], row[n_full], head_size,
                              head_size_v)[:, :tail, :].float()
        return k, v, ek, ev

    # Pass one: a single ranking, since residency is per block and shared
    # across every layer.
    rank = torch.zeros(n_full, device=device)
    for layer, q in enumerate(queries):
        k, _, ek, _ = parts(layer)
        kv_heads = k.shape[0]
        qq = q.float().reshape(kv_heads, -1, q.shape[-1])
        full = k if ek is None else torch.cat([k, ek], dim=1)
        w = torch.softmax(torch.einsum("kgd,knd->kgn", qq, full) * scale, -1)
        rank += w[..., :n_keys].reshape(*w.shape[:2], n_full,
                                        block_size).sum(-1).sum(dim=(0, 1))

    budget = max(1, int(round(share * n_full)))
    keep_drop = torch.topk(rank, min(budget, n_full)).indices
    out = {"n_full": n_full, "budget": budget}
    err = {("drop", 0): [], **{("degrade", b): [] for b in bits}}
    exact_for = {0: keep_drop}
    degraded_for = {}
    for b in bits:
        r = b / base_bits
        # Keeping every remaining block degraded is only affordable when
        # n_full*r fits the budget. At 8 bits against fp16 it does not, and
        # an unchecked formula silently hands that arm twice the memory --
        # which is exactly what it did. Spend on exact blocks first, then
        # degrade as many of the rest as the remainder buys, then drop.
        n_exact = int(round((budget - n_full * r) / max(1 - r, 1e-6)))
        n_exact = max(0, min(n_exact, n_full))
        n_deg = min(n_full - n_exact, int((budget - n_exact) / max(r, 1e-6)))
        n_deg = max(0, n_deg)
        order = torch.topk(rank, n_full).indices
        exact_for[b] = order[:n_exact]
        degraded_for[b] = order[n_exact:n_exact + n_deg]
        out[f"exact@{b}bit"] = n_exact
        out[f"degraded@{b}bit"] = n_deg
        out[f"cost@{b}bit"] = n_exact + n_deg * r

    for layer, q in enumerate(queries):
        k, v, ek, ev = parts(layer)
        kv_heads = k.shape[0]
        qq = q.float().reshape(kv_heads, -1, q.shape[-1])
        fk = k if ek is None else torch.cat([k, ek], dim=1)
        fv = v if ev is None else torch.cat([v, ev], dim=1)
        w = torch.softmax(torch.einsum("kgd,knd->kgn", qq, fk) * scale, -1)
        ref = torch.einsum("kgn,knd->kgd", w, fv)
        scale_ref = ref.norm(dim=-1).clamp(min=1e-9)

        def score(kk, vv, mask=None):
            s = torch.einsum("kgd,knd->kgn", qq, kk) * scale
            if mask is not None:
                s = s.masked_fill(mask, -float("inf"))
            o = torch.einsum("kgn,knd->kgd", torch.softmax(s, -1), vv)
            return float(((o - ref).norm(dim=-1) / scale_ref).mean())

        # drop: everything outside the exact set is invisible
        gone = torch.ones(n_full, dtype=torch.bool, device=device)
        gone[keep_drop] = False
        mask = torch.zeros(fk.shape[1], dtype=torch.bool, device=device)
        mask[:n_keys] = gone.repeat_interleave(block_size)
        err[("drop", 0)].append(score(fk, fv, mask.view(1, 1, -1)))

        # degrade: exact for the chosen few, quantized for everything else
        for b in bits:
            qk = _quantize(k.reshape(kv_heads, n_full, block_size, -1),
                           b).reshape(kv_heads, n_keys, -1)
            qv = _quantize(v.reshape(kv_heads, n_full, block_size, -1),
                           b).reshape(kv_heads, n_keys, -1)
            sel = torch.zeros(n_full, dtype=torch.bool, device=device)
            sel[exact_for[b]] = True
            held = sel.clone()
            held[degraded_for[b]] = True
            keep = sel.repeat_interleave(block_size).unsqueeze(-1)
            mk = torch.where(keep, k, qk)
            mv = torch.where(keep, v, qv)
            if ek is not None:
                mk = torch.cat([mk, ek], dim=1)
                mv = torch.cat([mv, ev], dim=1)
            # Anything the budget could not even hold degraded is absent.
            m = torch.zeros(mk.shape[1], dtype=torch.bool, device=device)
            m[:n_keys] = (~held).repeat_interleave(block_size)
            err[("degrade", b)].append(score(mk, mv, m.view(1, 1, -1)))

    out["drop"] = sum(err[("drop", 0)]) / len(err[("drop", 0)])
    for b in bits:
        out[f"degrade@{b}bit"] = sum(err[("degrade", b)]) / len(
            err[("degrade", b)])
    return out


def true_mass(caches: Sequence[torch.Tensor], tier, req_id: str,
              row: Sequence[int], resident, n_full: int,
              queries: Sequence[torch.Tensor], block_size: int,
              head_size: int, scale: float, head_size_v: int | None = None,
              tail: int = 0) -> list[float] | None:
    """Each block's true attention mass, summed over every layer and head.

    The ceiling a demand signal is estimating. Reconstructs evicted blocks
    from the host tier, which is the one thing an eviction method could never
    do and the reason this number is available here at all.
    """
    if not queries or n_full <= 0 or not scale:
        return None
    device = caches[0].device
    n_keys = n_full * block_size
    total = torch.zeros(n_full, device=device)
    #: One layer at a time. `_keys_for` materialises every layer's keys as
    #: float32 before any are scored, which on a long context is gigabytes
    #: in a single allocation -- 2.8 GiB at 40 blocks of 2112 tokens across
    #: 8 layers, which OOM'd the ceiling measurement twice. Only one layer
    #: is ever needed at once, and the peak drops by the layer count.
    for layer, q in enumerate(queries):
        one = _keys_for(caches[layer:layer + 1], tier, req_id, row, n_full,
                        head_size, head_size_v, set(resident), device)
        if one is None:
            return None
        layer_keys = one[0]
        kv_heads = layer_keys.shape[0]
        if layer_keys.shape[1] < n_keys:
            return None
        layer_keys = layer_keys[:, :n_keys, :]
        qq = q.float().reshape(kv_heads, -1, q.shape[-1])
        full = layer_keys
        if tail > 0 and len(row) > n_full:
            edge = block_keys(caches[layer], row[n_full], head_size,
                              head_size_v)[:, :tail, :].float()
            full = torch.cat([layer_keys, edge], dim=1)
        w = torch.softmax(torch.einsum("kgd,knd->kgn", qq, full) * scale, -1)
        total += w[..., :n_keys].reshape(*w.shape[:2], n_full,
                                         block_size).sum(-1).sum(dim=(0, 1))
        del one, layer_keys, full, w
    return [float(x) for x in total]


def _kv_for_layer(caches, tier, req_id: str, row, n_full: int, layer: int,
                  head_size: int, head_size_v, resident, device):
    """Keys *and* values for one layer, resident from GPU, evicted from host.

    One layer at a time by construction: the whole-model version costs
    gigabytes on a long context and OOM'd the ceiling run twice.
    """
    head_v = head_size if head_size_v is None else head_size_v
    ks, vs = [], []
    cache = caches[layer]
    for i in range(n_full):
        if i in resident and i < len(row):
            block = cache[row[i]]
            if block.ndim != 3 or block.shape[-1] != head_size + head_v:
                return None
            ks.append(block[..., :head_size].float())
            vs.append(block[..., head_size:].float())
        else:
            slot = tier._slot.get((req_id, i)) if tier is not None else None
            if slot is None:
                return None
            held = tier.buffers[layer][slot].to(device, non_blocking=False)
            if held.shape[-1] != head_size + head_v:
                return None
            ks.append(held[..., :head_size].float())
            vs.append(held[..., head_size:].float())
    return torch.cat(ks, dim=1), torch.cat(vs, dim=1)


def true_impact(caches: Sequence[torch.Tensor], tier, req_id: str,
                row: Sequence[int], resident, n_full: int,
                queries: Sequence[torch.Tensor], block_size: int,
                head_size: int, scale: float, head_size_v: int | None = None,
                tail: int = 0) -> list[float] | None:
    """How much dropping each block would move the attention output.

    Not attention mass. Mass is what every signal in this repo has estimated
    -- bounds, the 2-bit summary, quest -- and a mass oracle with perfect
    knowledge scored 0.2372 against recency's 0.2374, so there is nothing in
    that quantity to win. Mass says how much weight a block carries; it does
    not say whether carrying it changes the answer.

    What does is available in closed form. For an output `o = sum_k w_k v_k`,
    removing block `b` of mass `m_b` and mass-weighted mean value `v_b`
    renormalises the rest, giving

        delta_o = m_b * (o - v_b) / (1 - m_b)

    so importance is mass *weighted by how far the block's values sit from
    what attention was already producing*. A heavy block whose values match
    the centroid is nearly free to drop; a light block pulling the output
    somewhere else is not. No extra forward passes -- the terms are already
    in the weights and values.

    This is the oracle that settles the premise rather than the signal. If
    ranking by it cannot beat recency either, no demand signal can, because
    this *is* the quantity a demand signal would be trying to predict.
    """
    if not queries or n_full <= 0 or not scale:
        return None
    device = caches[0].device
    n_keys = n_full * block_size
    resident = set(resident)
    total = torch.zeros(n_full, device=device)

    for layer, q in enumerate(queries):
        got = _kv_for_layer(caches, tier, req_id, row, n_full, layer,
                            head_size, head_size_v, resident, device)
        if got is None:
            return None
        k, v = got
        kv_heads = k.shape[0]
        if k.shape[1] < n_keys:
            return None
        k, v = k[:, :n_keys], v[:, :n_keys]
        qq = q.float().reshape(kv_heads, -1, q.shape[-1])
        fk, fv = k, v
        if tail > 0 and len(row) > n_full:
            edge = caches[layer][row[n_full]]
            hv = head_size if head_size_v is None else head_size_v
            fk = torch.cat([k, edge[..., :head_size][:, :tail, :].float()], 1)
            fv = torch.cat([v, edge[..., head_size:][:, :tail, :].float()], 1)
        w = torch.softmax(torch.einsum("kgd,knd->kgn", qq, fk) * scale, -1)
        o = torch.einsum("kgn,knd->kgd", w, fv)

        wb = w[..., :n_keys].reshape(*w.shape[:2], n_full, block_size)
        vb = v.reshape(kv_heads, n_full, block_size, -1)
        m = wb.sum(-1)                                     # [kv, grp, blocks]
        num = torch.einsum("kgnb,knbd->kgnd", wb, vb)
        vbar = num / m.clamp(min=1e-12).unsqueeze(-1)
        delta = (o.unsqueeze(2) - vbar).norm(dim=-1)
        total += (m * delta / (1 - m).clamp(min=1e-6)).sum(dim=(0, 1))
        del got, k, v, fk, fv, w, o, wb, vb, num, vbar, delta
    return [float(x) for x in total]
