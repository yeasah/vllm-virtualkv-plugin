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
from .bounds import block_keys

#: Mass below which a block is treated as not worth fetching. Spread over
#: several orders of magnitude because the shape of the curve is the result:
#: a working set that barely moves across it means the blocks are sharply
#: separated into wanted and not, and a threshold is safe to pick.
EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4)


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
            skip_true[e] &= (mass < e).all(dim=1).all(dim=0)
        true_mass_max = torch.maximum(true_mass_max,
                                      mass.amax(dim=(0, 1)))
        # How many orders of magnitude the bound overstates the truth. A
        # linear mean is meaningless here: the bound can exceed the real mass
        # by 1e14, and one such block would swamp any average.
        slack.append(float((ceiling.amax(dim=(0, 1)).clamp(min=1e-30).log10()
                            - mass.amax(dim=(0, 1)).clamp(min=1e-30).log10()
                            ).mean()))

    out = {"n_full": n_full, "layers": len(keys),
           "slack": sum(slack) / len(slack)}
    for e in epsilons:
        out[f"bound@{e:g}"] = n_full - int(skip_bound[e].sum())
        out[f"oracle@{e:g}"] = n_full - int(skip_true[e].sum())
        # A bound that ever rules out a block the truth wanted is not an upper
        # bound, and the whole guarantee rests on that never happening.
        out[f"unsound@{e:g}"] = int((skip_bound[e] & ~skip_true[e]).sum())
    return out
