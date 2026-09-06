"""What the policy should have kept, measured against what it did keep.

Transfer counters say how much moved. They cannot say whether it was the right
thing: a policy that fetches constantly and a policy that fetches well look
identical in `copied_in`. The question a policy is refined against is how much
of the attention the model actually wanted was resident when it wanted it, and
that needs ground truth rather than the policy's own opinion.

It is computable here, and only here. A block that has been evicted still
exists -- its keys are in the host tier -- so the true scores for *every*
block, resident or not, can be reconstructed and softmaxed exactly as attention
would. That is a luxury an eviction method does not have: having thrown the
keys away, it can never find out what they were worth.

Reported per decode step, averaged over layers and query heads:

    missed          attention mass on blocks that were not resident. The
                    number that predicts quality: zero means the restricted
                    set was as good as the whole context for that step.
    worst_layer     the same for whichever layer fared worst, because
                    residency is shared across layers and an average hides a
                    layer that lost everything.
    fetched_mass    mass carried by blocks restored this step -- whether the
                    fetch paid for itself.
    evicted_mass    mass that leaving those blocks would have cost had the
                    step needed them, which is the eviction's price.

This is instrumentation, not a code path: it recomputes full attention in
Python and is far slower than the thing it measures. It exists to be switched
on for a measurement run and off otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .bounds import block_keys


def _keys_for(caches: Sequence[torch.Tensor], tier, req_id: str,
              row: Sequence[int], n_full: int, head_size: int,
              head_size_v: int | None, resident: set[int], device):
    """Every full block's keys, taken from the GPU or the host as required."""
    per_layer = []
    for layer, cache in enumerate(caches):
        blocks = []
        for i in range(n_full):
            if i in resident and i < len(row):
                keys = block_keys(cache, row[i], head_size, head_size_v)
            else:
                slot = tier._slot.get((req_id, i)) if tier is not None else None
                if slot is None:
                    return None            # cannot audit what nothing holds
                held = tier.buffers[layer][slot].to(device, non_blocking=False)
                head_v = head_size if head_size_v is None else head_size_v
                if held.shape[-1] != head_size + head_v:
                    return None
                keys = held[..., :head_size]
            blocks.append(keys.float())
        per_layer.append(torch.cat(blocks, dim=1))    # [kv_heads, keys, dim]
    return per_layer


def audit_step(caches, tier, req_id, row, resident, n_full, queries,
               block_size, head_size, head_size_v=None,
               restored=(), evicted=(), scale=None):
    """Attention mass the policy left behind, for one decode step.

    `queries` must be the queries of the step whose residency is being judged,
    not the next one's -- the point is to score the decision against the
    attention it was actually serving.

    `scale` is the kernel's softmax scale and is not optional in any
    meaningful sense. The captured query is the one entering
    `Attention.forward`, before the backend applies `1/sqrt(head_size)`, so
    softmaxing it raw exponentiates logits an order of magnitude too large.
    That does not merely rescale the answer: it reports mass off a
    distribution far sharper than the model's, which understates how much
    attention sits outside the resident set -- the exact quantity this
    module exists to report.
    """
    if not queries or n_full <= 0:
        return None
    device = caches[0].device
    resident = set(resident)
    keys = _keys_for(caches, tier, req_id, row, n_full, head_size,
                     head_size_v, resident, device)
    if keys is None:
        return None

    n_keys = n_full * block_size
    block_of = torch.arange(n_keys, device=device) // block_size
    out_mask = torch.tensor(
        [i not in resident for i in range(n_full)], device=device)[block_of]
    restored_mask = torch.tensor(
        [i in set(restored) for i in range(n_full)], device=device)[block_of]
    evicted_mask = torch.tensor(
        [i in set(evicted) for i in range(n_full)], device=device)[block_of]

    missed, fetched, evicted_cost, per_layer_missed = [], [], [], []
    for q, layer_keys in zip(queries, keys):
        kv_heads = layer_keys.shape[0]
        q = q.float().reshape(kv_heads, -1, q.shape[-1])      # [kv, group, dim]
        scores = torch.einsum("kgd,knd->kgn", q, layer_keys)
        if scale is not None:
            scores = scores * scale
        weights = torch.softmax(scores, dim=-1)
        missed.append(weights[..., out_mask].sum(-1).mean().item())
        fetched.append(weights[..., restored_mask].sum(-1).mean().item())
        evicted_cost.append(weights[..., evicted_mask].sum(-1).mean().item())
        per_layer_missed.append(missed[-1])

    n = len(missed)
    return {
        "missed": sum(missed) / n,
        "worst_layer": max(per_layer_missed),
        "fetched_mass": sum(fetched) / n,
        "evicted_mass": sum(evicted_cost) / n,
        "resident_blocks": len(resident),
        "total_blocks": n_full,
    }
