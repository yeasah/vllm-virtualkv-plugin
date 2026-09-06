"""Scoring residency against the query, which only the worker can do.

Every policy so far is a pure function of how many blocks exist and how far the
request has got, which is what lets the scheduler and the worker each call it
and agree. A query-aware policy cannot be: the query exists only inside the
forward, on the worker's side, and the scheduler that owns freeing has no way
to compute it. This is the point the design has been predicting since block
order was first measured -- the decision has to travel worker to scheduler.

It travels one step stale, and that is not a compromise but the only thing
available. Residency has to be settled *before* the forward, because the blocks
must be in the block table when attention runs; the query for that forward does
not exist until it is under way. So step N's queries choose step N+1's
residency. A policy that needs to see the query it is serving cannot exist in a
pager at all, which is worth stating plainly: what is being bet on is that
attention moves slowly enough between adjacent tokens for last step's ranking
to be a good guide to this one's.

Bounds are computed from a block's keys while it is resident and kept after it
leaves, which is the only order that works -- a block's keys are unreadable
once it is gone, so a block never scored while present can never be scored at
all, and would never be asked back.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .bounds import compute_bounds


class QuestScorer:
    """Per-request block bounds, and a ranking of them against a query.

    Memory is the thing to watch here rather than time: bounds are
    `num_kv_heads * head_size * 2` values per block per layer, which is a
    double-digit percentage of a block on a small-headed model and comes out of
    the same budget the blocks do. Stored in fp32 for arithmetic reasons and
    not yet in whatever the cheapest faithful format is.
    """

    def __init__(self, head_size: int, head_size_v: int | None = None,
                 head_agg: str = "max", layer_agg: str = "max",
                 decay: float = 1.0) -> None:
        self.head_size = head_size
        self.head_size_v = head_size_v
        self.head_agg = head_agg
        self.layer_agg = layer_agg
        #: How much of a block's standing comes from this step. 1.0 ranks on
        #: the current step alone, which is what makes a scored policy thrash:
        #: it re-decides residency from scratch every token and the set moves
        #: with it. Below 1.0 the ranking is evidence accumulated over the
        #: request -- a new signal is acted on at once, an old one fades
        #: instead of vanishing, so a block stays resident for a while after it
        #: stops being indicated.
        self.decay = decay
        #: (req_id, logical index) -> [layers, kv_heads, 2, head_size]
        self.bounds: dict[tuple[str, int], torch.Tensor] = {}
        #: (req_id, logical index) -> accumulated standing, in [0, 1]
        self.standing: dict[tuple[str, int], float] = {}
        self.computed = 0

    def observe(self, req_id: str, index: int, caches: Sequence[torch.Tensor],
                block_id: int) -> None:
        """Take a block's bounds while it is still readable. Idempotent."""
        key = (req_id, index)
        if key in self.bounds:
            return
        self.bounds[key] = compute_bounds(caches, block_id, self.head_size,
                                          self.head_size_v)
        self.computed += 1

    def forget(self, req_id: str) -> None:
        for key in [k for k in self.bounds if k[0] == req_id]:
            del self.bounds[key]
        for key in [k for k in self.standing if k[0] == req_id]:
            del self.standing[key]

    def known(self, req_id: str, indices: Sequence[int]) -> list[int]:
        return [i for i in indices if (req_id, i) in self.bounds]

    def rank(self, req_id: str, indices: Sequence[int],
             queries: Sequence[torch.Tensor]) -> list[tuple[int, float]]:
        """Blocks ordered by how high anything in them could score. Best first.

        Only blocks whose bounds are known can be ranked; the caller decides
        what to do with the rest, and treating an unscored block as unwanted
        would be a way to lose one forever.
        """
        known = self.known(req_id, indices)
        if not known or not queries:
            return []
        stacked = torch.stack([self.bounds[(req_id, i)] for i in known])
        from .bounds import bound_scores

        scores = bound_scores(queries, stacked, head_agg=self.head_agg,
                              layer_agg=self.layer_agg)
        if self.decay >= 1.0:
            order = torch.argsort(scores, descending=True).tolist()
            return [(known[i], float(scores[i])) for i in order]

        # A step's scores become a distribution before they are accumulated.
        # Their scale rides on the query's norm, so mixing raw scores across
        # steps would let one step's magnitude outvote another's shape --
        # which is the opposite of what accumulating is for. Min-max rather
        # than softmax: a softmax over bounds this widely spread is nearly
        # one-hot, and would accumulate almost as spikily as no smoothing.
        low, high = float(scores.min()), float(scores.max())
        spread = high - low
        shares = ((scores - low) / spread) if spread > 0 else torch.ones_like(scores)
        shares = shares / float(shares.sum())

        blended = []
        for i, index in enumerate(known):
            key = (req_id, index)
            share = float(shares[i])
            # A block seen for the first time starts at what it is worth now,
            # not at zero: a new signal has to be actionable immediately or
            # smoothing becomes a refusal to notice anything.
            previous = self.standing.get(key, share)
            value = self.decay * share + (1.0 - self.decay) * previous
            self.standing[key] = value
            blended.append(value)

        order = sorted(range(len(known)), key=lambda i: -blended[i])
        return [(known[i], blended[i]) for i in order]

    def stats(self) -> dict:
        return {"blocks_scored": len(self.bounds), "computed": self.computed}


class QueryCapture:
    """Last step's per-layer queries, taken on the way into attention.

    A forward pre-hook rather than anything cleverer, which makes this
    eager-only: hooks do not run inside a replayed CUDA graph. Queries are kept
    per request row, since a batch's rows belong to different requests and a
    ranking is per request.
    """

    def __init__(self) -> None:
        self.installed = False
        self.layers: list[torch.Tensor] = []
        self.previous: list[torch.Tensor] = []
        self.head_size = 0
        self.num_heads = 0

    def install(self, model) -> None:
        if self.installed:
            return
        try:
            from vllm.model_executor.layers.attention import Attention
        except ImportError:                      # pre-rename tree
            from vllm.attention.layer import Attention

        found = [(n, m) for n, m in model.named_modules()
                 if isinstance(m, Attention)]
        for _, module in found:
            module.register_forward_pre_hook(self._record, with_kwargs=True)
            self.head_size = getattr(module, "head_size", self.head_size)
            self.num_heads = getattr(module, "num_heads", self.num_heads)
        self.installed = bool(found)

    def _record(self, module, args, kwargs):
        query = kwargs.get("query") if kwargs else None
        if query is None and args:
            query = args[0]
        if isinstance(query, torch.Tensor):
            self.layers.append(query.detach())
        return None

    def rotate(self) -> None:
        """Make this step's captures the ones the next step will score with."""
        if self.layers:
            self.previous = self.layers
        self.layers = []

    def for_row(self, row: int) -> list[torch.Tensor]:
        """Last step's queries for one batch row, as [num_heads, head_size]."""
        out = []
        for q in self.previous:
            if q.ndim == 2 and row < q.shape[0]:
                out.append(q[row].reshape(-1, self.head_size))
            elif q.ndim == 3 and row < q.shape[0]:
                out.append(q[row])
            else:
                return []
        return out
