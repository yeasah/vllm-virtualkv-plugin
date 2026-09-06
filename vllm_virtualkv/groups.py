"""Which KV cache group is ours, on a model that has more than one.

A hybrid model -- Qwen3.5 and its relatives put full attention in a minority
of layers and linear/GDN state in the rest -- gives vLLM *two* KV cache
groups, and only one of them is paged. Everything the pager touches is
per-group or per-layer, and the group it wants is not reliably first:

- `block_tables`, `slot_mappings` and `kernel_block_sizes` are indexed by
  group, so using index 0 can rewrite the linear group's table.
- `runner.kv_caches` is ordered by *layer index across every layer that has
  any state at all* (see `bind_kv_cache`), so on a hybrid it interleaves K/V
  caches with mamba state. Handing that list to `block_keys` means slicing
  conv/ssm state as though the first `head_size` channels were keys.

Neither mistake crashes. They produce plausible tensors, which is the
failure class that has cost this project the most, so this resolves the
group once and **asserts** what it found rather than trusting an index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GroupError(RuntimeError):
    """The paged group could not be identified, so nothing may proceed.

    Raised rather than falling back to group 0: a wrong group is silent, and
    a pager that silently pages the wrong tensor is worse than one that
    refuses to start.
    """


@dataclass
class PagedGroup:
    """Where the paged layers live, in every indexing scheme that matters."""

    index: int
    #: Positions within `runner.kv_caches`, in layer order.
    cache_positions: list[int]
    layer_names: list[str]
    spec: Any

    def caches(self, runner_kv_caches: list) -> list:
        """Just the paged layers' caches, in layer order."""
        return [runner_kv_caches[i] for i in self.cache_positions]


def _is_paged(spec: Any) -> bool:
    # By duck type rather than isinstance: the spec class is built at import
    # time against whatever base the installed vLLM offers, so a subclass
    # check would tie this to that construction.
    return all(hasattr(spec, f) for f in
               ("budget_blocks", "sink_blocks", "policy_name"))


def resolve(runner) -> PagedGroup:
    """Find the one group whose spec is ours, or refuse."""
    try:
        from vllm.v1.worker.utils import extract_layer_index
    except ImportError as exc:                      # pragma: no cover
        raise GroupError(f"cannot map layer names to indices: {exc}") from exc

    config = getattr(runner, "kv_cache_config", None)
    groups = getattr(config, "kv_cache_groups", None)
    if not groups:
        raise GroupError("the runner exposes no kv_cache_groups")

    found = [(i, g) for i, g in enumerate(groups) if _is_paged(g.kv_cache_spec)]
    if not found:
        kinds = ", ".join(type(g.kv_cache_spec).__name__ for g in groups)
        raise GroupError(
            f"no paged group among {len(groups)} ({kinds}). The spec patch "
            f"did not reach this model's attention layers, so paging would "
            f"have run against someone else's cache")
    if len(found) > 1:
        raise GroupError(
            f"{len(found)} paged groups; this assumes exactly one block "
            f"table is paged and would need a per-group pager otherwise")

    index, group = found[0]

    # Mirror bind_kv_cache: runner.kv_caches is ordered by layer index over
    # every layer holding state, across all groups.
    every = [n for g in groups for n in g.layer_names]
    order = sorted({extract_layer_index(n) for n in every})
    position = {idx: i for i, idx in enumerate(order)}
    try:
        positions = sorted(position[extract_layer_index(n)]
                           for n in group.layer_names)
    except KeyError as exc:                          # pragma: no cover
        raise GroupError(f"layer {exc} is not in the cache ordering") from exc

    caches = getattr(runner, "kv_caches", None)
    if caches is not None and len(caches) != len(order):
        raise GroupError(
            f"kv_caches has {len(caches)} entries against {len(order)} layers "
            f"with state; the ordering assumption does not hold here")

    return PagedGroup(index=index, cache_positions=positions,
                      layer_names=list(group.layer_names),
                      spec=group.kv_cache_spec)
