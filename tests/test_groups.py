"""Picking our KV cache group on a model that has more than one.

A hybrid puts full attention in a minority of layers and linear/GDN state in
the rest, so vLLM builds two groups and only one is paged. Nothing here
crashes when it is got wrong: `block_tables[0]` may be the linear group's
table, and `runner.kv_caches` is ordered by layer index across *every* layer
with state, so slicing it wholesale feeds conv/ssm state to key extraction as
though the first `head_size` channels were keys.
"""

import pytest

from vllm_virtualkv.groups import GroupError, resolve


class Spec:
    """A paged spec, recognised by its paging fields rather than its class."""
    budget_blocks = 64
    sink_blocks = 2
    policy_name = "recency"


class Mamba:
    """Linear-attention state: no paging fields, so not ours."""
    conv_state_shape = (4, 128)


def runner(groups, n_caches=None):
    class Group:
        def __init__(self, names, spec):
            self.layer_names, self.kv_cache_spec = names, spec

    class Config:
        kv_cache_groups = [Group(n, s) for n, s in groups]

    class Runner:
        kv_cache_config = Config()
        kv_caches = list(range(
            n_caches if n_caches is not None
            else sum(len(n) for n, _ in groups)))
    return Runner()


def names(*ix):
    return [f"model.layers.{i}.self_attn.attn" for i in ix]


def test_qwen3_5_shape_full_attention_every_fourth_layer():
    # 8 full-attention layers of 32, which is the case this exists for.
    full = list(range(3, 32, 4))
    linear = [i for i in range(32) if i not in full]
    g = resolve(runner([(names(*linear), Mamba()), (names(*full), Spec())]))
    assert g.index == 1, "the paged group is not first here, and that is why"
    # Positions into runner.kv_caches, which is ordered by layer index over
    # all 32 layers -- so they are the layer numbers themselves.
    assert g.cache_positions == full
    assert g.caches(list(range(32))) == full


def test_paged_group_first_still_works():
    g = resolve(runner([(names(0, 1), Spec()), (names(2, 3), Mamba())]))
    assert g.index == 0 and g.cache_positions == [0, 1]


def test_uniform_model_has_one_group():
    g = resolve(runner([(names(0, 1, 2), Spec())]))
    assert g.index == 0 and g.cache_positions == [0, 1, 2]


def test_no_paged_group_refuses_rather_than_defaulting_to_zero():
    with pytest.raises(GroupError, match="no paged group"):
        resolve(runner([(names(0, 1), Mamba())]))


def test_two_paged_groups_refuse():
    with pytest.raises(GroupError, match="2 paged groups"):
        resolve(runner([(names(0), Spec()), (names(1), Spec())]))


def test_cache_ordering_mismatch_refuses():
    # More layers with state than caches: the ordering assumption is broken,
    # and proceeding would index the wrong tensors.
    with pytest.raises(GroupError, match="ordering assumption"):
        resolve(runner([(names(0, 1), Spec()), (names(2, 3), Mamba())],
                       n_caches=3))


def test_missing_groups_refuses():
    class Bare:
        kv_cache_config = None
    with pytest.raises(GroupError, match="no kv_cache_groups"):
        resolve(Bare())


def test_a_pager_that_never_fired_refuses_to_report():
    """Installed and executed are different properties.

    vLLM carries two GPU model runners with the same class name and picks
    between them per model, so a patch can land on the one the live path
    never touches. Everything else then looks healthy: no error, correct
    output, a clean guard, and zero transfers -- indistinguishable from a
    working plugin until someone checks a counter.
    """
    import pytest
    from vllm_virtualkv.worker import WorkerPager

    pager = WorkerPager()
    assert pager.fired == 0
    with pytest.raises(RuntimeError, match="never ran"):
        pager.fired_or_raise()

    pager.fired = 1
    pager.fired_or_raise()          # having run, it says nothing


def test_impact_closed_form_equals_dropping_the_block():
    """`m*(o - v_b)/(1 - m)` is the exact renormalised output shift.

    The importance oracle avoids n extra attention passes per step by using
    this identity, so it is only worth anything if it is actually equal to
    what it replaces.
    """
    import torch

    torch.manual_seed(0)
    kv, grp, nb, bs, d = 2, 3, 5, 4, 8
    n = nb * bs
    q, k, v = torch.randn(kv, grp, d), torch.randn(kv, n, d), torch.randn(kv, n, d)
    w = torch.softmax(torch.einsum("kgd,knd->kgn", q, k), -1)
    o = torch.einsum("kgn,knd->kgd", w, v)

    wb, vb = w.reshape(kv, grp, nb, bs), v.reshape(kv, nb, bs, d)
    m = wb.sum(-1)
    vbar = (torch.einsum("kgnb,knbd->kgnd", wb, vb)
            / m.clamp(min=1e-12).unsqueeze(-1))
    closed = m * (o.unsqueeze(2) - vbar).norm(dim=-1) / (1 - m).clamp(min=1e-6)

    for b in range(nb):
        keep = torch.ones(n, dtype=torch.bool)
        keep[b * bs:(b + 1) * bs] = False
        w2 = w[..., keep] / w[..., keep].sum(-1, keepdim=True)
        shift = (torch.einsum("kgn,knd->kgd", w2, v[:, keep]) - o).norm(dim=-1)
        assert torch.allclose(closed[..., b], shift, atol=1e-5)


def test_vectorised_gather_matches_the_per_block_loop():
    """The fast path exists so the oracles can run at fine granularity.

    At block 16 a 64k context is 4096 blocks, so the per-block Python loop
    costs 147k slices per step across 36 layers against 1440 at block 2112.
    That is the difference between an oracle run finishing and not -- but
    only if it returns the same tensors.
    """
    import torch
    from vllm_virtualkv.workingset import _kv_for_layer

    hs = hv = 8
    nb, bs, heads = 6, 4, 3
    cache = torch.randn(20, heads, bs, hs + hv)
    row = [7, 3, 11, 2, 15, 4]

    fast_k, fast_v = _kv_for_layer([cache], None, "r", row, nb, 0, hs, hv,
                                   set(range(nb)), cache.device)
    slow_k = torch.cat([cache[row[i]][..., :hs].float() for i in range(nb)], 1)
    slow_v = torch.cat([cache[row[i]][..., hs:].float() for i in range(nb)], 1)

    assert torch.allclose(fast_k, slow_k)
    assert torch.allclose(fast_v, slow_v)


def test_greedy_joint_selection_beats_marginal_ranking():
    """The set cost is a norm of a sum, so error vectors cancel.

    `impactoracle` ranks blocks by their individual leave-one-out shift and
    lost to recency. Dropping a set costs ||sum d_b|| / (1 - sum m_b), which
    is not the sum of the individual costs -- blocks whose errors oppose can
    be dropped together nearly free, and magnitude ranking cannot see it.
    """
    import itertools

    import torch

    torch.manual_seed(1)
    LH, nb, dv, budget = 6, 10, 8, 4
    d, m = torch.randn(LH, nb, dv), torch.rand(LH, nb) * 0.06

    def cost(keep):
        drop = [b for b in range(nb) if b not in keep]
        dd, mm = d[:, drop].sum(1), m[:, drop].sum(1)
        return float((dd.norm(dim=-1) / (1 - mm).clamp(min=1e-6)).sum())

    best = min(itertools.combinations(range(nb), budget), key=cost)

    cur_d, cur_m, avail, keep = d.sum(1), m.sum(1), list(range(nb)), []
    for _ in range(budget):
        b = min(avail, key=lambda b: float(
            ((cur_d - d[:, b]).norm(dim=-1)
             / (1 - (cur_m - m[:, b])).clamp(min=1e-6)).sum()))
        keep.append(b); avail.remove(b)
        cur_d, cur_m = cur_d - d[:, b], cur_m - m[:, b]

    marginal = sorted(range(nb),
                      key=lambda b: -float(d[:, b].norm(dim=-1).sum()))[:budget]

    assert cost(keep) <= cost(best) * 1.001, "greedy should reach the optimum"
    assert cost(marginal) > cost(keep), "marginal ranking should be worse"
