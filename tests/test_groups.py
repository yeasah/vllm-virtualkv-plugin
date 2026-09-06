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
