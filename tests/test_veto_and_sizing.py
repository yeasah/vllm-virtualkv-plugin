"""The two things that keep a full host tier from becoming a lost block.

A budget is a promise that everything not resident is somewhere else. If the
tier cannot take a block, the only safe answer is to leave it on the GPU --
holding VRAM the budget said would be free, which is a broken promise, rather
than freeing a block whose only copy is the one being freed, which is a lost
one.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm_virtualkv import state as pager_state  # noqa: E402
from vllm_virtualkv.config import Config  # noqa: E402
from vllm_virtualkv.integration import check_host_tier_size  # noqa: E402

from test_manager import BLOCK, give_blocks, make_manager, null_positions  # noqa: E402


def test_a_refused_eviction_is_not_freed():
    mgr, pool = make_manager(budget=8)
    give_blocks(mgr, pool, "r", 20)

    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)
    chosen = sorted(mgr.pending_evictions["r"])
    assert chosen, "nothing was chosen for eviction"

    # The worker could not copy two of them out.
    refused = set(chosen[:2])
    mgr.state.get("r").refused = set(refused)

    free_before = pool.get_num_free_blocks()
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)

    nulls = set(null_positions(mgr, "r"))
    assert not (refused & nulls), (
        "a block the worker could not back up was freed anyway; its only copy "
        "was the one just returned to the pool"
    )
    assert pool.get_num_free_blocks() == free_before + len(chosen) - len(refused)
    assert mgr.evictions_refused == len(refused)


def test_a_refused_block_is_chosen_again_once_there_is_room():
    """The veto has to heal, or one full moment costs the budget forever."""
    mgr, pool = make_manager(budget=8)
    give_blocks(mgr, pool, "r", 20)
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)
    chosen = sorted(mgr.pending_evictions["r"])
    refused = set(chosen[:2])
    mgr.state.get("r").refused = set(refused)
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)

    # Tier has room again: nothing is refused this time.
    mgr.state.get("r").refused = set()
    assert refused <= set(mgr.pending_evictions["r"]), (
        "the refused blocks were not re-chosen, so the budget is permanently "
        "larger than it was asked to be"
    )
    mgr.remove_skipped_blocks("r", processed_computed_tokens=20 * BLOCK)
    assert refused <= set(null_positions(mgr, "r"))


def fake_config(max_len, block_size, max_num_seqs):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        model_config=SimpleNamespace(max_model_len=max_len),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_host_tier_requirement_is_the_displaced_blocks():
    cfg = Config(budget=16, host_slots=10_000)
    needed, message = check_host_tier_size(cfg, fake_config(2048, 16, 4))
    assert needed == 4 * (128 - 16) == 448
    assert message is None


def test_an_undersized_tier_is_reported_with_the_number_to_set():
    cfg = Config(budget=16, host_slots=100)
    needed, message = check_host_tier_size(cfg, fake_config(2048, 16, 4))
    assert needed == 448
    # The number, and the advice that the better fix is not to set it at all.
    assert message and "448" in message and "Unset it" in message


def test_evicting_nothing_needs_no_tier():
    needed, message = check_host_tier_size(
        Config(budget=0, host_slots=1), fake_config(1 << 20, 16, 64))
    assert (needed, message) == (0, None)


def test_an_unset_tier_sizes_itself_and_says_nothing():
    """The point of the rule: a knob checked against a threshold defaults to it.

    An operator has a card, a model and a concurrency target. Every input to
    this number is something the engine already knows, so making them look it
    up and type it back is busywork that also gets stale the moment any of the
    three changes.
    """
    needed, message = check_host_tier_size(
        Config(budget=16, host_slots=None), fake_config(2048, 16, 4))
    assert needed == 448
    assert message is None, "auto-sizing should not warn about itself"


def test_the_requirement_tracks_the_things_it_is_derived_from():
    from vllm_virtualkv.integration import required_host_slots

    base = required_host_slots(16, fake_config(2048, 16, 4))
    assert required_host_slots(16, fake_config(4096, 16, 4)) > base, "context"
    assert required_host_slots(16, fake_config(2048, 16, 8)) > base, "concurrency"
    assert required_host_slots(64, fake_config(2048, 16, 4)) < base, "budget"
    assert required_host_slots(0, fake_config(1 << 20, 16, 64)) == 0


def test_a_budget_can_be_written_in_the_unit_that_makes_sense():
    from vllm_virtualkv.config import parse_budget, resolve_budget

    engine = fake_config(8192, 16, 4)
    assert resolve_budget(parse_budget("64"), engine) == 64
    assert resolve_budget(parse_budget("1024t"), engine) == 64
    assert resolve_budget(parse_budget("25%"), engine) == 128
    assert resolve_budget(parse_budget(0), engine) == 0


def test_the_same_budget_means_the_same_thing_at_a_different_block_size():
    """The reason blocks are the wrong unit: they are not the operator's.

    A sweep written in blocks is not a sweep of the same quantity once the
    engine's block size changes, so results taken at one geometry cannot be
    compared with another.
    """
    from vllm_virtualkv.config import parse_budget, resolve_budget

    spec = parse_budget("1024t")
    assert resolve_budget(spec, fake_config(8192, 16, 1)) == 64
    assert resolve_budget(spec, fake_config(8192, 32, 1)) == 32   # same tokens
    blocks = parse_budget("64")
    assert resolve_budget(blocks, fake_config(8192, 16, 1)) == 64
    assert resolve_budget(blocks, fake_config(8192, 32, 1)) == 64  # twice the tokens


def test_a_budget_that_cannot_be_read_is_refused_at_startup():
    from vllm_virtualkv.config import parse_budget

    for bad in ("abc", "150%", "-5%", "12x"):
        with pytest.raises(ValueError):
            parse_budget(bad)


def test_a_sink_larger_than_the_resolved_budget_is_caught_late():
    """`25%` cannot be checked against a sink until an engine says how big it is."""
    cfg = Config(budget="1%", sink=8)
    cfg.validate()                       # nothing knowable yet
    with pytest.raises(ValueError, match="exceeds the resolved budget"):
        cfg.resolve(fake_config(2048, 16, 1))      # 1% of 2048 = 1 block


def test_the_tier_is_sized_from_the_resolved_budget_not_the_written_one():
    """A budget written as a share is not a number until an engine exists.

    The worker is constructed before one does, so capturing the budget at
    construction gave `required_host_slots(0, ...)` -- a tier of one slot,
    which refuses nearly every eviction and quietly stops paging. The plugin
    kept working, because a refused eviction leaves the block resident, so the
    only symptoms were a refusal counter in the tens of thousands and no
    memory being saved.
    """
    from vllm_virtualkv.integration import required_host_slots
    from vllm_virtualkv.worker import WorkerPager

    engine = fake_config(8192, 16, 1)
    cfg = Config(budget="25%")
    pager = WorkerPager(config=cfg)

    assert cfg.budget == 0, "the premise: unresolved at construction"
    cfg.resolve(engine)                      # what the spec hook does later
    assert cfg.budget == 128

    # Faithful enough for the group resolver: one paged group, one layer.
    # A stub without groups is now refused rather than silently paging
    # whatever tensor happened to be first, which is the point of
    # `groups.resolve`.
    class FakeSpec:
        budget_blocks = 128
        sink_blocks = 2
        policy_name = "recency"
        head_size = 2
        head_size_v = 2

    class FakeGroup:
        layer_names = ["model.layers.0.self_attn.attn"]
        kv_cache_spec = FakeSpec()

    class FakeConfig:
        kv_cache_groups = [FakeGroup()]

    class FakeRunner:
        vllm_config = engine
        kv_caches = [torch.zeros((4, 1, 1, 2))]
        kv_cache_config = FakeConfig()

    pager._attach(FakeRunner())
    assert pager.budget == 128
    assert pager.tier.num_slots == required_host_slots(128, engine) == 384, (
        f"tier sized {pager.tier.num_slots} for a budget the worker read as "
        f"{pager.budget}")
