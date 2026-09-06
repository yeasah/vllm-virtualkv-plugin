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
    assert message and "VLLM_VIRTUALKV_HOST_SLOTS=448" in message


def test_evicting_nothing_needs_no_tier():
    needed, message = check_host_tier_size(
        Config(budget=0, host_slots=1), fake_config(1 << 20, 16, 64))
    assert (needed, message) == (0, None)
