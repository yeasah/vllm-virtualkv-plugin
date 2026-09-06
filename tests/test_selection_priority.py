"""What gets cut when the reserved set overflows the budget.

Sinks, unscored blocks and the recency floor are all reserved before anything
is ranked, and together they routinely exceed the budget -- early in a request
almost nothing has been scored. The cut therefore happens often, and it used
to sort by block index and drop the tail, which discards the *newest* blocks:
the ones holding what the generation just wrote.

It was found by a measurement, not by a test. `massoracle` with recent equal
to its budget should reduce exactly to `recency` and instead scored five times
worse end to end.
"""

from vllm_virtualkv.policy import choose


def test_newest_survives_an_overflowing_recent_window():
    # 2 sinks + 64 recent = 66 reserved against a budget of 64.
    sel = choose(n_full=90, budget=64, sink=2, recent=64,
                 unknown=[], ranked=list(range(90)))
    assert 89 in sel and 88 in sel, "the newest blocks were cut"
    assert len(sel) == 64


def test_newest_survives_a_large_unscored_set():
    sel = choose(n_full=90, budget=64, sink=2, recent=32,
                 unknown=list(range(40)), ranked=list(range(90)))
    assert 89 in sel, "the newest block was cut"


def test_unscored_blocks_are_never_dropped():
    # More unscored than budget: the budget yields rather than the safety rule.
    unknown = list(range(70))
    sel = choose(n_full=90, budget=64, sink=2, recent=8,
                 unknown=unknown, ranked=[])
    assert set(unknown) <= set(sel)


def test_sinks_outrank_everything():
    sel = choose(n_full=90, budget=4, sink=2, recent=64,
                 unknown=[], ranked=list(range(90)))
    assert 0 in sel and 1 in sel


def test_ranking_fills_what_reservations_leave():
    sel = choose(n_full=90, budget=10, sink=2, recent=2,
                 unknown=[], ranked=[50, 51, 52, 53, 54, 55, 56])
    assert {0, 1, 88, 89} <= set(sel)
    assert len([i for i in sel if 50 <= i <= 56]) == 6
