"""Residency policies, called from both sides of the engine.

The scheduler decides what to free and the worker decides what the kernel may
read, and those two decisions have to be the same one. The cheapest way to
guarantee that is not two implementations reviewed against each other but a
single pure function that both import -- so the policy takes only what both
sides already know, and takes no engine objects at all.

The clock is `num_computed_tokens`, because it is the one quantity the
scheduler and the worker both have and both agree on. A wall-clock step counter
would drift the moment one side skipped a step.

Every policy returns row indices of *full* blocks, ascending. The partial tail
block is never a policy decision: it holds the key this step is about to write,
so it is always resident and always last, and the callers append it.
"""

from __future__ import annotations

import math

from collections.abc import Sequence

from typing import Protocol


class Policy(Protocol):
    """What every policy here is, so a new one has something to conform to.

    `resident` answers one question -- given a request with `n_full` completely
    full blocks and `num_computed` tokens committed, which of those blocks
    should be on the GPU -- and answers it as ascending *logical* row indices.
    It must be a pure function of its arguments: the scheduler and the worker
    both call it, on different clocks, and anything it remembers between calls
    would drift between the two.

    The partial tail block is never returned. It holds the key the current step
    is about to write, so it is always resident and always last, and the
    callers append it.
    """

    name: str

    def resident(self, n_full: int, num_computed: int) -> list[int]: ...


class Recency:
    """Sinks plus the most recent blocks -- StreamingLLM's set.

    Shippable with no calibration, and the honest baseline for anything
    cleverer. Note what it cannot do: its window only ever slides forward, so
    it never asks for a block back and its fetch rate is zero after warmup.
    A pager running this policy exercises eviction and never exercises
    restore, which is why `Stress` exists.
    """

    name = "recency"

    def __init__(self, budget: int, sink: int = 2) -> None:
        self.budget = budget
        self.sink = sink

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if n_full <= self.budget:
            return list(range(n_full))
        sink = min(self.sink, self.budget)
        keep = list(range(sink))
        keep += list(range(n_full - (self.budget - sink), n_full))
        return sorted(set(keep))


class Stress:
    """Deliberately churns the resident set, to exercise the restore path.

    Not a policy anyone would ship: it exists because `Recency` never fetches,
    so a pager tested only under recency would leave its entire restore path --
    the whole difference between paging and eviction -- unexercised. This keeps
    the sinks and the newest blocks, then fills the remaining budget with a
    window that walks backwards through the middle of the context, so blocks
    leave and are asked for again at a rate the caller sets.

    `churn` is how many blocks the walking window advances per decode step,
    which makes the fetch rate a dial: it is the knob that turns "policy error
    costs latency" from an assertion into a measurement.
    """

    name = "stress"

    def __init__(self, budget: int, sink: int = 2, recent: int = 4,
                 churn: int = 1) -> None:
        self.budget = budget
        self.sink = sink
        self.recent = recent
        self.churn = churn

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if n_full <= self.budget:
            return list(range(n_full))
        sink = min(self.sink, self.budget)
        recent = min(self.recent, max(0, self.budget - sink))
        keep = set(range(sink))
        keep |= set(range(n_full - recent, n_full))
        roam = self.budget - len(keep)
        if roam > 0:
            span = max(1, n_full - sink - recent)
            start = (num_computed // max(1, self.churn)) % span
            for i in range(roam):
                keep.add(sink + (start + i) % span)
        return sorted(x for x in keep if 0 <= x < n_full)


class Churn:
    """Cycles blocks out and asks for every one of them straight back.

    Saves no memory, and is not trying to: it exists so the machinery can be
    tested against the strongest reference there is. Every step it evicts a
    rotating window and requests the whole of the previous window back, so
    each block makes a full round trip -- copied to the host, freed,
    reallocated somewhere else, copied back -- while the model never loses
    sight of anything. **The output must therefore be bit-identical to running
    without the plugin**, which no policy that actually drops context can be
    compared against.

    That is the end-to-end version of a proof this project otherwise only has
    in pieces: the host tier round trip is bit-exact in isolation, and a
    hand-driven relocation is bit-exact through the model, but nothing until
    now exercised the *manager's own* evict-and-restore path against an exact
    reference.

    `budget` is read as the number of blocks to cycle per step, not as a
    resident count -- the one policy here where that argument means something
    different, because "how much do you keep" is not the question it answers.
    Requires `show_pending`, since a block is only still readable during the
    step it was chosen in.

    **Consecutive windows must not overlap**, which is the whole contract and
    is easy to get wrong twice. A block chosen again on the step after it was
    chosen has already been freed, so it is neither resident nor pending, and
    it quietly stays away instead of cycling -- context lost, with every
    counter still reading clean.

    The first attempt advanced a sliding window by one block, which overlaps
    obviously. The second advanced it by its own width, computing the offset
    modulo the number of full blocks -- which is disjoint only while that
    number holds still, and it does not: the moment the request fills another
    block the modulus changes and the window can land back on what it just
    evicted. That failure showed up as a divergence appearing at exactly the
    step the context crossed a block boundary.

    So the window is not a moving offset at all. Blocks are grouped into runs
    of `width` and a run is evicted when its index falls in the step's residue
    class mod `PHASES`. Which blocks those are depends only on the block index
    and the step, never on how many blocks exist, so growing the context cannot
    make two consecutive steps overlap.
    """

    name = "churn"
    #: How many steps before a block can be chosen again. Two would suffice for
    #: disjointness; four keeps each step's traffic to about a quarter of the
    #: context.
    PHASES = 4

    def __init__(self, budget: int, sink: int = 0) -> None:
        self.width = max(1, budget)
        self.sink = sink

    def window(self, n_full: int, num_computed: int) -> set[int]:
        if n_full <= self.sink:
            return set()
        phase = num_computed % self.PHASES
        return {i for i in range(self.sink, n_full)
                if ((i - self.sink) // self.width) % self.PHASES == phase}

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        return sorted(set(range(n_full)) - self.window(n_full, num_computed))


class Oracle:
    """Recency, plus a set of blocks the caller says matter. The ceiling.

    Not a policy that could exist in production -- it is told the answer. It
    exists to separate two questions that a bad quality number cannot
    distinguish on its own: whether the *machinery* can deliver good output at
    a low budget, and whether a *policy* can find the right blocks. If the
    oracle retrieves at a budget where recency does not, the mechanism is fine
    and the remaining gap is policy, which is the phase this project is trying
    to reach. If even the oracle fails, nothing about scoring will help.

    `must_keep` is a class attribute because a policy is constructed from a
    frozen spec with nowhere to thread a set through. That is fine for an
    instrument and would not be for anything else.
    """

    name = "oracle"
    must_keep: set[int] = set()

    def __init__(self, budget: int, sink: int = 2) -> None:
        self.budget = budget
        self.sink = sink

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if n_full <= self.budget:
            return list(range(n_full))
        sink = min(self.sink, self.budget)
        keep = set(range(sink))
        keep |= {i for i in self.must_keep if 0 <= i < n_full}
        keep = set(sorted(keep)[: self.budget])
        recent = self.budget - len(keep)
        keep |= set(range(n_full - recent, n_full))
        return sorted(i for i in keep if 0 <= i < n_full)


class OracleLate(Oracle):
    """An oracle that only wants the block back after `after` tokens.

    The point is the round trip, not the policy. Plain `Oracle` keeps the block
    it was told about from the beginning, so it never restores anything and a
    good answer from it proves only that residency works. This one lets the
    block be evicted during prefill -- its contents going out to the host tier
    -- and asks for it back at decode time, so answering correctly means the
    bytes made the journey and still mean what they meant. That is the
    end-to-end version of the bit-exactness the host tier proves in isolation,
    and the only arrangement in which a needle can test it: the answer has to
    be generated after the restore.
    """

    name = "oracle_late"
    after: int = 1 << 60

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        if num_computed < self.after:
            keep = set(range(min(self.sink, self.budget)))
            recent = self.budget - len(keep)
            keep |= set(range(n_full - recent, n_full))
            return sorted(i for i in keep if 0 <= i < n_full)
        return super().resident(n_full, num_computed)


class Quest:
    """Residency ranked by an upper bound on each block's attention score.

    The scoring happens on the worker (see `quest.py`); this class exists so
    the manager has something to fall back on before the first ranking arrives,
    and so the policy name means something on both sides. Until the worker has
    seen a query and taken a block's bounds, this behaves as recency -- which
    is also what it degrades to for any block whose bounds were never captured,
    since a block that cannot be scored must not be silently treated as
    unwanted.
    """

    name = "quest"

    def __init__(self, budget: int, sink: int = 2) -> None:
        self.budget = budget
        self.sink = sink
        self._fallback = Recency(budget, sink)

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        return self._fallback.resident(n_full, num_computed)


class MassOracle:
    """Residency ranked by the *measured* attention mass of the previous step.

    Not shippable and not meant to be: it reconstructs every block's keys and
    softmaxes them each step, which costs more than the attention it is
    steering. It exists to answer a question that has to be settled before any
    estimator is worth building -- whether capturing attention mass actually
    buys output quality.

    Every scored policy is chasing this ceiling. `quest` with min/max bounds
    holds 0.66 of the mass, a 2-bit key summary holds 0.87, and this holds
    0.87 with no estimator at all, because it *is* the truth one step stale.
    If that does not convert into output quality well above `recency` (0.27),
    then mass capture is the wrong objective and a better estimator of it is
    wasted work. This repo has already seen the proxy and the outcome
    disagree, which is why the ceiling gets tested before the machine.
    """

    name = "massoracle"

    def __init__(self, budget: int, sink: int = 2) -> None:
        self.budget = budget
        self.sink = sink
        self._fallback = Recency(budget, sink)

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        return self._fallback.resident(n_full, num_computed)


class ImpactOracle:
    """Residency ranked by how much dropping a block would move the output.

    The other ceiling, and the one that settles the premise. `MassOracle`
    ranks on attention mass and matched recency exactly (0.2372 against
    0.2374), so there is no headroom in that quantity for any estimator of
    it to find. This ranks on `m*(o - v_b)/(1 - m)` -- mass weighted by how
    far a block's values sit from the output attention was already producing
    -- which is what a demand signal would actually be trying to predict.

    If this cannot beat recency either, the demand-signal premise is dead
    rather than merely the signals tried so far.
    """

    name = "impactoracle"

    def __init__(self, budget: int, sink: int = 2) -> None:
        self.budget = budget
        self.sink = sink
        self._fallback = Recency(budget, sink)

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        return self._fallback.resident(n_full, num_computed)


class SetOracle:
    """Residency chosen as a *set*, by greedy joint minimisation.

    The last question the demand-signal work leaves: the marginal oracle
    ranks blocks by their own leave-one-out shift, and dropping a set is not
    the sum of dropping its members -- the cost is a norm of a sum, so error
    vectors cancel. This picks greedily against the joint objective instead,
    which is the true ceiling for selection.

    Not shippable, and not meant to be. It exists to say *why* recency wins,
    not to beat it: if greedy joint selection converges on something shaped
    like a contiguous window, contiguity stops being a lucky heuristic and
    becomes the answer.
    """

    name = "setoracle"

    def __init__(self, budget: int, sink: int = 2) -> None:
        self.budget = budget
        self.sink = sink
        self._fallback = Recency(budget, sink)

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        return self._fallback.resident(n_full, num_computed)


class Full:
    """Everything resident. The control arm, and it must be bit-exact.

    A pager configured with this has to reproduce an unpaged run exactly; any
    deviation is a mechanism bug rather than a policy cost, which is what keeps
    a bug from being read as an accuracy result.
    """

    name = "full"

    def __init__(self, budget: int = 0, sink: int = 0) -> None:
        pass

    def resident(self, n_full: int, num_computed: int) -> list[int]:
        return list(range(n_full))


def positional_prior(n_full: int, front: float = 0.10,
                     back: float = 0.10) -> list[float]:
    """A U-shaped weight over block positions, in fractions of the context.

    Attention is reliably U-shaped: heavy at the start, heavy at the end,
    thin through the middle. This repo has measured both ends of that --
    two blocks holding 46% of all mass, and a recency window being the one
    thing no policy beat -- and has been expressing it as two hard
    reservations, `sink` and `recent`, both counted in *blocks*. That makes
    them mean different things on different models, and it spends budget in
    all-or-nothing lumps.

    As a multiplier on a demand signal it does something the reservations
    cannot: it lets a score overcome the prior when the evidence is strong,
    instead of fencing off the ends and ranking whatever is left. `quest`
    lost to `recency` in all nine long-context configurations partly by
    spending its budget through the middle; this makes the middle expensive
    rather than forbidden.

    Scale-free by construction -- `front` and `back` are fractions of the
    context, so the shape holds as a session grows rather than needing a
    block count retuned per model.
    """
    if n_full <= 0:
        return []
    fs = max(1.0, front * n_full)
    bs = max(1.0, back * n_full)
    out = []
    for i in range(n_full):
        head = math.exp(-i / fs)
        tail = math.exp(-(n_full - 1 - i) / bs)
        out.append(head + tail)
    return out


def choose(n_full: int, budget: int, sink: int, recent: int,
           unknown: Sequence[int], ranked: Sequence[int]) -> list[int]:
    """The resident set, as a priority order that is cut and then sorted.

    The order is load-bearing and the cut is where it shows. Sinks first,
    then blocks that were never scored (a block that could not be ranked must
    not be treated as unwanted), then the recent window newest-first, then the
    ranking. Reserved sets overflow the budget routinely -- early in a request
    almost nothing has been scored -- and something has to give.

    Cutting a list *sorted by block index* discards the highest-numbered
    blocks, which are the newest: the ones holding what the generation just
    wrote. That is the most valuable thing in the context and it was what went
    first. Cut by priority, sort afterwards.
    """
    keep = list(range(min(sink, n_full)))
    keep += list(unknown)
    if recent:
        keep += list(range(n_full - 1, max(0, n_full - recent) - 1, -1))
    for index in ranked:
        if len(set(keep)) >= budget:
            break
        keep.append(index)
    ordered: list[int] = []
    seen: set[int] = set()
    for i in keep:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return sorted(ordered[:max(budget, len(unknown))])


POLICIES: dict[str, type[Policy]] = {
    p.name: p for p in (Recency, Stress, Churn, Quest, MassOracle, ImpactOracle, SetOracle,
                        Oracle,
                        OracleLate, Full)
}
