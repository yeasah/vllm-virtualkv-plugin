"""The worker half: move the bytes, then show the kernel the resident set.

Everything else in this package decides things. This is the part that acts, and
the order it acts in is the whole correctness argument:

    1. copy in    a block the manager restored this step holds whatever the
                  pool last left in it until this runs, and the forward is
                  about to read it
    2. copy out   a block chosen for eviction is still allocated and still
                  intact for exactly one step; this is that step
    3. view       hand the attention metadata builder its own shortened
                  `seq_lens` and block table
    4. guard      check the view names nothing it does not own

**The view goes to the metadata builder, never to `input_batch.seq_lens`.**
That field means two things at once: how many keys attention reads, and how far
through its prompt the request is -- `gpu/sample/sampler.py` classifies a row
with `seq_len < prefill_len` as still prefilling and emits no token for it. A
pager is exactly the thing that needs those two numbers to differ, so shortening
the shared tensor does not fail, it *hangs*: the row stops producing tokens, is
rescheduled forever, and the engine spins with the progress bar at zero.
`CommonAttentionMetadata` carries its own `seq_lens` and `block_table_tensor`
and has a `replace()`, so the kernel can be shown a shorter context while the
rest of the engine keeps seeing the real one.

It hooks `prepare_attn` *after* the real one, which is what keeps this step's
own KV write correct for free: the slot mapping was computed from the untouched
row, so the new key lands in the tail block, which is always resident and
always last. Nothing here has to reason about writes at all.

Steps 1 and 3 are both before the forward, which is safe for a reason worth
stating: the forward writes only the tail block, and an evicted block is never
the tail. If that ever stops being true -- a policy that evicts the block being
written -- the copy-out would race the write, so the resident set including the
tail is an invariant the guard checks rather than a convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from . import state as pager_state
from .guard import ResidencyGuard
from .hosttier import HostTier, HostTierFull
from .audit import audit_step
from .quest import QueryCapture, QuestScorer

if TYPE_CHECKING:
    # Under TYPE_CHECKING only: this package is imported by vLLM's plugin
    # loader before much of vLLM itself is importable, and none of these are
    # needed at runtime. They are here because the shapes flowing through the
    # hooks are the hardest part of this file to read from the code alone --
    # `prepared`, in particular, is a tuple of a *tuple of tensors* and a
    # single tensor, one entry per KV cache group.
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    #: What `GPUModelRunner.prepare_attn` returns: per KV cache group, the
    #: gathered block tables, and the slot mappings for this step's tokens.
    PreparedAttn = tuple[tuple[torch.Tensor, ...], torch.Tensor | None]
    #: batch index -> (logical row, resident logical indices, view seq_len)
    StepPlan = dict[int, tuple[list[int], list[int], int]]


class WorkerPager:
    """Applies the published decisions to the KV cache and the kernel's view."""

    def __init__(self, host_slots: int | None = None,
                 scheduler: Scheduler | None = None, verify: bool = True,
                 budget: int = 0, show_pending: bool = False,
                 config=None) -> None:
        #: Read at attach time rather than now, because a budget written as a
        #: share or in tokens is not resolved into blocks until an engine
        #: exists -- and this object is built before one does. Capturing the
        #: numbers here instead gave a tier sized for a budget of zero.
        self.config = config
        self.show_pending = show_pending
        #: None means derive it from the engine at attach time, which is the
        #: first moment the model length, block size and concurrency are all
        #: known here.
        self.host_slots: int | None = host_slots
        self.budget = budget
        self.scheduler = scheduler
        self.verify = verify
        self.state: pager_state.PagerState = pager_state.current()
        self.tier: HostTier | None = None
        self.guard: ResidencyGuard | None = None
        self.runner: GPUModelRunner | None = None
        self.steps: int = 0
        self.copied_in: int = 0
        self.copied_out: int = 0
        self.missing_host_copy: int = 0
        #: steps where the manager's committed token count and the worker's
        #: disagree. The two sides derive the tail block from this number, so a
        #: lag between them shifts the whole view by a block.
        self.clock_mismatch: int = 0
        #: evictions vetoed because the host tier had no room for them
        self.evictions_refused: int = 0
        #: requests whose host copies have been let go
        self.released: int = 0
        #: set when the policy is query-aware; see quest.py
        self.scorer: QuestScorer | None = None
        self.capture: QueryCapture | None = None
        self.ranked: int = 0
        self.unscored: int = 0
        #: steps where last step's queries were not available to rank with
        self.no_queries: int = 0
        #: the last set the scorer chose, kept for inspection because the
        #: shared state drops it the moment the request finishes
        self.last_selection: list[int] = []
        #: per-step attention-mass audit, when config.audit is on
        self.audit_rows: list[dict] = []
        self._last_view: dict = {}
        #: this step's decisions, read back by `view` at the metadata builder
        self._plan: StepPlan = {}

    def attach_scheduler(self, scheduler: Scheduler) -> None:
        """Turn on the ownership check, once a scheduler is reachable.

        Only possible in-process. Kept separate from construction because the
        plugin entry point runs before an engine exists, while a test holding
        an `LLM` can hand one over afterwards.
        """
        self.scheduler = scheduler
        if self.guard is not None:
            self.guard.scheduler = scheduler

    def install(self) -> None:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        if getattr(GPUModelRunner.prepare_attn, "_pager_hooked", False):
            return
        original = GPUModelRunner.prepare_attn

        def hooked(runner: GPUModelRunner, input_batch: InputBatch,
                   *args: Any, **kwargs: Any) -> PreparedAttn:
            self._attach(runner)
            self._wrap_builders(runner)
            self._sync_rows(runner, input_batch)
            out = original(runner, input_batch, *args, **kwargs)
            self.apply(runner, input_batch, out)
            return out

        hooked._pager_hooked = True
        GPUModelRunner.prepare_attn = hooked

    def _wrap_builders(self, runner: GPUModelRunner) -> None:
        """Interpose on every full-attention metadata builder, once."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        for groups in runner.attn_groups:
            for group in groups:
                if not isinstance(group.kv_cache_spec, FullAttentionSpec):
                    continue
                for builder in group.metadata_builders:
                    if getattr(builder.build, "_pager_hooked", False):
                        continue
                    original = builder.build

                    def wrapper(*a: Any, _orig=original, **kw: Any) -> Any:
                        key = "common_attn_metadata"
                        if key in kw:
                            kw[key] = self.view(kw[key])
                        elif a:
                            a = (self.view(a[0]),) + a[1:]
                        return _orig(*a, **kw)

                    wrapper._pager_hooked = True
                    builder.build = wrapper

    def _sync_rows(self, runner: GPUModelRunner, batch: InputBatch) -> None:
        """Make the worker's own block table equal the manager's mapping.

        This has to run *before* the stock `prepare_attn`, because
        `compute_slot_mappings` indexes the worker's persistent row
        positionally to decide where this step's key is written -- and that row
        is not the logical mapping. Restored blocks reach the worker through
        the append channel, so its row grows faster than the context does, and
        index `p // block_size` stops meaning position `p`. The symptom is the
        new key landing inside a *restored* block, corrupting it, which the
        guard reports as a write to something other than the last resident
        block.

        Copying the manager's row over it fixes the slot mapping by
        construction rather than by compensating for it somewhere else, and
        resetting `num_blocks` keeps the next append landing at the logical
        end.
        """
        if not self.state.steps:
            return
        tables = runner.block_tables
        table = tables.block_tables[0]
        for b in range(batch.num_reqs):
            if int(batch.num_scheduled_tokens[b]) != 1:
                continue
            step = self.state.get(batch.req_ids[b])
            if step is None or not step.row:
                continue
            req_idx = int(batch.idx_mapping_np[b])
            row = torch.tensor(step.row, dtype=torch.int32, device="cpu")
            table.gpu[req_idx, :len(step.row)] = row.to(table.gpu.device)
            tables.num_blocks.np[0, req_idx] = len(step.row)
        tables.num_blocks.copy_to_uva()

    def view(self, common: CommonAttentionMetadata) -> CommonAttentionMetadata:
        """Replace the kernel's context length and block table with the view."""
        if not self._plan:
            return common
        seq_lens = common.seq_lens.clone()
        table = common.block_table_tensor.clone()
        for b, (row, resident, seq_len) in self._plan.items():
            if b >= table.shape[0]:
                continue
            for slot, idx in enumerate(resident):
                table[b][slot] = row[idx]
            seq_lens[b] = seq_len
        return common.replace(seq_lens=seq_lens, block_table_tensor=table)

    def _attach(self, runner: GPUModelRunner) -> None:
        if self.tier is None:
            from .integration import required_host_slots

            self.runner = runner
            budget, slots = self.budget, self.host_slots
            if self.config is not None:
                budget, slots = self.config.budget, self.config.host_slots
                self.budget, self.host_slots = budget, slots
            if slots is None:
                slots = max(1, required_host_slots(budget, runner.vllm_config))
            self.host_slots = slots
            self.tier = HostTier(runner.kv_caches, slots)
            self.guard = ResidencyGuard(self.scheduler)
            if self.config is not None and (self.config.policy == "quest"
                                            or self.config.audit):
                spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
                self.scorer = QuestScorer(
                    head_size=spec.head_size, head_size_v=spec.head_size_v,
                    head_agg=self.config.head_agg,
                    layer_agg=self.config.layer_agg)
                self.capture = QueryCapture()
                self.capture.install(runner.model)
                self._spec = spec

    def apply(self, runner: GPUModelRunner, batch: InputBatch,
              prepared: PreparedAttn) -> None:
        block_tables, slot_mappings = prepared
        if not block_tables:
            return
        self._plan = {}
        if self.capture is not None:
            # This forward's queries have not happened yet; what is held is the
            # previous one's, which is what the ranking is allowed to use.
            self.capture.rotate()
            if self.config is not None and self.config.audit:
                self._audit_previous(runner)
        self._release_finished()
        table = block_tables[0]
        slots = slot_mappings[0] if slot_mappings is not None else None
        block_size = runner.block_tables.kernel_block_sizes[0]
        caches = runner.kv_caches
        seq_lens = batch.seq_lens
        intended = {}

        for b in range(batch.num_reqs):
            req_id = batch.req_ids[b]
            step = self.state.get(req_id)
            if step is None or not step.resident:
                continue
            # Transport follows the manager's decisions wherever they are made;
            # the *view* is decode-only. Tying both to decode steps is what let
            # a block be freed during a prefill chunk with nothing having
            # copied it out -- silently, since the guard only inspects decode
            # rows too.
            decoding = int(batch.num_scheduled_tokens[b]) == 1
            self.steps += decoding
            computed = int(batch.num_computed_tokens_np[b])
            if step.num_computed != computed:
                self.clock_mismatch += 1

            # 1. copy in, before anything reads the restored blocks
            for idx, block_id in step.restored:
                key = (req_id, idx)
                if key in self.tier:
                    self.tier.load(key, caches, block_id)
                    self.copied_in += 1
                else:
                    # A restore with nothing behind it: the block was never
                    # evicted, or its host copy was dropped. Counted rather
                    # than ignored -- it means the model is about to read
                    # whatever the pool left there.
                    self.missing_host_copy += 1

            # 2. copy out, while the chosen blocks are still allocated. A
            # block that will not fit in the host tier is *refused* rather than
            # dropped: the manager reads this back and leaves it allocated,
            # because the alternative to holding VRAM we said we would not hold
            # is losing the block's only copy.
            step.refused = set()
            for idx, block_id in step.evicting:
                try:
                    self.tier.store((req_id, idx), caches, block_id)
                except HostTierFull:
                    step.refused.add(idx)
                    self.evictions_refused += 1
                    continue
                self.copied_out += 1

            if not decoding:
                continue

            if self.scorer is not None:
                self._score(req_id, b, step, computed, block_size, caches)

            # 3. the view -- recorded here, applied at the metadata builder
            row = step.row
            resident = self._resident_now(step, computed, block_size, len(row))
            self._validate(req_id, row, resident, len(caches[0]))
            tail_count = computed % block_size + 1
            seq_len = (len(resident) - 1) * block_size + tail_count
            self._plan[b] = (row, resident, seq_len)
            intended[req_id] = len(resident)
            if self.config is not None and self.config.audit:
                self._last_view[req_id] = {
                    "row": list(row), "row_index": b,
                    "resident": {i for i in resident if i < computed // block_size},
                    "n_full": computed // block_size, "block_size": block_size,
                    "restored": [i for i, _ in step.restored],
                    "evicted": sorted(step.evicting and
                                      {i for i, _ in step.evicting} or set()),
                }

        # 4. and check what the kernel is about to be shown. The guard reads
        # the same view the builder will get, built here rather than from the
        # gathered table, which is deliberately left alone.
        if self.verify and self.guard is not None and intended:
            view_table = table.clone()
            view_seq = seq_lens.clone()
            for b, (row, resident, seq_len) in self._plan.items():
                for slot, idx in enumerate(resident):
                    view_table[b][slot] = row[idx]
                view_seq[b] = seq_len
            self.guard.check_step(batch, view_table, view_seq, slots,
                                  block_size, intended)

    def _resident_now(self, step: pager_state.RequestStep, computed: int,
                      block_size: int, row_len: int) -> list[int]:
        """The manager's choice of full blocks, with *this* step's tail.

        The two sides run on different clocks and must: the manager frees
        against the committed prefix (`total_computed_tokens -
        num_in_flight_tokens`), because an in-flight step is still reading
        blocks above it, while the view has to describe where this step's key
        is actually written. Taking the manager's set wholesale puts the tail
        one block behind whenever those differ, which is the `write_target`
        violation the guard reported.

        So: the policy's decision about full blocks is the manager's, since
        that is what it froze its freeing on, and everything from the
        manager's tail forward is kept unconditionally. Those indices are never
        freeable -- the manager only frees below its own tail -- so they are
        real, they hold real keys, and dropping them would make the newest
        tokens invisible rather than merely unpaged.
        """
        mgr_tail = step.num_computed // block_size
        tail = computed // block_size
        chosen = step.resident
        if self.show_pending and step.evicting:
            # Still allocated, still valid, and freed only on the next pass.
            # Hiding them wastes a step of context for nothing.
            chosen = sorted(set(chosen) | {i for i, _ in step.evicting})
        keep = [i for i in chosen if i < mgr_tail]
        keep += [i for i in range(mgr_tail, tail + 1) if i < row_len]
        return keep

    def _score(self, req_id, row_index, step, computed, block_size, caches):
        """Take bounds for what is here, and choose what should be here next.

        Bounds can only be taken while a block is resident, so this runs before
        anything is evicted; a block never observed can never be ranked, and
        would never be asked back. Blocks whose bounds are unknown are kept
        rather than dropped, which is the safe direction to be wrong in.
        """
        budget = self.budget or 0
        if not budget:
            return
        n_full = computed // block_size
        row = step.row
        # Everything still *allocated*, not everything still resident. Blocks
        # chosen for eviction this step are readable for exactly this step, and
        # they are the ones about to leave -- observing only the resident set
        # meant a block was evicted before its bounds were ever taken, so it
        # could never be ranked and never asked back. At the first paging step
        # that is most of the context.
        observable = set(step.resident) | {i for i, _ in step.evicting}
        for i in observable:
            if i < n_full and i < len(row) and row[i] != 0:
                self.scorer.observe(req_id, i, caches, row[i])

        queries = self.capture.for_row(row_index) if self.capture else []
        if not queries:
            self.no_queries += 1
            return
        ranked = self.scorer.rank(req_id, range(n_full), queries)
        if not ranked:
            return
        self.ranked += 1
        sink = self.config.sink if self.config else 0
        keep = list(range(min(sink, n_full)))
        unknown = [i for i in range(n_full)
                   if (req_id, i) not in self.scorer.bounds]
        self.unscored += len(unknown)
        keep += unknown                          # never drop what was not scored
        for index, _score in ranked:
            if len(set(keep)) >= budget:
                break
            keep.append(index)
        selection = sorted(set(keep))[:max(budget, len(unknown))]
        self.last_selection = selection
        # Only a query-aware *policy* may steer residency. The scorer also runs
        # under `audit`, which needs its bounds and queries -- and publishing
        # from there would make switching the measurement on change the thing
        # being measured, which it did: an audited `recency` run silently
        # became `quest` and the two reported identical numbers.
        if self.config is not None and self.config.policy == "quest":
            self.state.desired[req_id] = selection

    def _audit_previous(self, runner) -> None:
        """Judge the *previous* step's residency against its own queries.

        Deliberately retrospective. A step's decision has to be scored against
        the attention it was actually serving, and that query does not exist
        until the forward it belongs to has run.
        """
        for req_id, saved in list(self._last_view.items()):
            queries = self.capture.for_row(saved["row_index"])
            row = audit_step(
                runner.kv_caches, self.tier, req_id, saved["row"],
                saved["resident"], saved["n_full"], queries,
                saved["block_size"], self._spec.head_size,
                self._spec.head_size_v, saved["restored"], saved["evicted"])
            if row is not None:
                row["req"] = req_id
                self.audit_rows.append(row)
        self._last_view = {}

    def _release_finished(self) -> None:
        """Give back the host slots of requests the scheduler has let go.

        Nothing else does. A slot held past the end of its request is never
        reused, so a long-running server leaks one per evicted block per
        request until the tier is full -- at which point every eviction is
        refused, the budget silently stops being a budget, and the only
        symptom is that residency creeps upward. The kind of thing a
        four-request test cannot see.
        """
        if self.tier is None:
            return
        for req_id in self.state.take_finished():
            self.tier.release_request(req_id)
            if self.scorer is not None:
                self.scorer.forget(req_id)
            self.released += 1

    def _validate(self, req_id: str, row: list[int], resident: list[int],
                  num_blocks: int) -> None:
        """Fail where the mistake is, not where the GPU notices it.

        A bad block id written into the view is dereferenced by the attention
        kernel, so it surfaces as an asynchronous illegal access at whatever
        call happens to synchronise next -- which is somewhere else entirely.
        These are two comparisons per resident block against a Python list, and
        they turn that into an exception naming the request and the index.
        """
        bad = [i for i in resident if not 0 <= i < len(row)]
        if bad:
            raise IndexError(
                f"{req_id}: resident indices {bad[:4]} are outside a row of "
                f"{len(row)} blocks")
        ids = [row[i] for i in resident]
        bad_ids = [x for x in ids if not 0 <= x < num_blocks]
        if bad_ids:
            raise IndexError(
                f"{req_id}: block ids {bad_ids[:4]} are outside a pool of "
                f"{num_blocks} blocks")

    def _audit_summary(self) -> dict[str, Any] | None:
        if not self.audit_rows:
            return None
        n = len(self.audit_rows)
        pick = lambda k: sum(r[k] for r in self.audit_rows) / n   # noqa: E731
        return {
            "steps": n,
            "missed_mass": pick("missed"),
            "worst_layer_missed": max(r["worst_layer"] for r in self.audit_rows),
            "fetched_mass": pick("fetched_mass"),
            "evicted_mass": pick("evicted_mass"),
            "resident_blocks": pick("resident_blocks"),
            "total_blocks": pick("total_blocks"),
        }

    def summary(self) -> dict[str, Any]:
        out = {"steps": self.steps, "copied_in": self.copied_in,
               "copied_out": self.copied_out,
               "missing_host_copy": self.missing_host_copy,
               "evictions_refused": self.evictions_refused,
               "released": self.released,
               "ranked": self.ranked,
               "unscored": self.unscored,
               "no_queries": self.no_queries,
               "last_selection": list(self.last_selection),
               "audit": self._audit_summary(),
               "clock_mismatch": self.clock_mismatch}
        if self.tier is not None:
            out["tier"] = self.tier.stats()
        if self.guard is not None:
            out["guard"] = self.guard.summary()
        return out
