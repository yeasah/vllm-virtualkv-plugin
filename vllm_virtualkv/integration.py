"""Every place this reaches into vLLM, in one file on purpose.

Three of the five things this needs are documented extension points and are
registered normally. Two are patches, and they live here rather than scattered
through the code that uses them, so that "what does this plugin actually
monkeypatch" has a one-file answer -- for review, for the eventual move to its
own repo, and for noticing when upstream makes one of them unnecessary.

    registered      KVCacheSpecRegistry.register  spec and manager
    registered      register_backend(...)         the view, in a builder
    registered      vllm.general_plugins          startup

    patched         Attention.get_kv_cache_spec   choose the paged spec
    patched         GPUModelRunner.prepare_attn   sync the worker's row

**`Attention.get_kv_cache_spec`.** `customize_spec` looks like the intended
hook and is even called on the full-attention path -- but only to measure a
page size, after which the layer builds and returns a plain
`FullAttentionSpec` regardless. Its docstring calls itself "a temporary
compatibility API" and says the end state is for the backend to build the spec
directly (vllm#42449). So this patch has an upstream expiry date; check before
carrying it forward.

**`GPUModelRunner.prepare_attn`.** The worker's block table has to equal the
manager's logical mapping before `compute_slot_mappings` reads it positionally.
It does not, because restored blocks reach the worker through the *append*
channel while the manager places them at their index. There is no seam for
this: the scheduler-to-worker protocol can say "here are more blocks" and
cannot say "this block now lives at index i". Closing it properly means a field
on `SchedulerOutput`, which is the same protocol gap a query-aware policy will
need anyway.
"""

from __future__ import annotations

from dataclasses import fields

from .config import Config
from .manager import build_manager_class, make_spec_class, register


def required_host_slots(budget: int, vllm_config) -> int:
    """Blocks the tier must hold for a budget to mean what it says.

    A budget is a promise that everything not resident is somewhere else, so
    the tier has to take the difference for every request that can be in
    flight at once. All three inputs are things the engine already knows and
    the operator would otherwise be looking up to do this arithmetic by hand.
    """
    if not budget:
        return 0                          # evicting nothing needs no tier
    block_size = vllm_config.cache_config.block_size
    max_len = vllm_config.model_config.max_model_len
    concurrency = max(1, vllm_config.scheduler_config.max_num_seqs)
    per_request = max(0, -(-max_len // block_size) - budget)
    return concurrency * per_request


def check_host_tier_size(config: Config, vllm_config) -> tuple[int, str | None]:
    """Is the host tier big enough for what the budget implies?

    Relaxing a residency budget is a promise: everything not resident is
    somewhere else. The tier has to be able to hold the difference, for every
    request that can be in flight at once --

        slots >= max_num_seqs * (ceil(max_model_len / block_size) - budget)

    -- or a request will reach a point where it cannot evict any further. That
    is survivable now (the worker refuses the eviction and the block stays on
    the GPU) but it means the budget silently stops being the budget, and
    once the startup context guard is relaxed it stops being survivable at all:
    the memory the relaxation was counting on will not be there.

    Returns the requirement, and a message only when an *explicitly set* tier
    is too small -- the default derives from this same number, so the message
    is for someone who overrode it. Under-provisioning degrades rather than
    corrupts today, and a plugin that refused to start over a heuristic it
    computed itself would be worse than one that says what it needs.
    """
    needed = required_host_slots(config.budget, vllm_config)
    if config.host_slots is None or config.host_slots >= needed:
        return needed, None
    max_len = vllm_config.model_config.max_model_len
    concurrency = max(1, vllm_config.scheduler_config.max_num_seqs)
    return needed, (
        f"VLLM_VIRTUALKV_HOST_SLOTS was set to {config.host_slots}, but a "
        f"budget of {config.budget} over {max_len} tokens at {concurrency} "
        f"concurrent requests can displace {needed} blocks. Evictions past "
        f"that point are refused, so the resident set will exceed the budget. "
        f"Unset it to size the tier automatically, or set it to {needed}."
    )


def patch_spec(config: Config):
    """Make full-attention layers ask for the paged spec. Returns the original.

    Sliding-window and other non-full specs are passed through untouched: their
    kernels rebuild key position from the block's index in the row, which
    permuting or compacting breaks -- measured, not assumed.
    """
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    paged_cls = make_spec_class()
    register(paged_cls, build_manager_class())
    original = Attention.get_kv_cache_spec

    warned = []

    def hooked(self, vllm_config):
        spec = original(self, vllm_config)
        if type(spec) is not FullAttentionSpec:
            return spec
        if not warned:
            warned.append(True)
            _, message = check_host_tier_size(config, vllm_config)
            if message:
                from vllm.logger import init_logger

                init_logger(__name__).warning("virtualkv: %s", message)
        common = {f.name: getattr(spec, f.name) for f in fields(spec)}
        return paged_cls(**common, budget_blocks=config.budget,
                         sink_blocks=config.sink, policy_name=config.policy)

    hooked._kvpager_original = original
    Attention.get_kv_cache_spec = hooked
    return original


def unpatch_spec(original) -> None:
    from vllm.model_executor.layers.attention import Attention

    Attention.get_kv_cache_spec = original


def enable(config: Config | None = None, scheduler=None):
    """Register everything, install both halves, and return what was built.

    Both halves: patching the spec alone gives a manager that evicts with
    nothing imposing a view and nothing moving bytes, which is strictly worse
    than not being installed. They are enabled together or not at all.

    The guard defaults to its worker-local checks, because that is the
    configuration a deployment runs in -- the ownership check needs the
    scheduler's allocation table and therefore an in-process engine. Callers
    that have a scheduler (tests, the tools in this repo) pass it and get the
    fourth check.
    """
    from .worker import WorkerPager

    config = config or Config.from_env()
    original = patch_spec(config)
    pager = WorkerPager(host_slots=config.host_slots, scheduler=scheduler,
                        verify=config.verify, budget=config.budget,
                        show_pending=config.show_pending)
    pager.install()
    return config, pager, original


__all__ = ["Config", "check_host_tier_size", "enable", "patch_spec",
           "required_host_slots", "unpatch_spec"]
