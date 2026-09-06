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

    def hooked(self, vllm_config):
        spec = original(self, vllm_config)
        if type(spec) is not FullAttentionSpec:
            return spec
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
                        verify=config.verify)
    pager.install()
    return config, pager, original


__all__ = ["Config", "enable", "patch_spec", "unpatch_spec"]
