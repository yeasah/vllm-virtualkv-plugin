"""Virtual memory for a vLLM KV cache.

vLLM's PagedAttention virtualised the *allocation* of KV blocks. This
virtualises their *residency*: a request's context may be larger than the GPU
can hold, with the remainder kept in host memory and brought back when it is
wanted. The block table is a page table, the per-step view is address
translation, and the host tier is swap.

The property that makes it worth doing is what it costs to be wrong. Eviction
discards, so a mistake is unrecoverable -- a needle at an evicted position is
gone. Nothing here is discarded, so a mistaken residency decision is a *stall*,
not a wrong answer, which is what lets a policy be aggressive.

Turned on by environment, since a plugin loaded through an entry point is
constructed before any of its own code runs:

    VLLM_VIRTUALKV=1                 enable at all
    VLLM_VIRTUALKV_BUDGET=64         resident full blocks per request
                                     (0 = evict nothing: the control arm,
                                      which must be bit-identical to running
                                      without the plugin)
    VLLM_VIRTUALKV_POLICY=recency    recency | stress | full | oracle
    VLLM_VIRTUALKV_SINK=2            leading blocks always kept
    VLLM_VIRTUALKV_HOST_SLOTS=1024   host tier size, in blocks
    VLLM_VIRTUALKV_VERIFY=1          run the residency guard

What is measured, and what is not, is in README.md. The short version: the
mechanism reproduces a full-context answer token-for-token at 12% residency
when the right blocks are kept, and the shipped policy is not the one that
keeps them.
"""

from .config import Config  # noqa: F401
from .guard import ResidencyGuard, Violation  # noqa: F401
from .hosttier import HostTier, HostTierFull  # noqa: F401
from .policy import POLICIES, Full, Oracle, Recency, Stress  # noqa: F401
from .state import PagerState, current, reset  # noqa: F401
from .worker import WorkerPager  # noqa: F401

__version__ = "0.0.1"


def register() -> None:
    """`vllm.general_plugins` entry point. Runs in every engine process.

    Deliberately quiet when unconfigured: this is imported by every vLLM
    process on the machine once installed, and a plugin that patches attention
    because it happens to be on the path would be a menace. `VLLM_VIRTUALKV`
    must be set explicitly, and a budget of zero still wires everything up
    while evicting nothing -- that combination is the control arm, not a
    no-op, and it is the one configuration whose output must be identical to
    not having the plugin at all.
    """
    import os

    if os.environ.get("VLLM_VIRTUALKV", "") in ("", "0", "false", "no"):
        return

    from vllm.logger import init_logger

    from .integration import enable

    config, _pager, _original = enable()
    init_logger(__name__).info("virtualkv enabled: %s", config)
