"""How a deployment turns this on, without editing code.

Kept apart from `integration.py` so that reading or validating a configuration
does not import vLLM: the tests exercise this without a GPU, and the entry
point checks it before deciding whether to touch anything.
"""

from __future__ import annotations

import os

from .policy import POLICIES


def parse_budget(raw: str | int) -> tuple[str, float]:
    """Read a residency budget written in whichever unit made sense.

        64      64 blocks -- an engine detail the operator did not choose
        1024t   1024 tokens
        25%     a quarter of max_model_len

    Blocks are the unit the code wants and the worst one to ask for: block size
    is not the operator's decision, so a budget in blocks cannot be compared
    across models or engines, and a sweep written in it is not a sweep of the
    same thing. Tokens and fractions resolve to blocks once the engine is up
    and both numbers are known.

    Returns (kind, value) so the format can be rejected at startup rather than
    at the first request.
    """
    if isinstance(raw, int):
        return ("blocks", float(raw))
    text = str(raw).strip().lower()
    if not text:
        return ("blocks", 0.0)
    try:
        if text.endswith("%"):
            value = float(text[:-1])
            if not 0 <= value <= 100:
                raise ValueError
            return ("fraction", value / 100.0)
        if text.endswith("t"):
            return ("tokens", float(text[:-1]))
        if text.endswith("b"):
            return ("blocks", float(text[:-1]))
        return ("blocks", float(text))
    except ValueError:
        raise ValueError(
            f"cannot read a budget from {raw!r}; use blocks (64), tokens "
            f"(1024t) or a share of max_model_len (25%)"
        ) from None


def resolve_budget(parsed: tuple[str, float], vllm_config) -> int:
    """Turn a parsed budget into whole blocks, now that the engine is known."""
    kind, value = parsed
    block_size = vllm_config.cache_config.block_size
    if kind == "tokens":
        value = value / block_size
    elif kind == "fraction":
        value = value * vllm_config.model_config.max_model_len / block_size
    return int(value) if value <= 0 else max(1, int(value))


class Config:
    """How a deployment turns this on, without editing code.

    Environment rather than engine kwargs because a plugin loaded through an
    entry point has no argument to receive: it is constructed by vLLM before
    anything of ours runs.
    """

    PREFIX = "VLLM_VIRTUALKV_"

    def __init__(self, budget=0, sink=2, policy="recency", host_slots=None,
                 verify=True, show_pending=False, head_agg="max",
                 layer_agg="max"):
        #: (kind, value) until an engine exists; `budget_blocks` after.
        self.budget_spec = parse_budget(budget)
        #: Whole blocks. Only meaningful once `resolve` has run, except when
        #: the budget was given in blocks to begin with.
        self.budget = int(self.budget_spec[1]) if self.budget_spec[0] == "blocks" else 0
        self.sink = sink
        self.policy = policy
        self.host_slots = host_slots
        self.verify = verify
        #: Show the kernel blocks that are chosen for eviction but not yet
        #: freed. They are still allocated and still valid for one more step,
        #: so reading them is free quality -- but it makes the effective
        #: resident set the budget *plus* whatever is in flight, which muddies
        #: a measurement of what a budget buys. Off by default so a quality
        #: number means what it says; on for the churn policy, which depends
        #: on it to keep the context whole.
        self.show_pending = show_pending
        #: How a block's per-head and per-layer bounds collapse into the one
        #: number residency is decided on. `head_agg="max"` is not a default
        #: taken on principle: with `mean` the scored policy failed to select
        #: the block holding a planted answer and with `max` it selected it,
        #: at the same budget. Retrieval signal sits in a few query heads and
        #: averaging over the group dilutes it -- which is the opposite of the
        #: earlier finding that mean captures more *mass* across a GQA group,
        #: a different question that this default was wrongly imported from.
        self.head_agg = head_agg
        self.layer_agg = layer_agg

    @classmethod
    def from_env(cls, env=None) -> "Config":
        env = os.environ if env is None else env

        def get(name, default, cast):
            raw = env.get(cls.PREFIX + name)
            return default if raw is None or raw == "" else cast(raw)

        cfg = cls(
            budget=get("BUDGET", 0, str),
            sink=get("SINK", 2, int),
            policy=get("POLICY", "recency", str),
            # None means derive it. A knob that is checked against a hard
            # threshold should default to a value computed from that threshold,
            # not to a number the operator has to look up: the engine knows
            # max_model_len, the block size and the concurrency, and the
            # operator would only be arithmetic-ing the same three.
            host_slots=get("HOST_SLOTS", None,
                           lambda x: None if x == "auto" else int(x)),
            verify=get("VERIFY", True, lambda x: x not in ("0", "false", "no")),
            show_pending=get("SHOW_PENDING", False,
                             lambda x: x not in ("0", "false", "no")),
            head_agg=get("HEAD_AGG", "max", str),
            layer_agg=get("LAYER_AGG", "max", str),
        )
        cfg.validate()
        return cfg

    def resolve(self, vllm_config) -> int:
        """Fix the budget in blocks, once the engine can say how big one is."""
        self.budget = resolve_budget(self.budget_spec, vllm_config)
        if self.budget and self.sink > self.budget:
            raise ValueError(
                f"sink ({self.sink}) exceeds the resolved budget "
                f"({self.budget} blocks from {self.describe_budget()})")
        return self.budget

    def describe_budget(self) -> str:
        kind, value = self.budget_spec
        if kind == "tokens":
            return f"{value:g} tokens"
        if kind == "fraction":
            return f"{value * 100:g}% of max_model_len"
        return f"{value:g} blocks"

    def validate(self) -> None:
        for name, value in (("head_agg", self.head_agg),
                            ("layer_agg", self.layer_agg)):
            if value not in ("max", "mean"):
                raise ValueError(f"{name} must be 'max' or 'mean', not {value!r}")
        if self.policy == "churn" and not self.show_pending:
            raise ValueError(
                "policy 'churn' needs show_pending: it depends on reading a "
                "block during the step it was chosen for eviction, and without "
                "that it drops context instead of cycling it")
        if self.policy not in POLICIES:
            raise ValueError(
                f"unknown policy {self.policy!r}; have {sorted(POLICIES)}")
        if self.budget_spec[1] < 0 or self.sink < 0:
            raise ValueError("budget and sink must be non-negative")
        if self.budget_spec[0] == "blocks" and self.budget and \
                self.sink > self.budget:
            raise ValueError(
                f"sink ({self.sink}) cannot exceed budget ({self.budget})")
        if self.host_slots is not None and self.host_slots <= 0:
            raise ValueError("host_slots must be positive, or unset for auto")

    @property
    def enabled(self) -> bool:
        """A budget of zero is the control arm: wired up, evicting nothing."""
        return True

    def __repr__(self) -> str:
        slots = "auto" if self.host_slots is None else self.host_slots
        return (f"Config(budget={self.describe_budget()}, sink={self.sink}, "
                f"policy={self.policy!r}, host_slots={slots}, "
                f"verify={self.verify}, show_pending={self.show_pending})")
