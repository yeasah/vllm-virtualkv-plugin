"""How a deployment turns this on, without editing code.

Kept apart from `integration.py` so that reading or validating a configuration
does not import vLLM: the tests exercise this without a GPU, and the entry
point checks it before deciding whether to touch anything.
"""

from __future__ import annotations

import os

from .policy import POLICIES


class Config:
    """How a deployment turns this on, without editing code.

    Environment rather than engine kwargs because a plugin loaded through an
    entry point has no argument to receive: it is constructed by vLLM before
    anything of ours runs.
    """

    PREFIX = "VLLM_VIRTUALKV_"

    def __init__(self, budget=0, sink=2, policy="recency", host_slots=None,
                 verify=True):
        self.budget = budget
        self.sink = sink
        self.policy = policy
        self.host_slots = host_slots
        self.verify = verify

    @classmethod
    def from_env(cls, env=None) -> "Config":
        env = os.environ if env is None else env

        def get(name, default, cast):
            raw = env.get(cls.PREFIX + name)
            return default if raw is None or raw == "" else cast(raw)

        cfg = cls(
            budget=get("BUDGET", 0, int),
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
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(
                f"unknown policy {self.policy!r}; have {sorted(POLICIES)}")
        if self.budget < 0 or self.sink < 0:
            raise ValueError("budget and sink must be non-negative")
        if self.budget and self.sink > self.budget:
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
        return (f"Config(budget={self.budget}, sink={self.sink}, "
                f"policy={self.policy!r}, host_slots={slots}, "
                f"verify={self.verify})")
