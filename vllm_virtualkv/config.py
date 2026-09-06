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

    def __init__(self, budget=0, sink=2, policy="recency", host_slots=1024,
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
            host_slots=get("HOST_SLOTS", 1024, int),
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
        if self.host_slots <= 0:
            raise ValueError("host_slots must be positive")

    @property
    def enabled(self) -> bool:
        """A budget of zero is the control arm: wired up, evicting nothing."""
        return True

    def __repr__(self) -> str:
        return (f"Config(budget={self.budget}, sink={self.sink}, "
                f"policy={self.policy!r}, host_slots={self.host_slots}, "
                f"verify={self.verify})")
