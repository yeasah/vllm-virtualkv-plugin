#!/usr/bin/env python3
"""Show that teacher forcing bites, and that it does not disturb what it reads.

A forcer that silently did nothing would produce a beautiful table: every arm
would free-run, agree with itself, and report flip 0.0000. So this asserts the
two halves separately.

- **Null.** Force the baseline's own tokens back into the baseline. Nothing may
  change: same ids, same logprobs to the bit, every rank 1. If forcing
  perturbed the distribution it is reading, this is where it shows.
- **Bite.** Force a *corrupted* script -- one token replaced mid-turn. The
  output must follow the corruption rather than the model's preference, the
  model's own choice at that step must differ from what was forced, and some
  rank must exceed 1. A no-op forcer fails all three.

    tools/force_selftest.py MODEL --transcript session.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "session_turns.py")


def run(model, transcript, out, script="", turns=2, extra=()):
    env = {**os.environ, "VLLM_VIRTUALKV": "0"}
    cmd = [sys.executable, "-u", TOOL, model, "--transcript", transcript,
           "--turns", str(turns), "--arm", "off", "--out", out, *extra]
    if script:
        cmd += ["--script", script]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        # A forced arm that could not follow its script kills itself on
        # purpose. That is a result here, not a reason to stop.
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        print("  arm exited nonzero:", tail[-1] if tail else "(no output)")
        return None
    with open(out) as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--transcript", default="session.json")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--util", type=float, default=0.60)
    args = ap.parse_args()
    #: --force on every invocation, including the unscripted one: it also
    #: selects the batching, and a baseline taken under different batching
    #: from the forced runs would make the null comparison meaningless.
    extra = ["--max-tokens", str(args.max_tokens),
             "--max-len", str(args.max_len), "--util", str(args.util),
             "--force"]

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        j = lambda n: os.path.join(tmp, n)  # noqa: E731

        free = run(args.model, args.transcript, j("free.json"),
                   turns=args.turns, extra=extra)
        if free is None:
            raise SystemExit("the unforced baseline arm failed; nothing to test")
        with open(j("script.json"), "w") as f:
            json.dump(free, f)

        # --- null -----------------------------------------------------------
        null = run(args.model, args.transcript, j("null.json"),
                   script=j("script.json"), turns=args.turns, extra=extra)
        print("null (force the baseline into itself)")
        if null is None:
            print("  FAIL  the forced arm could not reproduce the baseline")
            return 1
        same_ids = all(a["ids"] == b["ids"]
                       for a, b in zip(free["turns"], null["turns"]))
        same_lp = all(a["lp"] == b["lp"]
                      for a, b in zip(free["turns"], null["turns"]))
        ranks = [r for t in null["turns"] for r in t["sel_rank"]
                 if r is not None]
        all_rank1 = all(r == 1 for r in ranks)
        nat_matches = all(t["natural"][:len(t["ids"])] == t["ids"]
                          for t in null["turns"] if t["natural"])
        for label, got in [("ids unchanged", same_ids),
                           ("logprobs unchanged", same_lp),
                           ("every rank 1", all_rank1),
                           ("model's own choice == forced", nat_matches)]:
            print(f"  {'ok ' if got else 'FAIL'}  {label}")
            ok &= got

        # --- bite -----------------------------------------------------------
        corrupt = json.loads(json.dumps(free))
        first = corrupt["turns"][0]["ids"]
        at = min(5, len(first) - 1)
        was = first[at]
        first[at] = 1 if was != 1 else 2
        with open(j("bad.json"), "w") as f:
            json.dump(corrupt, f)
        bit = run(args.model, args.transcript, j("bit.json"),
                  script=j("bad.json"), turns=args.turns, extra=extra)
        print(f"\nbite (token {at} of turn 0: {was} -> {first[at]})")
        if bit is None:
            print("  FAIL  the arm refused the corrupted script -- forcing "
                  "is not reaching the sampler")
            print("\nFORCING IS NOT DOING WHAT IT CLAIMS")
            return 1

        got_ids = bit["turns"][0]["ids"]
        followed = got_ids[:at + 1] == first[:at + 1]
        natural = bit["turns"][0]["natural"]
        disagreed = len(natural) > at and natural[at] != first[at]
        ranks = [r for t in bit["turns"] for r in t["sel_rank"]
                 if r is not None]
        fell = any(r > 1 for r in ranks)
        for label, got in [("output followed the corruption", followed),
                           ("model would have chosen otherwise", disagreed),
                           ("some rank > 1", fell)]:
            print(f"  {'ok ' if got else 'FAIL'}  {label}")
            ok &= got
        if ranks:
            print(f"  max rank {max(ranks)}, "
                  f"{sum(1 for r in ranks if r > 1)}/{len(ranks)} steps > 1")

    print("\n" + ("forcing works and is non-perturbing"
                  if ok else "FORCING IS NOT DOING WHAT IT CLAIMS"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
