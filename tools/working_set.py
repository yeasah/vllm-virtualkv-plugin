#!/usr/bin/env python3
"""How many blocks a decode step cannot do without, before any policy exists.

This is the measurement the design stands or falls on, and it needs no
mechanism that is not already built. Residency is worth managing only if a
step's attention can be reproduced from a fraction of the blocks; if most of
them carry mass, there is nothing for a policy to be clever about and the
stalls are unavoidable.

Run with everything resident (`full`, budget 0) so the step is the real one
and every block's true contribution is available to be compared against.

    tools/working_set.py MODEL --transcript session.json [--turns 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session_turns import render  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--transcript", default="session.json")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--util", type=float, default=0.75)
    args = ap.parse_args()

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    os.environ["VLLM_VIRTUALKV"] = "1"
    os.environ["VLLM_VIRTUALKV_POLICY"] = "full"
    os.environ["VLLM_VIRTUALKV_BUDGET"] = "0"
    os.environ["VLLM_VIRTUALKV_WORKING_SET"] = "1"

    from vllm import LLM, SamplingParams
    from vllm_virtualkv import WorkerPager

    made = []
    original = WorkerPager.__init__

    def tracked(self, *a, **k):
        original(self, *a, **k)
        made.append(self)

    WorkerPager.__init__ = tracked

    with open(args.transcript) as f:
        session = json.load(f)["session"]

    llm = LLM(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.util, enforce_eager=True,
              enable_prefix_caching=True, max_num_seqs=1,
              trust_remote_code=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    messages: list[dict] = []
    done = 0
    for i, msg in enumerate(session):
        if msg["role"] != "user" or i + 1 >= len(session):
            continue
        messages.append({"role": "user", "content": msg["content"]})
        prompt = render(tok, messages, False)
        if len(tok(prompt).input_ids) + args.max_tokens > args.max_len:
            break
        llm.generate([prompt], params)
        messages.append({"role": "assistant",
                         "content": session[i + 1]["content"]})
        done += 1
        print(f"  turn {done}: context {len(prompt)} chars", flush=True)
        if done >= args.turns:
            break

    ws = made[0].summary().get("working_set") if made else None
    if not ws:
        print("no working-set rows -- the measurement did not run")
        return 1

    n = ws["n_full"]
    print(f"\n{args.model}  {ws['steps']} decode steps, "
          f"{n:.0f} blocks of context on average; the bound overstates "
          f"true mass by {ws['slack']:.1f} orders of magnitude on average")
    print("\n  epsilon      bound          oracle         gap     worst step")
    for e in ("0.1", "0.01", "0.001", "0.0001"):
        b, o = ws.get(f"bound@{e}"), ws.get(f"oracle@{e}")
        if b is None:
            continue
        print(f"  {e:>8s}  {b:7.1f} ({b / n:5.1%})  {o:7.1f} ({o / n:5.1%})  "
              f"{b - o:6.1f}  {ws.get(f'max_bound@{e}', 0):7.0f}")
    sm = made[0].summary().get("summaries") if made else None
    if sm:
        print(f"\n  resident summaries, budget {sm['budget']:.0f} of "
              f"{sm['n_full']:.0f} blocks -- share of true attention mass "
              f"their pick holds")
        order = [("oracle", "oracle (ceiling)"), ("q8", "8-bit keys"),
                 ("q4", "4-bit keys"), ("q2", "2-bit keys"),
                 ("bound", "min/max bound"), ("layer0", "layer-0 queries"),
                 ("recency", "recency (floor)"),
                 ("oracle_stale", "oracle, 1 step stale"),
                 ("q2_stale", "2-bit keys, 1 step stale")]
        for k, label in order:
            if k in sm:
                print(f"    {label:20s} {sm[k]:.4f}")
        print(f"\n  under the oracle pick: median layer keeps "
              f"{sm.get('median_layer', 0):.4f}, worst layer keeps "
              f"{sm.get('worst_layer', 0):.4f} "
              f"(layer {sm.get('worst_layer_idx', 0):.0f})")
    print(f"\n  blocks a single layer demands at epsilon "
          f"{min((0.1, 0.01, 0.001, 0.0001)):g}: "
          f"min {ws.get('layer_alone_min', 0):.1f}, "
          f"mean {ws.get('layer_alone_mean', 0):.1f}, "
          f"max {ws.get('layer_alone_max', 0):.1f} of {n:.0f} "
          f"-- against {ws.get('oracle@0.0001', 0):.1f} for the union")

    bad = sum(ws.get(f"unsound@{e}", 0)
              for e in ("0.1", "0.01", "0.001", "0.0001"))
    print(f"\n  {'bound stayed sound at every step' if bad == 0 else f'BOUND UNSOUND on {bad} block-steps -- it is not an upper bound'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
