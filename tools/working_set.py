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
    ap.add_argument("--thinking", action="store_true",
                    help="let a thinking model think, which is where the "
                         "decode tokens actually are")
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
        prompt = render(tok, messages, args.thinking)
        if len(tok(prompt).input_ids) + args.max_tokens > args.max_len:
            break
        llm.generate([prompt], params)
        messages.append({"role": "assistant",
                         "content": session[i + 1]["content"]})
        done += 1
        print(f"  turn {done}: context {len(prompt)} chars", flush=True)
        if done >= args.turns:
            break

    if made:
        made[0].fired_or_raise()
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
                 ("sink_recency", "sinks + recency"),
                 ("recency", "recency, no sinks"),
                 ("oracle_stale", "oracle, 1 step stale"),
                 ("q2_stale", "2-bit keys, 1 step stale")]
        for k, label in order:
            if k in sm:
                print(f"    {label:20s} {sm[k]:.4f}")
        shares = (0.02, 0.05, 0.10, 0.25, 0.50)
        if f"oracle@{shares[0]:g}" in sm:
            print("\n  how the prize changes with the budget"
                  " -- mass captured\n")
            print("   budget   blocks   sinks+recency   2-bit   oracle   "
                  "gap to close")
            for sh in shares:
                sr = sm.get(f"sinkrec@{sh:g}", 0)
                q2 = sm.get(f"q2@{sh:g}", 0)
                orc = sm.get(f"oracle@{sh:g}", 0)
                print(f"   {sh:6.0%}   {sm.get(f'budget@{sh:g}', 0):6.0f}   "
                      f"{sr:13.4f}   {q2:.4f}   {orc:.4f}   "
                      f"{orc - sr:+.4f}")
        grains = [(k, v) for k, v in sorted(sm.items()) if k.startswith("grain@")]
        if grains:
            print("\n  granularity at a fixed token budget -- oracle "
                  "selection, only the block size changes\n")
            print("   block size   blocks kept   mass captured")
            for k, v in sorted(grains, key=lambda kv: int(kv[0].split("@")[1])):
                sz = int(k.split("@")[1])
                print(f"   {sz:10d}   {sm.get(f'grainblocks@{sz}', 0):11.0f}"
                      f"   {v:.4f}")
        print(f"\n  mass in the first two blocks alone: "
              f"{sm.get('sink_mass', 0):.4f}")
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

    ti = made[0].summary().get("tiers") if made else None
    if ti:
        print(f"\n  equal VRAM, budget {ti['budget']:.0f} exact blocks of "
              f"{ti['n_full']:.0f} -- relative L2 error of the attention output")
        print(f"    drop, {ti['budget']:.0f} exact + rest absent      "
              f"{ti['drop']:.4f}")
        for b in (8, 4, 2):
            k = f"degrade@{b}bit"
            if k in ti:
                print(f"    {b}-bit, {ti.get(f'exact@{b}bit', 0):4.0f} exact + "
                      f"{ti.get(f'degraded@{b}bit', 0):4.0f} degraded  "
                      f"{ti[k]:.4f}   (cost "
                      f"{ti.get(f'cost@{b}bit', 0):.1f})")

    bad = sum(ws.get(f"unsound@{e}", 0)
              for e in ("0.1", "0.01", "0.001", "0.0001"))
    print(f"\n  {'bound stayed sound at every step' if bad == 0 else f'BOUND UNSOUND on {bad} block-steps -- it is not an upper bound'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
