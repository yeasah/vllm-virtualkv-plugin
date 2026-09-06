#!/usr/bin/env python3
"""A real conversation, replayed: attention pressure without a benchmark.

`gsm8k_turns.py` grows a context but its turns are independent questions, so
the history is ballast and no policy is punished for dropping it. A real
session is the opposite: turn four says "make that shorter" and means the
thing in turn three. That is the pressure a residency policy has to survive,
and it needs no benchmark to produce -- a transcript that already exists has
it, and does not have to come from this model, or from any model.

**The transcript is fixed, so every arm sees the same bytes.** The assistant
turns come from the file, never from whichever arm is running, so the arms
cannot fork the way they would if each fed back its own generation. That is
`--replay` from `gsm8k_turns.py` with the bootstrap removed: no baseline arm
has to run first to make the history exist.

**What replaces accuracy.** There is no gold answer for "expand on that", and
inventing one would be writing a benchmark, which is the work this avoids. The
question a pager actually has to answer is not "is it right" but *"is it what
the model would have said"* -- so the reference is the `off` arm and the score
is fidelity to it:

- **agreement** -- of the baseline's tokens, how many the arm emitted before
  its first divergence, summed over turns. Graded, unlike a correct/incorrect
  column, and it does not need the task to have an answer.
- **drift** -- mean |delta logprob| on the tokens it *did* agree on. This is
  the sensitive one: an arm can match the baseline token for token while
  sitting much closer to flipping, and drift sees that where agreement cannot.
  It is what separates two policies that both happen to score 12/12.

Turns after the first divergence are compared anyway and reported, but their
prompts differ from the baseline's by then, so read them as accumulated damage
rather than as an independent trial.

    tools/session_turns.py --fetch session.json [--chain 3]
    tools/session_turns.py MODEL --transcript session.json --budget 25%
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsm8k_turns import ARMS, ENV, render  # noqa: E402

#: UltraChat's SFT test split: multi-turn by construction, English, ungated,
#: and long enough per turn that a budget binds. The rows API hands over a
#: few conversations without pulling the whole parquet.
SOURCE = ("HuggingFaceH4/ultrachat_200k", "default", "test_sft")
ROWS = "https://datasets-server.huggingface.co/rows?"


def fetch(args) -> int:
    """Chain a few real conversations into one session and write it out.

    Chaining is not a trick to pad the context: a long session *is* several
    topics one after another, and it makes the transcript discriminating in
    both directions -- a turn depends on its own conversation, so a policy
    that keeps too little of the recent past loses it, while everything before
    the current topic is genuinely droppable and a policy that cannot tell the
    difference wastes its budget on it.
    """
    ds, cfg, split = SOURCE
    picked, offset = [], args.offset
    while len(picked) < args.chain:
        q = urllib.parse.urlencode(
            dict(dataset=ds, config=cfg, split=split, offset=offset, length=20))
        with urllib.request.urlopen(ROWS + q, timeout=60) as r:
            batch = json.load(r)["rows"]
        if not batch:
            raise SystemExit("ran out of rows before filling the chain")
        offset += len(batch)
        for row in batch:
            msgs = row["row"]["messages"]
            chars = sum(len(m["content"]) for m in msgs)
            if len(msgs) >= args.min_turns * 2 and chars >= args.min_chars:
                picked.append(msgs)
                if len(picked) == args.chain:
                    break

    session = [m for conv in picked for m in conv]
    out = {"source": f"{ds}:{split}", "conversations": len(picked),
           "session": [{"role": m["role"], "content": m["content"]}
                       for m in session]}
    with open(args.fetch, "w") as f:
        json.dump(out, f, indent=1)
    users = sum(1 for m in session if m["role"] == "user")
    chars = sum(len(m["content"]) for m in session)
    print(f"{args.fetch}: {len(picked)} conversations, {users} user turns, "
          f"{chars} chars, from {ds}:{split}")
    return 0


class Forcer:
    """Make an arm decode the baseline's tokens instead of its own.

    Free-running, an arm's first wrong token ends the comparison: every step
    after it is conditioned on a different sequence, so the damage can only be
    reported as *where* it broke. Forcing keeps every arm on one trajectory,
    which makes each step's distribution directly comparable to the
    baseline's, and turns a stopping point into a per-step magnitude.

    It is not a substitute for the free-running number. Forcing measures the
    damage under the counterfactual that the arm never actually derails, so a
    policy that would spiral looks merely dented. Read the two together.

    The seam is `Sampler.sample`, chosen because the returned logprobs are
    computed upstream of it from the raw logits -- so substituting the token
    here does not touch the distribution being reported, and the rank comes
    back as the baseline token's true rank in this arm's own distribution,
    over the whole vocabulary rather than within the requested top-k.

    **There are two Sampler classes in the tree and only one is live.**
    `vllm/v1/sample/sampler.py` is the superseded one; the GPU runner imports
    `vllm/v1/worker/gpu/sample/sampler.py`. Patching the wrong one is silent:
    every arm free-runs, agrees with itself, and the table reports a perfect
    `flip 0.0000` for a policy that was never tested. That is why
    `force_selftest.py` forces a *corrupted* script -- a no-op forcer passes
    every other check.
    """

    def __init__(self) -> None:
        self.script: list[int] = []
        self.pos = 0
        self.natural: list[int] = []

    def load(self, ids: list[int]) -> None:
        self.script, self.pos, self.natural = ids, 0, []

    def install(self) -> str:
        import torch
        try:
            from vllm.v1.worker.gpu.sample.sampler import Sampler
        except ImportError:  # pragma: no cover - older trees
            from vllm.v1.sample.sampler import Sampler

        original = Sampler.sample

        def sample(inner, logits, *a, **k):
            sampled, processed = original(inner, logits, *a, **k)
            if self.pos < len(self.script) and sampled.numel() == 1:
                self.natural.append(int(sampled.reshape(-1)[0]))
                sampled = torch.full_like(sampled, self.script[self.pos])
                self.pos += 1
            return sampled, processed

        Sampler.sample = sample
        return f"{Sampler.__module__}.Sampler.sample"


def one_arm(args) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
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

    script = None
    if args.script:
        with open(args.script) as f:
            script = [t["ids"] for t in json.load(f)["turns"]]
    forcer = Forcer() if script is not None else None

    if args.audit:
        os.environ["VLLM_VIRTUALKV_AUDIT"] = "1"
    #: `max_num_batched_tokens` is raised for *every* arm under --force, not
    #: just the scripted ones: a prefill chunk that completes no output token
    #: still calls the sampler, which would silently consume a script entry
    #: and shift the rest of the turn. Applying it uniformly also keeps the
    #: baseline's logprobs taken under the same batching as the arms', so the
    #: per-step deltas are not measuring a change in chunking.
    extra = {}
    if args.force or args.script:
        extra["max_num_batched_tokens"] = max(args.max_len, 8192)
    llm = LLM(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.util, enforce_eager=True,
              enable_prefix_caching=True, max_num_seqs=1,
              trust_remote_code=True, **extra)
    tok = llm.get_tokenizer()
    if forcer is not None:
        print(f"[force] patched {forcer.install()}", flush=True)

    messages: list[dict] = []
    turns = []
    for i, msg in enumerate(session):
        if msg["role"] != "user" or i + 1 >= len(session):
            continue
        messages.append({"role": "user", "content": msg["content"]})
        prompt = render(tok, messages, args.thinking)
        budget = args.max_tokens
        if script is not None:
            if len(turns) >= len(script):
                break
            budget = len(script[len(turns)])
            forcer.load(script[len(turns)])
        if len(tok(prompt).input_ids) + budget > args.max_len:
            break
        params = SamplingParams(temperature=0.0, max_tokens=budget, logprobs=5)
        out = llm.generate([prompt], params)[0]
        got = out.outputs[0]
        ids = [int(x) for x in got.token_ids]
        #: The logprob and *rank* of the token actually emitted. Under
        #: forcing that token is the baseline's, so the rank is the
        #: baseline token's exact position in this arm's distribution --
        #: full-vocab, not truncated to the top-k, because vLLM ranks the
        #: sampled token by counting the whole row.
        #: Keep a slot per step even if the emitted token is somehow absent
        #: from the returned dict: these lists are compared against another
        #: arm's by index, and a silently shorter one would shift every
        #: later step against the wrong baseline step.
        sel = [(lp[i].logprob, lp[i].rank) if i in lp else (None, None)
               for i, lp in zip(ids, got.logprobs or [])]
        if script is not None and ids != script[len(turns)]:
            raise SystemExit(
                f"forcing did not take on turn {len(turns)}: emitted "
                f"{len(ids)} tokens, script has {len(script[len(turns)])}, "
                f"first mismatch at "
                f"{next((i for i, (x, y) in enumerate(zip(ids, script[len(turns)])) if x != y), 'end')}"
                " -- the sampler seam is wrong or a prefill chunk ate a step")
        turns.append({
            "text": got.text,
            "ids": ids,
            "lp": [{str(k): round(v.logprob, 6) for k, v in st.items()}
                   for st in (got.logprobs or [])],
            "sel_lp": [round(a, 6) if a is not None else None
                       for a, _ in sel],
            "sel_rank": [b for _, b in sel],
            "natural": forcer.natural if forcer else [],
            "prompt_chars": len(prompt),
        })
        # The transcript's own reply, not the model's: that is the whole
        # point. Every arm continues from identical bytes.
        messages.append({"role": "assistant",
                         "content": session[i + 1]["content"]})
        if len(turns) >= args.turns:
            break

    result = {"turns": turns}
    if made:
        s = made[0].summary()
        result["pager"] = {k: v for k, v in s.items()
                           if k not in ("tier", "guard")}
        result["guard"] = {"violations": s["guard"]["violations"],
                           "by_check": s["guard"]["by_check"],
                           "steps": s["guard"]["steps_checked"]}
    with open(args.out, "w") as f:
        json.dump(result, f)


def compare(ref: dict, arm: dict) -> tuple[int, int, list[float]]:
    """Agreeing prefix length, baseline length, and per-token logprob deltas.

    Only the agreeing prefix is scored for drift. Past the first divergence
    the two arms are decoding different sequences, so a logprob difference
    there is not a residency measurement.
    """
    a, b = ref["ids"], arm["ids"]
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    deltas = []
    for i in range(k):
        la = ref["lp"][i].get(str(a[i])) if i < len(ref["lp"]) else None
        lb = arm["lp"][i].get(str(a[i])) if i < len(arm["lp"]) else None
        if la is not None and lb is not None:
            deltas.append(abs(la - lb))
    return k, len(a), deltas


def forced_line(ref: list[dict], turns: list[dict]) -> str:
    """Per-step damage over every step, since forcing never truncates a turn.

    Three numbers, and the third is the one the other two cannot give:

    - **flip** -- steps whose top token is not the baseline's. Free-running,
      the first of these ends the turn; here they are all counted.
    - **dlp** -- mean drop in the baseline token's logprob. Magnitude where a
      flip count is presence/absence.
    - **rank** -- where the baseline's token actually landed in this arm's
      distribution. `mean` stays near 1 whenever the damage is rare;
      `p99`/`max` say how far the token fell when it did fall, which is the
      difference between "a block of importance was lost" and "how important
      was the block that was lost".
    """
    dlp: list[float] = []
    ranks: list[int] = []
    for a, b in zip(ref, turns):
        for ra, rb in zip(a.get("sel_lp", []), b.get("sel_lp", [])):
            if ra is not None and rb is not None:
                dlp.append(ra - rb)
        ranks += [r for r in b.get("sel_rank", []) if r is not None]
    if not ranks:
        return "no forced steps recorded"
    ranks_sorted = sorted(ranks)
    p99 = ranks_sorted[min(len(ranks) - 1, int(0.99 * len(ranks)))]
    flips = sum(1 for r in ranks if r and r > 1)
    mean_dlp = sum(dlp) / len(dlp) if dlp else 0.0
    return (f"flip {flips / len(ranks):.4f} ({flips}/{len(ranks)})  "
            f"dlp {mean_dlp:+.6f}  rank mean {sum(ranks) / len(ranks):.3f} "
            f"p99 {p99} max {max(ranks)}")


def report(arms: dict, args: argparse.Namespace) -> None:
    ref = arms["off"]["turns"]
    with open(args.transcript) as f:
        src = json.load(f)
    print(f"\n{args.model}  {len(ref)} turns from {src['source']} "
          f"({src['conversations']} conversations), budget {args.budget}")
    print(f"  context grows to {ref[-1]['prompt_chars']} chars, "
          f"{sum(len(t['ids']) for t in ref)} tokens generated\n")
    for name in ARMS:
        turns = arms[name]["turns"]
        if args.force:
            print(f"  {name:8s} {forced_line(ref, turns)}")
        else:
            agreed = total = 0
            deltas: list[float] = []
            clean = 0
            for a, b in zip(ref, turns):
                k, n, d = compare(a, b)
                agreed += k
                total += n
                deltas += d
                clean += (k == n == len(b["ids"]))
            drift = sum(deltas) / len(deltas) if deltas else 0.0
            print(f"  {name:8s} agreement {agreed / max(total, 1):.4f} "
                  f"({agreed}/{total})  drift {drift:.6f}  "
                  f"turns identical {clean}/{len(ref)}")
        p = arms[name].get("pager", {})
        g = arms[name].get("guard", {})
        au = p.get("audit")
        if au:
            print(f"           mass missed {au['missed_mass']:.4f}  worst layer "
                  f"{au['worst_layer_missed']:.3f}  fetched "
                  f"{au['fetched_mass']:.4f}  evicted {au['evicted_mass']:.4f}"
                  f"  {au['resident_blocks']:.0f}/{au['total_blocks']:.0f} blocks")
        if p:
            print(f"           out {p.get('copied_out', 0):5d} in "
                  f"{p.get('copied_in', 0):5d}  unbacked "
                  f"{p.get('missing_host_copy', 0)}  refused "
                  f"{p.get('evictions_refused', 0)}"
                  f"   guard {g.get('violations', 0)}/{g.get('steps', 0)}")
    if arms["churn"]["turns"] and all(
            a["ids"] == b["ids"] and a["lp"] == b["lp"]
            for a, b in zip(ref, arms["churn"]["turns"])):
        print("\n  churn is bit-identical on a real session")
    else:
        print("\n  CHURN DIVERGED -- the machinery lost something")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--transcript", default="session.json")
    ap.add_argument("--fetch", default="",
                    help="write a transcript here instead of running")
    ap.add_argument("--chain", type=int, default=3,
                    help="conversations to chain into one session")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--min-turns", type=int, default=4)
    ap.add_argument("--min-chars", type=int, default=6000)
    ap.add_argument("--turns", type=int, default=99)
    ap.add_argument("--budget", default="25%")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--util", type=float, default=0.60)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="decode the baseline's tokens in every arm and score "
                         "per step, instead of stopping at first divergence")
    ap.add_argument("--script", default="",
                    help="internal: the baseline run whose tokens to force")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.fetch:
        return fetch(args)
    if not args.model:
        ap.error("a model is required unless --fetch is given")
    if not args.max_tokens:
        args.max_tokens = 2048 if args.thinking else 512

    if args.arm:
        one_arm(args)
        return 0

    arms = {}
    with tempfile.TemporaryDirectory() as tmp:
        script = ""
        for name in ARMS:
            path = os.path.join(tmp, f"{name}.json")
            print(f"=== {name} ===", flush=True)
            env = {**os.environ, **ENV[name]}
            if name != "off":
                env["VLLM_VIRTUALKV_BUDGET"] = args.budget
            cmd = [sys.executable, "-u", __file__, args.model,
                   "--transcript", args.transcript,
                   "--turns", str(args.turns),
                   "--max-len", str(args.max_len),
                   "--max-tokens", str(args.max_tokens),
                   "--util", str(args.util),
                   *(["--audit"] if args.audit else []),
                   *(["--thinking"] if args.thinking else []),
                   *(["--script", script] if script else []),
                   "--arm", name, "--out", path]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode != 0:
                print(proc.stdout[-1500:], proc.stderr[-2500:])
                raise SystemExit(f"arm {name} failed")
            with open(path) as f:
                arms[name] = json.load(f)
            # The baseline supplies the script every other arm decodes, so
            # it has to be the first arm and its file has to outlive the loop.
            if name == "off" and args.force:
                script = os.path.join(tmp, "script.json")
                with open(script, "w") as f:
                    json.dump(arms["off"], f)
    report(arms, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
