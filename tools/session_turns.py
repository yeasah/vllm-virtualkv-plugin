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
from gsm8k_turns import ENV as _GSM_ENV, render  # noqa: E402

#: Its own arms, not gsm8k's. `massoracle` is the ceiling this harness
#: exists to test -- residency ranked on measured attention mass -- and
#: `churn` is dropped because its bit-exactness is already established and
#: it costs a full copy-out/copy-in of the context every step.
ARMS = ("off", "truncate", "recency", "quest", "massoracle")
ENV = {**_GSM_ENV,
       "truncate": {"VLLM_VIRTUALKV": "0"},
       "massoracle": {"VLLM_VIRTUALKV": "1",
                      "VLLM_VIRTUALKV_POLICY": "massoracle"}}

#: `truncate` is the control the other arms were missing. Recency is very
#: nearly "use a shorter context", and a comparison in which every arm is
#: scored only against the *full* baseline cannot tell a policy that retained
#: distant context from one whose damage merely happened to be in
#: distribution -- a contiguous recent window is a shorter conversation, which
#: the model has seen a trillion of, while a mass-ranked set is a context full
#: of holes at positions that correspond to nothing it was trained on.
#:
#: So this arm runs no plugin at all and simply cuts the prompt to sinks plus
#: the most recent tokens, which is genuinely the shorter context rather than
#: an approximation of it. It is the floor the whole idea has to clear: a
#: policy that cannot beat throwing the context away has not shown that
#: keeping it was worth anything.

#: UltraChat's SFT test split: multi-turn by construction, English, ungated,
#: and long enough per turn that a budget binds. The rows API hands over a
#: few conversations without pulling the whole parquet.
SOURCE = ("HuggingFaceH4/ultrachat_200k", "default", "test_sft")
ROWS = "https://datasets-server.huggingface.co/rows?"


def from_trajectory(path: str, keep_reasoning: bool = True) -> dict:
    """A mini-swe-agent trace as an alternating transcript.

    Why an agent trace rather than a chat: the dependency is structural
    rather than incidental. The task statement sits in the first message and
    every later turn is still deciding what to do about it, so a recency
    window that has dropped it is deciding blind -- a needle that is load
    bearing on every turn, organic rather than planted. Tool outputs are
    large enough to push it far back, and file contents read early are
    referenced hundreds of turns later.

    The trace already alternates: an agent turn, then the output of what it
    ran. Tool results become the `user` side, so no structure is invented.
    Reasoning, prose and the command are flattened into one assistant message
    because the point is the token sequence and its recall demands, not
    faithful tool-call protocol -- and a `tool` role would need the model's
    template to agree with this trace's, which is a different model's.
    """
    with open(path) as f:
        traj = json.load(f)
    msgs = traj["messages"]

    def render(x, keep_reasoning: bool = True) -> str:
        parts = []
        if keep_reasoning and x.get("reasoning_content"):
            parts.append(str(x["reasoning_content"]))
        if x.get("content"):
            parts.append(str(x["content"]))
        for tc in x.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                cmd = json.loads(fn.get("arguments") or "{}").get("command")
            except json.JSONDecodeError:
                cmd = None
            parts.append(f"$ {cmd}" if cmd else str(fn))
        return "\n".join(p for p in parts if p)

    session, pending = [], []
    for x in msgs:
        role = x.get("role")
        if role in ("system", "user"):
            pending.append(render(x))
        elif role == "tool":
            pending.append(render(x))
        elif role == "assistant":
            text = render(x, keep_reasoning)
            if not text:
                continue
            # An assistant turn needs something before it to answer.
            session.append({"role": "user",
                            "content": "\n\n".join(pending) or "(continue)"})
            session.append({"role": "assistant", "content": text})
            pending = []
    return {"source": f"swe-agent:{traj.get('instance_id', path)}",
            "conversations": 1, "session": session}


def fetch(args) -> int:
    """Chain a few real conversations into one session and write it out.

    Chaining is not a trick to pad the context: a long session *is* several
    topics one after another, and it makes the transcript discriminating in
    both directions -- a turn depends on its own conversation, so a policy
    that keeps too little of the recent past loses it, while everything before
    the current topic is genuinely droppable and a policy that cannot tell the
    difference wastes its budget on it.
    """
    if args.traj:
        # Qwen3's template drops prior <think> blocks and keeps only each
        # earlier turn's final answer -- verified, not assumed. Reasoning is
        # 93% of assistant text in these traces and is where every turn
        # restates the task, so keeping it builds a transcript far more
        # redundant than any deployment, and that redundancy is exactly what
        # lets a recency window survive without the original task statement.
        out = from_trajectory(args.traj, keep_reasoning=not args.strip_reasoning)
        with open(args.fetch, "w") as f:
            json.dump(out, f, indent=1)
        turns = sum(1 for m in out["session"] if m["role"] == "user")
        chars = sum(len(m["content"]) for m in out["session"])
        print(f"{args.fetch}: {turns} agent turns, {chars} chars, "
              f"from {out['source']}")
        return 0
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


class Tap:
    """Force the baseline's tokens, and read the full distribution while there.

    Forcing exists because free-running truncates: an arm's first wrong token
    ends the comparison, so damage can only be reported as *where* it broke.
    Keeping every arm on one trajectory makes each step's distribution
    comparable to the baseline's, and turns a stopping point into a per-step
    magnitude. It is not a substitute for the free-running number -- it
    measures damage under the counterfactual that the arm never derails, so a
    policy that would spiral reads as merely dented.

    **Why a flip count and a mean logprob delta are not enough.** Both average
    over positions whose difficulty differs by orders of magnitude. Where the
    model is near-certain, a small perturbation is a destroyed fact; where it
    is choosing among fifteen synonyms, the same perturbation is nothing, and
    a flip there says more about English than about residency. Worse, *which*
    token diverges first is largely decided by where the flat distributions
    happen to fall, so two arms that diverge at different points were scored
    at structurally different positions. So the tap records, from the full
    logits it already has:

    - **entropy** of the baseline's distribution at each step, which is what
      the damage has to be read against. The steps that carry information out
      of an evicted block are the low-entropy ones.
    - **top-K of the baseline**, so an arm can compute a real divergence
      against it rather than a delta on one token. `KL(P_ref || P_arm)` is
      weighted by the baseline's own probabilities, so it does not care which
      token was sampled, and diffuse positions cannot dominate it the way a
      raw logprob delta lets them.

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

    K = 32
    NUCLEUS = 0.9

    def __init__(self) -> None:
        self.script: list[int] = []
        self.ref: list[tuple[list[int], list[float]]] = []
        self.step = 0
        self.natural: list[int] = []
        self.ent: list[float] = []
        self.nuc: list[int] = []
        self.kl: list[float] = []
        self.cov: list[float] = []
        self.topk: list[tuple[list[int], list[float]]] = []

    def load(self, ids: list[int] | None,
             ref: list[tuple[list[int], list[float]]] | None = None) -> None:
        self.script = ids or []
        self.ref = ref or []
        self.step = 0
        self.natural, self.ent, self.nuc = [], [], []
        self.kl, self.cov, self.topk = [], [], []

    def _observe(self, logits) -> None:
        import torch

        logp = torch.log_softmax(logits[0].float(), dim=-1)
        p = logp.exp()
        self.ent.append(float(-(p * logp).sum()))
        srt = torch.sort(p, descending=True).values
        cum = torch.cumsum(srt, 0)
        self.nuc.append(int(torch.searchsorted(cum, self.NUCLEUS)) + 1)
        tl, ti = torch.topk(logp, self.K)
        self.topk.append(([int(x) for x in ti], [round(float(x), 6)
                                                 for x in tl]))
        if self.step < len(self.ref):
            ids, lps = self.ref[self.step]
            rid = torch.as_tensor(ids, device=logp.device)
            rlp = torch.as_tensor(lps, device=logp.device, dtype=torch.float32)
            rp = rlp.exp()
            # KL(P_ref || P_arm) over the baseline's own top-K. The weight is
            # p_ref, so the omitted tail is small by construction; `cov`
            # reports how much of it was actually covered.
            self.kl.append(float((rp * (rlp - logp[rid])).sum()))
            self.cov.append(float(rp.sum()))

    def install(self) -> str:
        import torch
        try:
            from vllm.v1.worker.gpu.sample.sampler import Sampler
        except ImportError:  # pragma: no cover - older trees
            from vllm.v1.sample.sampler import Sampler

        original = Sampler.sample

        def sample(inner, logits, *a, **k):
            sampled, processed = original(inner, logits, *a, **k)
            if sampled.numel() == 1 and logits.shape[0] == 1:
                self._observe(logits)
                if self.step < len(self.script):
                    self.natural.append(int(sampled.reshape(-1)[0]))
                    sampled = torch.full_like(sampled, self.script[self.step])
                self.step += 1
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

    script = ref_topk = None
    if args.script:
        with open(args.script) as f:
            base = json.load(f)["turns"]
        script = [t["ids"] for t in base]
        ref_topk = [[(i, l) for i, l in t.get("topk", [])] for t in base]
    tap = Tap() if (args.force or args.script) else None

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
    if tap is not None:
        print(f"[tap] patched {tap.install()}", flush=True)

    messages: list[dict] = []
    turns = []
    for i, msg in enumerate(session):
        if msg["role"] != "user" or i + 1 >= len(session):
            continue
        messages.append({"role": "user", "content": msg["content"]})
        prompt = render(tok, messages, args.thinking)
        cut = None
        if args.truncate_tokens:
            ids = tok(prompt).input_ids
            if len(ids) > args.truncate_tokens:
                #: Sinks plus the recent window, which is what `recency`
                #: keeps -- so the two differ in whether the surviving
                #: tokens keep their original positions, not in which
                #: tokens survive.
                keep = args.truncate_tokens - args.truncate_sinks
                cut = list(ids[:args.truncate_sinks]) + list(ids[-keep:])
        budget = args.max_tokens
        if tap is not None:
            tap.load(None)
        if script is not None:
            if len(turns) >= len(script):
                break
            budget = len(script[len(turns)])
            tap.load(script[len(turns)], ref_topk[len(turns)])
        if len(tok(prompt).input_ids) + budget > args.max_len:
            break
        params = SamplingParams(temperature=0.0, max_tokens=budget, logprobs=5)
        out = llm.generate(
            [{"prompt_token_ids": cut} if cut else prompt], params)[0]
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
            "natural": tap.natural if tap else [],
            "ent": [round(x, 6) for x in tap.ent] if tap else [],
            "nuc": tap.nuc if tap else [],
            "kl": [round(x, 6) for x in tap.kl] if tap else [],
            "cov": [round(x, 6) for x in tap.cov] if tap else [],
            "topk": tap.topk if tap else [],
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


#: Read against the baseline's own uncertainty. Below ~0.5 nats the model is
#: effectively committed, so a flip is a destroyed fact rather than a change
#: of wording; above ~2 nats it was choosing between many acceptable
#: continuations and a flip says more about English than about residency.
BANDS = ((0.0, 0.5, "certain"), (0.5, 2.0, "mixed"), (2.0, 1e9, "diffuse"))


def forced_stats(ref: list[dict], turns: list[dict]) -> dict:
    """Per-step damage over every step, stratified by baseline entropy.

    Forcing never truncates a turn, so every step is scored. The headline is
    KL against the baseline's distribution rather than a delta on one token:
    it is weighted by the baseline's own probabilities, so it does not depend
    on which token happened to be sampled, and a diffuse position cannot
    dominate it the way a raw logprob delta lets it.

    The bands are the part that answers "how important was what was lost".
    A flip rate averaged over all positions is close to uninterpretable --
    it is dominated by wherever the flat distributions fell. The same rate
    restricted to positions where the baseline was near-certain is not.
    """
    rows: list[tuple[float, float, int, float]] = []
    for a, b in zip(ref, turns):
        ent = a.get("ent", [])
        kl = b.get("kl", [])
        rank = [r for r in b.get("sel_rank", [])]
        la, lb = a.get("sel_lp", []), b.get("sel_lp", [])
        for i in range(min(len(ent), len(rank))):
            d = (la[i] - lb[i]) if i < len(la) and i < len(lb) \
                and la[i] is not None and lb[i] is not None else 0.0
            rows.append((ent[i], kl[i] if i < len(kl) else 0.0,
                         rank[i] or 1, d))
    out = {"n": len(rows), "bands": []}
    if not rows:
        return out
    out["kl"] = sum(r[1] for r in rows) / len(rows)
    out["cov"] = (sum(sum(t.get("cov", [])) for t in turns)
                  / max(sum(len(t.get("cov", [])) for t in turns), 1))
    out["flip"] = sum(1 for r in rows if r[2] > 1) / len(rows)
    for lo, hi, label in BANDS:
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            continue
        ranks = sorted(r[2] for r in sel)
        out["bands"].append({
            "label": label, "n": len(sel),
            "flip": sum(1 for r in sel if r[2] > 1) / len(sel),
            "kl": sum(r[1] for r in sel) / len(sel),
            "dlp": sum(r[3] for r in sel) / len(sel),
            "p99": ranks[min(len(ranks) - 1, int(0.99 * len(ranks)))],
            "max": ranks[-1],
        })
    return out


def report(arms: dict, args: argparse.Namespace,
           running: tuple = ARMS) -> None:
    ref = arms["off"]["turns"]
    with open(args.transcript) as f:
        src = json.load(f)
    print(f"\n{args.model}  {len(ref)} turns from {src['source']} "
          f"({src['conversations']} conversations), budget {args.budget}")
    print(f"  context grows to {ref[-1]['prompt_chars']} chars, "
          f"{sum(len(t['ids']) for t in ref)} tokens generated\n")
    for name in running:
        turns = arms[name]["turns"]
        if args.force:
            st = forced_stats(ref, turns)
            print(f"  {name:8s} KL {st.get('kl', 0):.5f} "
                  f"(cov {st.get('cov', 0):.3f})  "
                  f"flip {st.get('flip', 0):.4f}  over {st['n']} steps")
            for band in st["bands"]:
                print(f"           {band['label']:8s} n={band['n']:5d}  "
                      f"flip {band['flip']:.4f}  KL {band['kl']:.5f}  "
                      f"dlp {band['dlp']:+.5f}  "
                      f"rank p99 {band['p99']:4d} max {band['max']}")
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
    if "churn" in arms:
        ok = all(a["ids"] == b["ids"] and a["lp"] == b["lp"]
                 for a, b in zip(ref, arms["churn"]["turns"]))
        print(f"\n  {'churn is bit-identical on a real session' if ok
                     else 'CHURN DIVERGED -- the machinery lost something'}")
    # A budget that never binds measures nothing, and an arm that evicted
    # nothing looks identical to a perfect one. Say so rather than printing
    # a table of 1.0000.
    idle = [n for n in running if n not in ("off", "truncate")
            and arms[n].get("pager", {}).get("copied_out", 0) == 0]
    if idle:
        print(f"\n  [!] {', '.join(idle)} evicted nothing -- the budget did "
              f"not bind, so this run compares nothing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("--transcript", default="session.json")
    ap.add_argument("--fetch", default="",
                    help="write a transcript here instead of running")
    ap.add_argument("--traj", default="",
                    help="build the transcript from a mini-swe-agent .traj.json")
    ap.add_argument("--strip-reasoning", action="store_true",
                    help="drop prior reasoning from history, as the chat "
                         "template does in a real agent loop")
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
    ap.add_argument("--block-size", type=int, default=16,
                    help="only to convert a block budget into a token cut")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--truncate-tokens", type=int, default=0,
                    help="run with the prompt cut to this many tokens")
    ap.add_argument("--truncate-sinks", type=int, default=32,
                    help="leading tokens the truncation keeps")
    ap.add_argument("--force", action="store_true",
                    help="decode the baseline's tokens in every arm and score "
                         "per step, instead of stopping at first divergence")
    ap.add_argument("--script", default="",
                    help="internal: the baseline run whose tokens to force")
    ap.add_argument("--arms", default="",
                    help="comma-separated subset to run; off is always first")
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

    running = ARMS
    if args.arms:
        want = [a.strip() for a in args.arms.split(",")]
        bad = [a for a in want if a not in ARMS]
        if bad:
            raise SystemExit(f"unknown arm(s): {', '.join(bad)}")
        running = tuple(["off"] + [a for a in want if a != "off"])
    arms = {}
    with tempfile.TemporaryDirectory() as tmp:
        script = ""
        for name in running:
            path = os.path.join(tmp, f"{name}.json")
            print(f"=== {name} ===", flush=True)
            env = {**os.environ, **ENV[name]}
            if name not in ("off", "truncate"):
                env["VLLM_VIRTUALKV_BUDGET"] = args.budget
            cut = []
            if name == "truncate":
                if not args.budget.isdigit():
                    raise SystemExit(
                        "the truncate arm needs --budget in blocks, so the "
                        f"token cut is unambiguous; got {args.budget!r}")
                cut = ["--truncate-tokens",
                       str(int(args.budget) * args.block_size)]
            cmd = [sys.executable, "-u", __file__, args.model,
                   "--transcript", args.transcript,
                   "--turns", str(args.turns),
                   "--max-len", str(args.max_len),
                   "--max-tokens", str(args.max_tokens),
                   "--util", str(args.util),
                   *(["--audit"] if args.audit else []),
                   *(["--thinking"] if args.thinking else []),
                   *(["--script", script] if script else []), *cut,
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
    report(arms, args, running)
    return 0


if __name__ == "__main__":
    sys.exit(main())
