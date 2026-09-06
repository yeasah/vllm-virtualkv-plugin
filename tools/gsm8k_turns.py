#!/usr/bin/env python3
"""GSM8K as a conversation: multi-turn pressure that is still repeatable.

Between a benchmark of independent prompts and pointing real traffic at the
thing, this sits in the middle. Each turn appends the previous question and
answer and asks the next one, so the context grows, the prefix cache hits
nearly everything, and the request shapes vary -- while the task still has
ground truth and the same trials can be replayed against every arm.

**What it is sensitive to, which is not what the needle test is sensitive to.**
Each GSM8K question needs the exemplars at the start and the question at the
end, and nothing in between: no policy worth the name evicts either, so a
correct pager should lose *no* accuracy here even at a tight budget. That makes
a drop a mechanism signal rather than a policy one, which is the opposite of
`quality.py`, where recency loses a needle because it cannot fetch. Passing
here says the machinery survives sustained multi-turn use; it says nothing
about whether a policy is any good.

**Every arm sees the same conversation.** Turns are extended with the *gold*
answer, never the model's own. Feeding back what the model said would fork the
context the first time two arms disagreed, and every turn after that would be
comparing different conversations rather than different residency -- the same
trap that makes generated-token comparison a bad instrument.

**`--chat` renders the conversation with the model's own template**, and for
anything modern that is the mode that matters. Two reasons, neither cosmetic:
instruction-tuned models degenerate on raw completion prompts in ways that have
nothing to do with residency, and a *thinking* model puts most of its decode
tokens in a reasoning trace -- which is the bulk of what a pager has to serve
during generation, and is absent entirely from a few-shot completion. A
residency policy measured without it is measured on the wrong token
distribution.

    tools/gsm8k_turns.py MODEL [--turns 24] [--budget 25%] [--shots 4] [--chat]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARMS = ("off", "churn", "recency", "quest")
ENV = {
    "off": {"VLLM_VIRTUALKV": "0"},
    "quest": {"VLLM_VIRTUALKV": "1", "VLLM_VIRTUALKV_POLICY": "quest"},
    "churn": {"VLLM_VIRTUALKV": "1", "VLLM_VIRTUALKV_POLICY": "churn",
              "VLLM_VIRTUALKV_SHOW_PENDING": "1"},
    "recency": {"VLLM_VIRTUALKV": "1", "VLLM_VIRTUALKV_POLICY": "recency"},
}


def render(tok, messages, thinking: bool) -> str:
    """Apply the chat template, asking for thinking only where that is a thing.

    `enable_thinking` is a Qwen-ism rather than a standard, so it is offered
    and withdrawn if the template does not take it -- guessing from the
    template's text would be its own small bug.
    """
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def gold_of(answer: str) -> str:
    return _normalise(answer.split("####")[-1].strip().replace(",", ""))


def predicted(text: str) -> str | None:
    """The last number in the answer, normalised the way the gold one is.

    The trailing dot matters: a regex ending in an optional dot happily matches "18." at the end of a
    sentence, and comparing that against a gold "18" as strings scores a
    correct answer wrong. It did, on this harness, and the resulting accuracy
    column was wrong per-arm rather than uniformly -- an arm whose model
    happened to end with a period was penalised against one that did not, which
    is exactly the shape of a difference that gets mistaken for a result.
    """
    hits = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not hits:
        return None
    return _normalise(hits[-1])


def _normalise(value: str) -> str:
    value = value.strip().rstrip(".")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def one_arm(args):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from vllm_virtualkv import WorkerPager

    made = []
    original = WorkerPager.__init__

    def tracked(self, *a, **k):
        original(self, *a, **k)
        made.append(self)

    WorkerPager.__init__ = tracked

    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    shots = "".join(
        f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n"
        for i in range(args.shots)
    )

    if args.audit:
        os.environ["VLLM_VIRTUALKV_AUDIT"] = "1"
    llm = LLM(model=args.model, max_model_len=args.max_len,
              gpu_memory_utilization=args.util, enforce_eager=True,
              enable_prefix_caching=True, max_num_seqs=1,
              trust_remote_code=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, logprobs=5,
        stop=None if args.chat else ["\n\n", "Question:"])

    # In chat mode the turns are real messages and the template does the
    # framing; the gold answer still stands in for the model's own, so every
    # arm sees one conversation.
    messages: list[dict] = []
    turns, context = [], shots
    for t in range(args.turns):
        item = test[t]
        if args.chat:
            messages.append({"role": "user", "content": item["question"]})
            prompt = render(tok, messages, args.thinking)
        else:
            prompt = context + f"Question: {item['question']}\nAnswer:"
        out = llm.generate([prompt], params)[0]
        text = out.outputs[0].text
        turns.append({
            "gold": gold_of(item["answer"]),
            "pred": predicted(text),
            "ids": [int(x) for x in out.outputs[0].token_ids],
            "lp": [{str(k): round(v.logprob, 6) for k, v in s.items()}
                   for s in (out.outputs[0].logprobs or [])],
            "prompt_chars": len(prompt),
        })
        # The gold answer, not the model's: every arm must see one conversation.
        if args.chat:
            messages.append({"role": "assistant", "content": item["answer"]})
        else:
            context = prompt + f" {item['answer']}\n\n"

    result = {"turns": turns}
    if made:
        s = made[0].summary()
        result["pager"] = {k: v for k, v in s.items()
                           if k not in ("tier", "guard")}
        result["tier"] = s["tier"]
        result["guard"] = {"violations": s["guard"]["violations"],
                           "by_check": s["guard"]["by_check"],
                           "steps": s["guard"]["steps_checked"]}
    with open(args.out, "w") as f:
        json.dump(result, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--budget", default="25%")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--util", type=float, default=0.60)
    ap.add_argument("--chat", action="store_true",
                    help="render turns with the model's chat template")
    ap.add_argument("--thinking", action="store_true",
                    help="let a thinking model think, in chat mode")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.arm:
        return one_arm(args) or 0

    arms = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ARMS:
            path = os.path.join(tmp, f"{name}.json")
            print(f"=== {name} ===", flush=True)
            env = {**os.environ, **ENV[name]}
            if name != "off":
                env["VLLM_VIRTUALKV_BUDGET"] = args.budget
            cmd = [sys.executable, "-u", __file__, args.model,
                   "--turns", str(args.turns), "--shots", str(args.shots),
                   "--max-len", str(args.max_len),
                   "--max-tokens", str(args.max_tokens),
                   "--util", str(args.util),
                   *(["--audit"] if args.audit else []),
                   *(["--chat"] if args.chat else []),
                   *(["--thinking"] if args.thinking else []),
                   "--arm", name, "--out", path]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode != 0:
                print(proc.stdout[-1500:], proc.stderr[-2500:])
                raise SystemExit(f"arm {name} failed")
            with open(path) as f:
                arms[name] = json.load(f)
    report(arms, args)
    return 0


def report(arms, args):
    ref = arms["off"]["turns"]
    n = len(ref)
    base_hits = [t["pred"] == t["gold"] for t in ref]
    mode = "chat" + (" +thinking" if args.thinking else "") if args.chat \
        else f"{args.shots}-shot completion"
    print(f"\n{args.model}  {n} turns, {mode}, budget "
          f"{args.budget}, prefix caching on")
    print(f"  context grows to {ref[-1]['prompt_chars']} chars\n")
    for name in ARMS:
        turns = arms[name]["turns"]
        hits = [t["pred"] == t["gold"] for t in turns]
        exact = all(a["ids"] == b["ids"] for a, b in zip(ref, turns))
        lp_exact = exact and all(a["lp"] == b["lp"] for a, b in zip(ref, turns))
        flips = sum(1 for a, b in zip(base_hits, hits) if a != b)
        p = arms[name].get("pager", {})
        g = arms[name].get("guard", {})
        print(f"  {name:8s} {sum(hits):2d}/{n} correct"
              f"   {'bit-identical' if lp_exact else 'tokens identical' if exact else f'{flips} turn(s) flipped'}")
        au = p.get("audit")
        if au:
            print(f"           mass missed {au['missed_mass']:.4f}  worst layer "
                  f"{au['worst_layer_missed']:.3f}  fetched "
                  f"{au['fetched_mass']:.4f}  evicted {au['evicted_mass']:.4f}"
                  f"  over {au['steps']} steps, "
                  f"{au['resident_blocks']:.0f}/{au['total_blocks']:.0f} blocks")
        if p:
            print(f"           out {p.get('copied_out', 0):5d} in "
                  f"{p.get('copied_in', 0):5d}  unbacked "
                  f"{p.get('missing_host_copy', 0)}  refused "
                  f"{p.get('evictions_refused', 0)}  released "
                  f"{p.get('released', 0)}"
                  f"   guard {g.get('violations', 0)}/{g.get('steps', 0)}")

    churn = arms["churn"]["turns"]
    ok = all(a["ids"] == b["ids"] and a["lp"] == b["lp"]
             for a, b in zip(ref, churn))
    print(f"\n  {'churn is bit-identical under multi-turn prefix pressure'
            if ok else 'CHURN DIVERGED -- the machinery lost something'}")
    if arms["churn"].get("guard", {}).get("violations"):
        print("  [!] guard violations in the churn arm")


if __name__ == "__main__":
    sys.exit(main())
