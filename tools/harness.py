"""Evaluation helpers, vendored from the project this grew out of.

These came from `vllm-exl3-plugin` (`tools/blocktable_permute.py`,
`tools/blocktable_evict.py`, `bench/core.py`), where the measurements that
justify this plugin were taken. Copied rather than imported so this repo stands
alone; the originals remain the record of how each was arrived at.

The one that carries an argument is `compare`. Comparing generated tokens is a
bad instrument for a KV change: it conflates numerical error with model
confidence, and once two runs diverge they are continuing *different* contexts,
so everything after the first difference compares prompts rather than
arithmetic. So divergence is truncated at the first token disagreement, and
`top_token_gone` counts the steps where the reference's own choice fell out of
the other side's top-k -- a large divergence that a mean over what remains
would hide.
"""

from __future__ import annotations

import math
import re
import os
import random


def kl(p: dict, q: dict) -> float:
    """KL(P||Q) over P's support, renormalized.

    Both sides are truncated to top-k, so Q may not cover all of P. Restricting
    to the shared support and renormalizing keeps this finite; it understates
    divergence when the top-k sets disagree, which is itself reported separately.
    """
    shared = [t for t in p if t in q]
    if not shared:
        return float("nan")
    zp = math.log(sum(math.exp(p[t]) for t in shared))
    zq = math.log(sum(math.exp(q[t]) for t in shared))
    total = 0.0
    for t in shared:
        lp, lq = p[t] - zp, q[t] - zq
        total += math.exp(lp) * (lp - lq)
    return total


def haystack(tok, ctx):
    """`ctx` tokens of wikitext-103, the same source `niah_kv.py` draws on."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text, i = "", 0
    while len(text) < (ctx + 2000) * 4:
        text += ds[i]["text"]
        i += 1
    return tok.encode(text)[:ctx]


def capture(result):
    gen = result.outputs[0]
    steps = []
    for pos in gen.logprobs or []:
        steps.append({str(t): round(lp.logprob, 6) for t, lp in pos.items()})
    return {
        "prompt_len": len(result.prompt_token_ids),
        "ids": [int(t) for t in gen.token_ids],
        "steps": steps,
    }


def compare(ref, arm):
    """Per-decode-step divergence, truncated at the first token disagreement.

    Past that point the two arms are continuing different token sequences, so
    their distributions are not answering the same question and averaging them
    in would understate or overstate divergence arbitrarily.

    `clean` is the one step that needs no floor to interpret. Generated token 0
    comes out of the prefill forward, which no arm touches -- so at step 1 the
    two arms have bit-identical weights, cache and input token, and whatever
    separates their distributions is the rewrite and nothing else, with no
    compounding yet. It is the probe's measurement taken at the logits instead
    of at layer 0, which makes it the only one available under CUDA graphs.
    """
    n = min(len(ref["ids"]), len(arm["ids"]))
    first = next((i for i in range(n) if ref["ids"][i] != arm["ids"][i]), None)
    upto = n if first is None else first + 1
    kls, dtop, gone = [], [], 0
    for i in range(upto):
        a, b = ref["steps"][i], arm["steps"][i]
        if not a or not b:
            continue
        kls.append(kl(a, b))
        t = max(a, key=a.get)
        if t in b:
            dtop.append(abs(a[t] - b[t]))
        else:
            # The reference's own choice is not even in the other arm's top-k.
            # That is the largest divergence this metric can see, and it is
            # invisible in `dlogprob_max`, which averages over what remains.
            gone += 1
    finite = [v for v in kls if v == v]
    return {
        "steps": upto,
        "first_divergence": first,
        "ids_match": first is None and len(ref["ids"]) == len(arm["ids"]),
        "kl_max": max(finite) if finite else 0.0,
        # No comparable entry means the reference's top token fell out of the
        # other arm's top-k entirely, which is a large divergence -- reporting
        # it as 0.0 would read as agreement.
        "dlogprob_max": max(dtop) if dtop else (0.0 if not kls else float("nan")),
        "dlogprob_mean": (sum(dtop) / len(dtop) if dtop
                          else (0.0 if not kls else float("nan"))),
        # Step 0 is the prefill's own output: it must be identical in every
        # arm, and if it is not, the run's premise is broken rather than its
        # result interesting.
        "top_token_gone": gone,
        "prefill_step_clean": len(kls) > 0 and kls[0] == 0.0,
        "clean_kl": kls[1] if len(kls) > 1 else float("nan"),
        "clean_dlogprob": dtop[1] if len(dtop) > 1 else float("nan"),
    }


NEEDLE = " The special magic Denver number is: {value}. "


QUESTION = ("\n\nWhat is the special magic Denver number mentioned in the text "
            "above? Answer with the number only.\nAnswer:")


def needle_prompt(tok, ids, block_size, depth_blocks, value):
    """Splice a magic number into the haystack at a known block boundary.

    Planting it at an exact multiple of the block size is what makes the two
    needle arms differ in one block and nothing else -- a needle straddling a
    boundary would live in two blocks and could survive the loss of either.
    """
    text = NEEDLE.format(value=value)
    ndl = tok.encode(text, add_special_tokens=False)
    at = depth_blocks * block_size
    spliced = ids[:at] + ndl + ids[at + len(ndl):]
    q = tok.encode(QUESTION, add_special_tokens=False)
    return spliced + q, at // block_size, (at + len(ndl) - 1) // block_size


def answered(text, value):
    return str(value) in re.sub(r"[,\s]", "", text)

