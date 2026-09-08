# Measuring this thing without fooling yourself

Nearly every quantitative claim this project made from reasoning rather than
measurement was wrong, and every claim that moved moved because a *control*
was wrong rather than a mechanism. This is the accumulated list, because the
same mistakes are cheap to repeat.

## What the harnesses measure

`tools/session_turns.py` replays a fixed transcript — an agent trajectory or
a chat session — through every arm, so all arms see byte-identical history
and the only variable is residency. There is no ground truth and none is
needed: the reference is the `off` arm, and the score is fidelity to it.

- **agreement** — baseline tokens emitted before the first divergence,
  summed over turns. Free-running, so no leak, but it truncates: a turn ends
  at its first wrong token.
- **drift** — mean |Δlogprob| on the tokens it did agree on. Sensitive to an
  arm sitting near a flip while still matching.
- **`--force`** — decodes the baseline's tokens in every arm, so every step
  is scored. Does not truncate, but **leaks**: the reference tokens were
  generated *with* the blocks the arm evicted, so feeding them in hands back
  what eviction removed, increasingly with position in the turn.
- **`--segment K`** — free-run K tokens, resync, repeat. Bounds cascade
  without the leak. Negative K is a segment *count*: −1 free-running, −2
  exactly one resync. Cut positions are drawn from the turn index, so they
  are deterministic, identical across arms, and spread across the interval.

## The resolution limit

**Agreement cannot resolve differences below ~0.03 at 163 turns.** A turn
scores its identical prefix, so one early flipped token discards the whole
turn's contribution, and a one-block configuration change can cause it. Three
metrics moving together is usually one cause showing up three times, not
three confirmations.

Noise falls as roughly 1/√turns, so halving the floor needs ~4× the turns and
~4× the runtime. `--traj a.json,b.json` concatenates trajectories for that;
it buys resolution and distractor load, not recall distance.

## Controls that were missing and mattered

- **`truncate`** — no plugin, prompt cut to the same tokens recency keeps.
  Without it, "beat recency" was the bar, and recency turns out to *be*
  truncation on some workloads (0.4606 against 0.4565, and worse on drift and
  exact turns). The bar is truncation.
- **Sinks in the baseline.** A sinkless recency selector understates itself
  by ~2.3×; see `sink.md`.
- **Representative decode volume.** `--thinking` produced 617 tokens/turn
  against 138 without, and that difference reversed a result.
- **A budget that binds.** A percentage budget against a short context evicts
  nothing and prints a column of 1.0000. The report now says so.

## Workloads, and what each cannot show

- **The needle** answers in the first token or two, before a query-aware
  policy has seen a query. It is policy-sensitive by construction and cannot
  show anything about the damage a policy does elsewhere.
- **GSM8K turns** are independent questions, so the history is ballast and no
  policy is punished for dropping it. Mechanism-sensitive, policy-insensitive.
- **Chat sessions** (UltraChat) have topic-local dependency: a 2048-token
  window holds the entire current topic, so nothing distant is needed and
  keeping it cannot pay.
- **Agent trajectories** (SWE-bench, `--traj`) have the structure: the task
  statement is in message 1 and every later turn still depends on it, with
  file contents read early and referenced hundreds of turns later.
  `reasoning_content` is **93% of assistant text** and must be included, but
  Qwen3's template *strips* prior `<think>` blocks — so `--strip-reasoning`
  matches deployment and produces a less redundant, harder transcript.

## Instrument bugs that changed conclusions

Recorded because each looked like a result:

- **`audit.py` never applied the attention scale.** Captured queries enter
  `Attention.forward` unscaled; the backend applies `1/√head_size` inside. So
  the audit softmaxed logits ~11× too large, off a far sharper distribution
  than the model's, understating exactly what it existed to report.
- **The working-set softmax excluded the partial tail block**, handing its
  mass to older blocks. That alone moved an oracle from 99.5% to 26.4%.
- **`--force` was never forwarded to arm subprocesses**, so no driver-level
  forced run ever produced data — and the report rendered zero rows as a
  flawless `flip 0.0000`.
- **The over-budget selection cut by block index**, discarding the *newest*
  blocks whenever sinks + recent + unscored exceeded the budget, which is the
  normal early state of a request. Found by a measurement, not a test.
- **`quality.py` defaulted to 1024 host slots** — 32 MiB at 32 KiB/block and
  **16.9 GiB** at 16.5 MiB/block, which OOM-killed the process on a hybrid.

## The pattern

Four of those share one shape: **a component that is installed but inert**.
The forcer patched a superseded `Sampler` class; the pager patched a
superseded runner class; a budget never bound; a flag never reached a
subprocess. Each produced a plausible table.

The defence is to check that a thing *ran*, not that it was configured:
`fired_or_raise()`, the "evicted nothing" warning, the forced-stats refusal
on zero rows, the host-tier byte cap. Three of those fired on real failures
within hours of being written.

The other recurring shape is **units**. `sink`, `recent` and `host_slots` are
denominated in blocks, and a block is not a fixed size — 32 KiB on one model,
16.5 MiB on another. That has caused an OOM kill, a confounded sweep, and a
budget entirely consumed by sinks. Anything checked against a threshold
should be derived from it, in the units the resource is actually spent in.
