# The attention sink, and how badly it was tuned

The single largest effect measured in this project is not a policy. It is
the number of leading blocks kept unconditionally.

## Two blocks hold half the attention

On a real session, **the first two blocks of 112 hold 0.4580 of all
attention mass**. That is the attention-sink phenomenon, and it dominates
every mass measurement here:

| selector, 25% budget | mass captured |
|---|---|
| oracle | 0.8812 |
| 2-bit keys, one step stale | 0.8811 |
| **sinks + recency (what ships)** | **0.7907** |
| recency without sinks | 0.3384 |

Adding two sink blocks moves recency from 0.34 to 0.79. Every comparison in
this repo that used a sinkless recency baseline overstated its selector by
about 2.3× — see `demand-signal.md`, where correcting it shrank a claimed 3.2×
advantage to 11%.

## The optimum is about a third of the budget

Swept on Qwen3.5-9B at a 12672t budget, 528-token blocks, 163 turns of an
agent trajectory, `recency`:

| sink | tokens | % of budget | agreement | drift | turns identical |
|---|---|---|---|---|---|
| 1 | 528 | 4% | 0.2312 | 0.026007 | 24 |
| 2 | 1056 | 8% | 0.1993 | 0.026666 | 15 |
| 4 | 2112 | 17% | 0.2563 | 0.025229 | 30 |
| **8** | **4224** | **33%** | **0.2604** | **0.023756** | **35** |
| 16 | 8448 | 67% | 0.2074 | 0.028295 | 25 |

Peak at a third of the budget spent on the front of the context, with drift
bottoming and identical turns peaking at the same point. Worth 13% over
`sink=1` and 31% over the worst setting — more than any policy change
measured here.

**Do not theorise the sink=2 dip.** Agreement is chaotic in the
configuration: a turn scores its identical *prefix*, so one early flipped
token discards the whole turn, and a one-block change can cause it. All three
columns dip together because all three are driven by the same few
divergences — one cause, not three confirmations.

## The knob is denominated wrong

`sink` is a count of *blocks*, so its meaning moves with the model: 2 blocks
is 32 tokens on Qwen3-8B and 1056 on a hybrid. That has caused three
separate failures — it consumed the *entire* budget on a hybrid needle run,
confounded a granularity sweep into giving the opposite answer, and makes the
default (2) sit at 8% of budget where the optimum is 33%.

It should take the token and percentage forms `budget` already accepts, and
default near a third of the budget. `recent` and `host_slots` have the same
defect; see `serving-knobs` in TODO.md.
