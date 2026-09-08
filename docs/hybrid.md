# Hybrid attention models

Linear/GDN attention in most layers, full attention in a few. Qwen3.5 and
its relatives, and what gets deployed — so the plugin has to work here, and
mostly does.

## What the plugin sees

Qwen3.5-9B has 8 full-attention layers of 32 (every fourth), 16 query heads
and 4 KV heads. That is **128 head-layer pairs sharing one block table**
against Qwen3-8B's 1152, and 32 KiB of KV per token against 144 KiB.

vLLM builds **four** KV cache groups for it — three `MambaSpec` groups of 8
layers each, and one `PagedAttentionSpec` — with the paged group at **index
3**. Three of four groups are mamba at the same page size, so ~75% of the KV
allocation is linear-attention state.

## What had to change

**Find the paged group; never assume index 0.** `block_tables`,
`slot_mappings` and `kernel_block_sizes` are per group, and `runner.kv_caches`
is ordered by layer index across *every* layer holding state, so it
interleaves K/V caches with mamba state. Slicing it wholesale feeds conv/ssm
state to key extraction as though its first `head_size` channels were keys —
which produces plausible tensors, not a crash.

`groups.resolve()` finds the paged group once and **refuses** on anything
ambiguous: no paged group, more than one, or a `kv_caches` length that
contradicts the layer ordering. A pager that will not start beats one that
pages the wrong thing.

**The guard needs the same group.** It indexes
`coordinator.single_type_managers`, and checking the paged view against the
mamba group's allocation table reports every block as unowned — 22 violations
in 22 steps until it was passed `group=self.group.index`.

## The runner trap

vLLM carries **two GPU model runners with the same class name**:
`vllm.v1.worker.gpu_model_runner` and `vllm.v1.worker.gpu.model_runner`. It
picks between them per model configuration, not per version:

```python
if model_config.is_hybrid and not is_default_v2_architecture:
    return False        # -> V1
```

`DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` has 8 entries and none are Qwen, so
**hybrids default to V1 by design** — the runner this plugin does not patch.
The hook installs, never fires, and the plugin loads, reports nothing, evicts
nothing and returns correct output. Indistinguishable from working.

`WorkerPager.fired_or_raise()` exists for exactly this, and every tool calls
it before reporting. Until V1 is supported, hybrids need
`VLLM_USE_V2_MODEL_RUNNER=1`.

Any diagnostic printing `type(x).__name__` is actively misleading here.
Print `__module__`.

## What works

On Qwen3.5-9B-exl3 with `VLLM_USE_V2_MODEL_RUNNER=1`:

- `full` is **bit-identical** to no plugin.
- The needle is **found by `oracle`** at 26% residency with 528-token blocks
  and lost by `recency` — the same shape as the original 16-token result on
  Llama-3.2-1B.
- **Zero guard violations** over ~230k guarded steps, 163 turns, 84k-token
  contexts, all four checks active.

## The rig

    max_model_len 131072, bf16 KV, util 0.85   ->  initialises
    weights 5.48 GiB, KV 5.57 GiB              ->  ~11 GiB of 15.5
    KV capacity 177,332 tokens                 ->  33 KiB/token, all groups
    block size 528                             ->  248 blocks at full context

Language model only; for max-context runs, util 0.95 and the host tier cap
raised to 50%. **Do not use fp8** — see `granularity.md`.

## Open

V1 runner support, or a refusal at startup. Hybrids default to it, so
without one of the two the plugin silently does nothing on the models that
matter most.
