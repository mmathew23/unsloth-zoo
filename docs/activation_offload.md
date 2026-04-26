# Activation Offload — Production Recipes

Activation offload reduces FSDP2 peak GPU memory at gradient-checkpoint
boundaries by moving saved tensors to pinned host memory and restoring
them on the backward pass. Unsloth ships several offload backends
behind environment variables; everything is opt-in (default behavior
unchanged).

## Two production recipes

Both have been validated against `native` (no offload) on torch
2.11.0+cu128, B200, SFT bs=4 seq=4096, with loss + grad-norm parity
inside `compare_training_runs(loss_tol=5e-3, grad_tol=8e-2)`.

### bounce_eager — text-heavy, NVLink datacenter GPUs

```bash
UNSLOTH_GC_OFFLOAD_BACKEND=hooks
UNSLOTH_GC_BOUNCE=1
UNSLOTH_GC_EAGER_PREFETCH=1
```

Measurements vs native:

|        | reserved | next5 |
|--------|---------:|------:|
| text   | −2.13 GiB | +0.7% |
| VL     | −2.35 GiB | −0.8% |

Mechanism: pack hook copies the source tensor into a GPU bounce slot
on the main stream, then the offload stream issues D2H to pinned host.
The bounce slot decouples source-tensor lifetime from the slow D2H.
Eager prefetch issues H2D restores ahead of demand.

**Hardware caveat**: the bounce-path D2D copy runs on the main stream.
On B200 NVLink (~3 TB/s) it consumes 0.16-0.19% of step time. On H100
or A100 NVLink (~0.6-0.9 TB/s) the cost stays small. On consumer PCIe
(~0.025 TB/s) projected step time degrades 15-24%. **Do not enable
this recipe on PCIe-only systems** without explicit verification.

### pref_d0_unshard — VL-heavy, cross-hardware

```bash
UNSLOTH_GC_OFFLOAD_BACKEND=hooks_prefetch
UNSLOTH_GC_PREFETCH_DEPTH=0
UNSLOTH_GC_FSDP2_UNSHARD_ASYNC_OP=1
```

Measurements vs native:

|        | reserved | next5 |
|--------|---------:|------:|
| text   | −1.13 GiB | −1.62% |
| VL     | **−8.55 GiB** | **+0.66%** |

Mechanism: the prefetch backend stages restore tensors in a per-slot
GPU ring that overlaps with FSDP2 unshard. `UNSHARD_ASYNC_OP=1` moves
the FSDP2 all-gather issue/wait to the default stream, eliminating
side-stream allocation fragmentation. Depth 0 keeps exactly one ring
slot live at a time.

**No hardware caveat**: there's no main-stream D2D copy, so PCIe is
fine. The trade-off is text throughput (-1.62%) for a much larger
VL memory win (-8.55 GiB).

## When to use which

| Workload signature | Recommendation |
|---|---|
| Pure text (Llama, Qwen text models) on NVLink | `bounce_eager` |
| Vision-language (Qwen-VL, Gemma-VL) on NVLink | `pref_d0_unshard` |
| Anything on PCIe / non-NVLinked GPUs | `pref_d0_unshard` |
| Memory pressure dominates throughput | `pref_d0_unshard` |
| Throughput dominates memory pressure | `bounce_eager` |

Both recipes pass loss + grad-norm parity, so you can pick on the
memory/throughput axis without correctness concern.

## Other env knobs (situational / research-only)

| Knob | Default | When to enable | Notes |
|---|---|---|---|
| `UNSLOTH_GC_PREFETCH_DEPTH` | `1` | Manual override | Set 0 to disable speculative prefetch (memory ↓, throughput depends on path) |
| `UNSLOTH_GC_PREFETCH_RING_SIZE` | `4` | Manual override | Number of restore-ring slots |
| `UNSLOTH_GC_EXPANDABLE_SEGMENTS` | `0` | Last-resort fragmentation fix | Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if no CUDA context yet. Reproducibility was poor across machine states; treat as opt-in safety net. |
| `UNSLOTH_GC_PREFETCH_RING_EAGER_FREE` | `0` | Last-resort | Drops persistent ring reference after restore_event. Memory savings only on fragmented allocator state; not reproducible on clean baselines. |
| `UNSLOTH_GC_FSDP2_CLEAR_FORWARD_PREFETCH` | `0` | **Diagnostic only** | Clears FSDP2 forward-prefetch lists. PFU_A confirmed no memory benefit on its own; pairing with `UNSHARD_ASYNC_OP=1` regresses throughput further than `UNSHARD_ASYNC_OP=1` alone. **Do not enable in production defaults.** |

## Backend reference

```bash
UNSLOTH_GC_OFFLOAD_BACKEND=<value>
```

| Value | Behavior |
|---|---|
| `hooks` (default) | Broad `saved_tensors_hooks` — packs every eligible internal tensor at the offload stream |
| `hooks_prefetch` | Same pack, but unpack uses a prefetch ring that overlaps with FSDP2 unshard |
| `boundary` (alias for `noop`) | Uses `_NoopSaveInputs.setup_context` — packs only the checkpoint-region's input tensors. Lighter pack count but currently captures fewer saved tensors than the broad path; doesn't deliver memory savings on its own. |
| `noop` | Internal name for the boundary path; back-compat alias |

## Falsified avenues (do not use as production defaults)

These have been tested and reliably either fail to deliver claimed
savings or actively regress at least one workload:

- `bounce_eager + UNSHARD_ASYNC_OP=1` — serializes bounce D2D with FSDP2
  all-gather on the main stream. BEU_B2 measured +131.5 ms / 2 active
  steps (+1.76% main-stream time).
- `bounce_eager + size-tier filter` — VL pack distribution is bimodal
  (218 large + 31 small) but filtering small packs made VL slower with
  no memory win (BVL_A2, VL_BOUNDARY).
- `boundary` backend alone — `_NoopSaveInputs.setup_context` only
  captures one input per checkpoint region; no broad memory savings.
- `expandable_segments` / `eager_free` ring "wins" against contaminated
  baselines — not reproducible on clean GPU pair, kept as research-only
  safety nets.

## Loss & gradient parity

Every shipped recipe passes
`compare_training_runs(loss_tol=5e-3, grad_tol=8e-2)` against `native`
on a 7-step measurement window. Worst observed across the matrix:
`max_loss_diff=0.0027`, `max_grad_norm_diff=0.0263`. No NaN events on
parity runs.

## References

- HP_A3 / HP_A3 rerun — FSDP2 unshard / ring-slot fragmentation
  identification; `UNSHARD_ASYNC_OP` mechanism.
- VERIFY1..VERIFY4 — universal Pareto verification across knob
  combinations.
- PFU_A — knob ablation showing `UNSHARD_ASYNC_OP=1` carries the win,
  `CLEAR_FORWARD_PREFETCH=1` adds nothing.
- BEU_A / BEU_B / BEU_B2 — interaction trace confirming bounce + unshard
  serialization on the main stream.
- BVL_A2 / BVL_B / BVL_C / VL_BOUNDARY — VL pack distribution and
  failure modes of size-filter / event-only / boundary alternatives.
- BE_A — bounce_eager D2D bandwidth sensitivity; PCIe failure mode.
