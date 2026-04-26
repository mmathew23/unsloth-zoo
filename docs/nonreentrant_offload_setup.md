# Non-Reentrant Activation Offload — Setup & Operator Guide

Audience: a future engineer (human or AI) who has no prior context on
this work. Goal: get from "fresh checkout" to "running a benchmark
that reproduces the production recipes" in this doc alone.

For a deeper measurement summary and the falsified-avenues list, see
the companion `activation_offload.md`. This file is the setup +
mechanism reference; that file is the empirical results reference.

---

## 1. What this is

Activation offload moves saved tensors from GPU to pinned host memory
during the forward pass and brings them back on the backward pass.
For FSDP2 + gradient checkpointing this can cut peak GPU memory
several GiB at a small throughput cost.

The implementation depends on **non-reentrant** gradient
checkpointing (`torch.utils.checkpoint(use_reentrant=False)`). The
reentrant path runs forward in a no-grad region and recomputes the
full subgraph in backward — there are no per-tensor save hooks to
intercept. The non-reentrant path goes through `saved_tensors_hooks`
and the `_NoopSaveInputs` autograd boundary, both of which we patch.

Two related repositories cooperate:

- **`unsloth_zoo`** (this repo, branch `feat/actoff`) — the
  gradient-checkpoint dispatch (`unsloth_checkpoint`), the offload
  backends, and the `saved_tensors_hooks` / `_NoopSaveInputs`
  patches. All `UNSLOTH_GC_*` knobs except the `_FSDP2_*` family
  live here.
- **`unsloth_fsdp2`** (sibling repo, branch `feat/nonreentrantgc`) —
  the FSDP2 trainer wrapper (`apply_simple_fsdp2`), the bench
  scripts, and the `UNSLOTH_GC_FSDP2_*` patches against
  `FSDPParamGroup.wait_for_unshard` and `FSDPCommContext`.

Both must be on the PYTHONPATH for offload to function end-to-end.

## 2. Repository layout (this workspace)

```
/home/mathew/fsdptesting/workspace_0/
├── include/
│   ├── actoff_clones/
│   │   ├── unsloth_zoo/             feat/actoff   <-- this repo
│   │   │   └── unsloth_zoo/gradient_checkpointing.py   (heart of offload)
│   │   └── unsloth/                 feat/actoff   (compiler/RL glue)
│   └── unsloth_fsdp2/               feat/nonreentrantgc
│       ├── unsloth_dist/accelerate_fsdp2_trainer.py    (apply_simple_fsdp2 + FSDP2 patches)
│       └── scripts/unsloth_release/                    (bench scripts)
├── temp/research_v3_HP_A2/venvs/torch211/               (the canonical torch 2.11 venv)
└── async_task_outputs/                                  (research reports)
```

The torch 2.11 venv is `temp/research_v3_HP_A2/venvs/torch211/bin/python`.
All measured numbers in `activation_offload.md` come from this venv on
B200 NVLink GPUs.

## 3. Minimum viable setup (run a benchmark)

```bash
PY=/home/mathew/fsdptesting/workspace_0/temp/research_v3_HP_A2/venvs/torch211/bin/python
WS=/home/mathew/fsdptesting/workspace_0
PYTHONPATH=$WS/include/actoff_clones/unsloth_zoo:$WS/include/actoff_clones/unsloth:$WS/include/unsloth_fsdp2

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=$PYTHONPATH \
$PY -m torch.distributed.run --standalone --nproc_per_node=2 \
  $WS/include/unsloth_fsdp2/scripts/unsloth_release/benchmark_sft_text_simple_fsdp_unsloth.py \
  --output_dir /tmp/bench_native
```

That runs the SFT-text benchmark with native (no offload) at the
script's defaults. The defaults are intentionally productionlike —
LoRA on, gradient checkpointing on, unsloth_gc on, torch.compile
disabled — because deviating from any of them changes which dependency
is on the critical path and therefore changes which knobs help (see
§ 7 below — "Why scale matters").

To enable an offload recipe, prepend env vars per § 5.

## 4. The non-reentrant dispatch chain

When an offload-eligible workload runs, the call chain is:

```
SFTTrainer / train loop
  └─ HF model.forward()
      └─ each transformer block calls _gradient_checkpointing_func(...)
          └─ unsloth_checkpoint(fn, *args, use_reentrant=...)              [gradient_checkpointing.py:2694]
              └─ if FSDP2 detected: auto-promote use_reentrant=False
              └─ _unsloth_checkpoint_nonreentrant(fn, *args, ...)          [gradient_checkpointing.py:2558]
                  ├─ resolve_gc_offload_backend(env UNSLOTH_GC_OFFLOAD_BACKEND)
                  ├─ for backend == "hooks" | "hooks_prefetch":
                  │     with UnslothOffloadActivations(dtype=...):         [activation_offloading.py]
                  │         original_checkpoint(fn, *args, use_reentrant=False)
                  │         (saved_tensors_hooks intercepts every tensor)
                  └─ for backend == "noop" | "boundary":
                        _noop_offload_state.set({"offloader": offloader})  [gradient_checkpointing.py:2636]
                        original_checkpoint(fn, *args, use_reentrant=False)
                        (patched _NoopSaveInputs.setup_context intercepts only inputs)
```

Two key patches make this work:

1. **`patch_unsloth_smart_gradient_checkpointing(dtype, use_reentrant)`**
   (`gradient_checkpointing.py:2742`) — replaces
   `torch.utils.checkpoint.checkpoint` and the corresponding
   `transformers.modeling_utils.checkpoint` with `unsloth_checkpoint`.
   Also calls `_patch_noop_save_inputs()`.

2. **`_patch_noop_save_inputs()`** (`gradient_checkpointing.py:856`) —
   replaces `torch.utils.checkpoint._NoopSaveInputs.setup_context` with
   `unsloth_setup_context`, which reads the `_noop_offload_state`
   ContextVar; if an offloader is present, eligible inputs go to
   `offloader.pack_hook()` instead of `ctx.save_for_backward()`.

   The non-reentrant `checkpoint(use_reentrant=False)` path uses
   `_NoopSaveInputs` internally to mark the boundary between the
   checkpointed region and the rest of the graph. By patching
   `setup_context` we get a clean place to offload only the tensors
   that cross the boundary (the "boundary" backend) — vs. the broad
   `saved_tensors_hooks` path that intercepts every internal tensor.

`patch_unsloth_smart_gradient_checkpointing()` must be called before
training. The bench scripts do this when `--unsloth_gc=True` (the
default).

## 5. Production recipes

There are two validated recipes. Pick by workload signature.
Measurements are vs. native on B200 NVLink, torch 2.11.0+cu128, SFT
bs=4 seq=4096:

### Recipe A — `bounce_eager` (text-heavy / NVLink datacenter)

```bash
UNSLOTH_GC_OFFLOAD_BACKEND=hooks
UNSLOTH_GC_BOUNCE=1
UNSLOTH_GC_EAGER_PREFETCH=1
```

text: −2.13 GiB / +0.7%   VL: −2.35 GiB / −0.8%

The pack hook copies into a GPU bounce slot on the main stream, then
the offload stream issues D2H to pinned host. The bounce slot
decouples source-tensor lifetime from the slow D2H. Eager prefetch
issues H2D restores ahead of demand.

**Hardware caveat**: the bounce-path D2D copy runs on the main stream.
B200/H100 NVLink absorbs it (<0.2% step time). On consumer PCIe
(~0.025 TB/s) projected step time degrades 15-24%. Do not enable on
PCIe-only systems without verifying.

### Recipe B — `pref_d0_unshard` (VL-heavy / cross-hardware)

```bash
UNSLOTH_GC_OFFLOAD_BACKEND=hooks_prefetch
UNSLOTH_GC_PREFETCH_DEPTH=0
UNSLOTH_GC_FSDP2_UNSHARD_ASYNC_OP=1
```

text: −1.13 GiB / −1.62%   VL: **−8.55 GiB** / **+0.66%**

The prefetch backend stages restore tensors in a per-slot GPU ring
that overlaps with FSDP2 unshard. `UNSHARD_ASYNC_OP=1` moves the FSDP2
all-gather issue/wait to the default stream, eliminating side-stream
allocation fragmentation. Depth 0 keeps exactly one ring slot live at
a time.

No hardware caveat. The trade-off is text throughput (-1.62%) for a
much larger VL memory win.

| Workload signature | Recipe |
|---|---|
| Pure text on NVLink | A (`bounce_eager`) |
| Vision-language on NVLink | B (`pref_d0_unshard`) |
| Anything on PCIe | B |
| Memory pressure dominates | B |
| Throughput dominates | A |

Both recipes pass loss + grad-norm parity vs. native within
`compare_training_runs(loss_tol=5e-3, grad_tol=8e-2)`.

## 6. Complete env knob inventory

### 6.1 unsloth_zoo offload knobs

Read in `unsloth_zoo/gradient_checkpointing.py` unless noted. Default
is the empty string (off / no-op) unless listed.

| Knob | Default | What it does |
|---|---|---|
| `UNSLOTH_GC_OFFLOAD_BACKEND` | `hooks` | `hooks` (broad), `hooks_prefetch` (broad + restore ring), `boundary` / `noop` (boundary inputs only) |
| `UNSLOTH_GC_PREFETCH_DEPTH` | `1` | Speculative H2D depth on prefetch backend; `0` = one slot live at a time |
| `UNSLOTH_GC_PREFETCH_RING_SIZE` | `4` | Restore-ring slot count; must be > max prefetch depth |
| `UNSLOTH_GC_BOUNCE` | off | Pack via GPU bounce slot before D2H (recipe A) |
| `UNSLOTH_GC_EAGER_PREFETCH` | off | Issue H2D restores eagerly at forward boundary (recipe A) |
| `UNSLOTH_GC_MIN_OFFLOAD_MB` | `2` | Minimum tensor size in MiB to qualify for offload |
| `UNSLOTH_GC_SKIP_LAST_N` | `1` | Trailing layers to keep resident on GPU |
| `UNSLOTH_GC_PIN_POOL_MIB` | off | Pre-allocate pinned-buffer pool (size MiB per slot) |
| `UNSLOTH_GC_PIN_POOL_COUNT` | `64` | Pool slot count, used only with `PIN_POOL_MIB` |
| `UNSLOTH_GC_AUX_STREAM` | off | Enable secondary CUDA stream for D2D copies |
| `UNSLOTH_GC_DISABLE_CPU_OFFLOAD` | off | Pass `use_reentrant=False` through with no offload (debugging) |
| `UNSLOTH_GC_EXPANDABLE_SEGMENTS` | off | Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` if no CUDA context yet; must be set at launch |
| `UNSLOTH_GC_PREFETCH_RING_EAGER_FREE` | off | Drop persistent ring reference after `restore_event`; last-resort fragmentation fix |
| `UNSLOTH_GC_NO_RECORD_STREAM` | off | Skip `tensor.record_stream`; pair with `PACK_WAIT_EVENT` |
| `UNSLOTH_GC_PACK_WAIT_EVENT` | off | Use explicit `Event.wait` instead of `record_stream` |
| `UNSLOTH_GC_SYNC_PACK` | off | Force `extra_stream.synchronize()` after each pack (safe, no overlap) |
| `UNSLOTH_GC_NARROW_BOTH` | off | Diagnostic: keep both stream waits but narrow the second |
| `UNSLOTH_GC_NARROW_WAIT` | off | **UNSAFE**, removes pre-restore main-stream fence |
| `UNSLOTH_GC_NVTX` | off | Emit NVTX markers around pack/unpack |
| `UNSLOTH_GC_PROFILE` | off | Per-module/per-shape pack/unpack timing dump at exit |
| `UNSLOTH_GC_CHECKPOINT_HIT_COUNT` | off | Per-step pack/unpack counts and prefetch hit/miss at exit |
| `UNSLOTH_GC_TARGET_GPU_DUMMY` | off | Diagnostic: allocate "CPU buffers" on GPU (timing only, garbage backward) |
| `UNSLOTH_SMART_GC_FSDP2` | `auto` | Set `off`/`disable` to suppress auto-promotion of `use_reentrant=False` when FSDP2 detected |

### 6.2 unsloth_fsdp2 trainer knobs

Read in `unsloth_dist/accelerate_fsdp2_trainer.py`.

| Knob | Default | What it does |
|---|---|---|
| `UNSLOTH_GC_FSDP2_UNSHARD_ASYNC_OP` | off | Calls `_set_unshard_async_op(True)` on every FSDP2-wrapped module; serializes unshard onto default stream. Recipe B uses this. |
| `UNSLOTH_GC_FSDP2_AVOID_WAIT` | off | Routes async unshard onto FSDP2's high-priority all-gather stream + wires singleton fwd/bwd prefetch chains. Workload-dependent; see § 7. |
| `UNSLOTH_GC_FSDP2_WAIT_LAZY` | off | Defers `wait_for_unshard` copy-out from pre-forward hook to the consuming module's forward boundary via `forward_pre_hook`. Stacks with AVOID_WAIT. |
| `UNSLOTH_GC_FSDP2_CLEAR_FORWARD_PREFETCH` | off | **Diagnostic only.** Clears explicit forward-prefetch lists; suppresses `configure_fsdp2_avoid_wait_prefetch` (mutually exclusive). Do not enable in production. |

Bench-script-level knobs (in `benchmark_sft_vl_simple_fsdp_unsloth.py`):

| Knob | Default | What it does |
|---|---|---|
| `UNSLOTH_GC_STEPEND_DRAIN` | off | Trainer callback that issues `wait_stream` + tiny alloc each step end; diagnostic for stream-sync correctness |
| `UNSLOTH_GC_WARMUP_SLOTS` | `0` | Pre-allocate + free N segments to warm allocator before training |
| `UNSLOTH_GC_WARMUP_MIB` | `64` | Size of each warmup segment (only with `WARMUP_SLOTS > 0`) |

## 7. Why scale matters (read before changing recipes)

Two prior research runs measured `UNSLOTH_GC_FSDP2_AVOID_WAIT=1` on
the same hardware, same script, same env, but different bs/seq:

| Run | bs | seq | native | pref_d0_async | +AVOID_WAIT | Recovery |
|---|---|---|---|---|---|---|
| PTC_FIX_B | 2 | 2048 | 26855 | 26204 | 26642 | **67%** |
| VAL_FINAL | 4 | 4096 | 38005 | 36674 | 36681 | **0.08%** |

Same code. Different operating point. The `wait_event(all_gather_event)`
fence at `_fsdp_collectives.py:447` is a near-fixed cost per FSDP
unshard. Per-step compute scales with bs×seq, so the fence's fraction
of step time shrinks as the operating point grows — at bs=4/seq=4096
it's invisible, at bs=2/seq=2048 it's the bottleneck.

This means: **the recipe recommendations in § 5 are anchored to
production scale (bs=4, seq=4096)**. At smaller scales other knobs
become attractive (e.g., `AVOID_WAIT` alone on text). Open follow-up:
sweep bs/seq to locate the crossover and turn this into a concrete
enable rule.

## 8. Bench script catalog

All in
`include/unsloth_fsdp2/scripts/unsloth_release/`. Defaults assume the
production-scale recipes are anchored at bs=4 seq=4096; the script
defaults are smaller (bs=2 seq=2048) for fast iteration. Override on
the command line.

| Script | Default model | bs / seq | LoRA | GC | unsloth_gc | use_reentrant | max_steps |
|---|---|---|---|---|---|---|---|
| `benchmark_sft_text_simple_fsdp_unsloth.py` | `unsloth/Meta-Llama-3.1-8B-Instruct` | 2 / 2048 | True | True | True | False | 30 |
| `benchmark_sft_vl_simple_fsdp_unsloth.py` | `unsloth/Qwen2.5-VL-7B-Instruct` | 2 / 2048 | True | True | True | False | 30 |
| `benchmark_grpo_deepseek_simple_fsdp_vllm_release_unsloth.py` | `unsloth/DeepSeek-R1-0528-Qwen3-8B` | 1 / 4096 | False | True | True | (n/a) | -1 |
| `benchmark_grpo_gemma3_simple_fsdp_vllm_unsloth.py` | `unsloth/gemma-3-1b-it` | 8 / 2048 | False | (default) | True | False | -1 |
| `benchmark_grpo_vl_simple_fsdp_release_unsloth.py` | `unsloth/Qwen2.5-VL-7B-Instruct` | 1 / 8192 | False | True | True | False | -1 |

Both SFT scripts set `os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")` at the top — torch.compile is off by default for these benches.

To match the production-scale numbers in § 5, override:
`--per_device_train_batch_size 4 --max_seq_length 4096`.

## 9. Falsified avenues (do NOT re-pursue without new evidence)

These have been measured and rejected. From `activation_offload.md`:

- `bounce_eager + UNSHARD_ASYNC_OP=1` — serializes bounce D2D with FSDP2 all-gather; +1.76% main-stream time; alternative D2D placements all regressed throughput AND inflated VL reserved memory.
- `bounce_eager + size-tier filter` — VL bimodal pack distribution; filtering small packs made VL slower with no memory win.
- `boundary` backend alone (without further patches) — captures only LM-side checkpoint inputs on VL; misses the vision-encoder packs that broad hooks catches. Not a universal replacement.
- `expandable_segments` / `eager_free` ring as defaults — wins only on contaminated baselines; not reproducible on clean GPU pair. Kept as last-resort safety nets.
- Line-level `wait_event` override (MP_SURGICAL) — regressed throughput on both workloads.
- `DEFER_WAIT` / FIX_A pattern (eager copy-out on side stream + record `copy_out_done`) — leaks reserved memory by ~12 GiB. `WAIT_LAZY` is the correct method-level patch instead.
- `WAIT_LAZY + AVOID_WAIT` as a default — fails the universal Pareto gate (text −1.62% on production-scale benches). Kept as opt-in for VL workloads that specifically want the +1.9% recovery on top of recipe B.

## 10. Where the research artifacts live

Surviving reports:
- `/home/mathew/fsdptesting/workspace_0/async_task_outputs/research_v3_PTC_FIX_B_alt_implement.md` — original AVOID_WAIT measurement at bs=2/seq=2048 (67% recovery)
- `.../research_v3_VAL_FINAL_wait_lazy_avoid_wait.md` — head-to-head at bs=4/seq=4096 (0.08% recovery)
- `.../research_v3_MP_WRAPPER_torch211.md` — WAIT_LAZY design listing (the original sandboxed implementation was cleaned; this report is the source of the reconstruction)
- `.../research_v3_VAL_AB_combined.md` — the FIX_A leak measurement

Surviving patched-tree directories:
- `/home/mathew/fsdptesting/workspace_0/temp/research_v3_PTC_FIX_B/patched_fsdp2/` — AVOID_WAIT-only reference impl (matches the committed code byte-for-byte modulo docstring placement)
- `/home/mathew/fsdptesting/workspace_0/temp/research_v3_VAL_AB/patched_fsdp2_fix_a_reconstructed/` — the leaky DEFER_WAIT design (do not copy)

The merged WAIT_LAZY+AVOID_WAIT tree from VAL_FINAL was in a sandbox
dir that has been cleaned. The committed code on `feat/nonreentrantgc`
(`accelerate_fsdp2_trainer.py`, commit `ddb71ea`) is the canonical
implementation.

## 11. How to add a new knob

1. Read the env var with `_env_enabled()` (in
   `accelerate_fsdp2_trainer.py`) or
   `os.environ.get(...) in ("1", "true", "True")` (in
   `gradient_checkpointing.py`).
2. Default OFF. No exception.
3. Document in this file (§ 6) and the relevant section of
   `activation_offload.md`.
4. Add a CPU parity test before any GPU benchmark.
5. Validate against `compare_training_runs(loss_tol=5e-3, grad_tol=8e-2)`
   on at least one text and one VL workload.
6. If the win is workload-dependent (compare § 7), document the
   regime explicitly. Do not promote a knob to a default based on a
   single operating point.

## 12. Checklist for a fresh agent picking this up

- [ ] `git -C include/actoff_clones/unsloth_zoo branch --show-current` says `feat/actoff`
- [ ] `git -C include/unsloth_fsdp2 branch --show-current` says `feat/nonreentrantgc`
- [ ] `temp/research_v3_HP_A2/venvs/torch211/bin/python -c 'import torch; print(torch.__version__)'` prints `2.11.0+cu128`
- [ ] At least 2 GPUs free in `nvidia-smi` from the candidate pool `[0,1,2,3,4,7]`
- [ ] PYTHONPATH includes both `unsloth_zoo` and `unsloth_fsdp2`
- [ ] Read this file, then `activation_offload.md`, then
      `async_task_outputs/research_v3_VAL_FINAL_wait_lazy_avoid_wait.md`
      for the most recent end-to-end validation
