# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import numpy as np
import functools
from typing import Union, Optional, List, Any
import os
import warnings
import gc
from .utils import _get_dtype, Version
from .device_type import (
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
)

__all__ = [
    "calculate_n_gradient_checkpoints",
    "prepare_n_gradient_checkpoints",
    "Unsloth_Offloaded_Gradient_Checkpointer",
    "unsloth_offloaded_gradient_checkpoint",
    "patch_unsloth_gradient_checkpointing",
    "unpatch_unsloth_gradient_checkpointing",

    "Unsloth_Gradient_Checkpointer",
    "unsloth_gradient_checkpoint",
    "patch_gradient_checkpointing",
    "unpatch_gradient_checkpointing",

    "patch_unsloth_smart_gradient_checkpointing",
    "unpatch_unsloth_smart_gradient_checkpointing",
    "reset_unsloth_gradient_checkpointing_buffers",
    "UnslothGradientCheckpointer",
    "resolve_sac_context_fn",
    "set_sac_policy",
    "get_sac_stats",
    "reset_sac_stats",
    "resolve_gc_offload_backend",
    "set_offload_backend",
    "_bind_gradient_checkpointing_func",
    "_set_unsloth_checkpoint_state",
    "_install_unsloth_stream_offload_wrapper",
    "_remove_unsloth_stream_offload_wrapper",
]

# ── Selective Activation Checkpointing (SAC) ──────────────────────────
#
# SAC lets you selectively save expensive ops' outputs (e.g. attention, matmul)
# instead of recomputing everything during backward.  It composes with CPU
# offloading: SAC decides *what* to save, saved_tensors_hooks decides *where*.
# However under torch compile saved_tensors_hooks may not be compatible with 
# SAC CPU_OFFLOAD policies.
#
# Only available with use_reentrant=False (PyTorch checkpoint context_fn kwarg).

try:
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )
    _SAC_AVAILABLE = True
except ImportError:
    _SAC_AVAILABLE = False

# Op sets are resolved lazily on first SAC use, not at import time.
_SAC_ATTENTION_OPS = None
_SAC_MATMUL_OPS = None
_SAC_ADD_OPS = None
_SAC_STATS = {
    "calls": 0,
    "must_save": 0,
    "prefer_save": 0,
    "must_cpu_offload": 0,
    "prefer_cpu_offload": 0,
    "prefer_recompute": 0,
    "ops": {},
    "phases": {
        "forward": {
            "calls": 0,
            "must_save": 0,
            "prefer_save": 0,
            "must_cpu_offload": 0,
            "prefer_cpu_offload": 0,
            "prefer_recompute": 0,
            "ops": {},
        },
        "recompute": {
            "calls": 0,
            "must_save": 0,
            "prefer_save": 0,
            "must_cpu_offload": 0,
            "prefer_cpu_offload": 0,
            "prefer_recompute": 0,
            "ops": {},
        },
    },
}
_GC_OFFLOAD_BACKENDS = {
    "unsloth_original",
    "unsloth_stream",
}
# Release backend surface: `unsloth_original` preserves the normal Unsloth
# checkpointing path; `unsloth_stream` is the supported activation-offload
# backend for both non-reentrant and reentrant stream checkpointing.
_gc_profile_patched = False


def _maybe_patch_gc_profile():
    """Install opt-in GC profiling outside the production hot path."""
    global _gc_profile_patched
    if _gc_profile_patched:
        return
    profile_value = os.environ.get("UNSLOTH_GC_PROFILE", "0")
    hit_count_value = os.environ.get("UNSLOTH_GC_CHECKPOINT_HIT_COUNT", "0")
    if (
        str(profile_value).strip().lower() in ("0", "false", "no", "off", "")
        and str(hit_count_value).strip().lower() in ("0", "false", "no", "off", "")
    ):
        return
    from ._profile.gradient_checkpointing import patch_unsloth_gc_profile
    patch_unsloth_gc_profile()
    _gc_profile_patched = True


def resolve_gc_offload_backend(backend = None):
    if backend is None:
        backend = os.environ.get("UNSLOTH_GC_OFFLOAD_BACKEND", "unsloth_original")
    backend = str(backend).strip().lower()
    if backend not in _GC_OFFLOAD_BACKENDS:
        raise ValueError(
            f"Unsloth: Unknown GC offload backend {backend!r}. "
            f"Available: {sorted(_GC_OFFLOAD_BACKENDS)}"
        )
    return backend


def _is_fsdp2_module(run_function) -> bool:
    """True iff ``run_function`` is bound to a module managed by FSDP2 fully_shard
    with real sharding (world_size > 1). Used by ``unsloth_checkpoint`` to auto-
    force ``use_reentrant=False`` under FSDP2 -- the reentrant path leaks memory
    via retained unshard state across layers, while the non-reentrant stream
    activation offload path composes cleanly with FSDP2.
    """
    import functools as _ft
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import (
            _get_module_fsdp_state,
        )
    except Exception:
        return False

    func = run_function
    while isinstance(func, _ft.partial):
        func = func.func
    module = getattr(func, "__self__", None)
    if module is None:
        return False
    try:
        if _get_module_fsdp_state(module) is None:
            return False
    except Exception:
        return False

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size() > 1
    except Exception:
        pass
    return False


def _try_resolve_op(name):
    try:
        parts = name.split(".")
        obj = torch.ops
        for p in parts:
            obj = getattr(obj, p)
        return obj
    except AttributeError:
        return None


def _sac_profile_enabled():
    return str(os.environ.get("UNSLOTH_SAC_PROFILE", "0")).strip().lower() not in (
        "0", "false", "no", "off", ""
    )


def _record_sac_policy_decision(ctx, op, decision):
    if not _sac_profile_enabled():
        return
    phase = "recompute" if getattr(ctx, "is_recompute", False) else "forward"
    phase_stats = _SAC_STATS["phases"][phase]
    _SAC_STATS["calls"] += 1
    phase_stats["calls"] += 1
    if decision == CheckpointPolicy.MUST_SAVE:
        _SAC_STATS["must_save"] += 1
        phase_stats["must_save"] += 1
    elif decision == CheckpointPolicy.PREFER_SAVE:
        _SAC_STATS["prefer_save"] += 1
        phase_stats["prefer_save"] += 1
    elif decision == CheckpointPolicy.MUST_CPU_OFFLOAD:
        _SAC_STATS["must_cpu_offload"] += 1
        phase_stats["must_cpu_offload"] += 1
    elif decision == CheckpointPolicy.PREFER_CPU_OFFLOAD:
        _SAC_STATS["prefer_cpu_offload"] += 1
        phase_stats["prefer_cpu_offload"] += 1
    else:
        _SAC_STATS["prefer_recompute"] += 1
        phase_stats["prefer_recompute"] += 1
    name = str(op)
    ops = _SAC_STATS["ops"]
    ops[name] = ops.get(name, 0) + 1
    phase_ops = phase_stats["ops"]
    phase_ops[name] = phase_ops.get(name, 0) + 1


def reset_sac_stats():
    _SAC_STATS["calls"] = 0
    _SAC_STATS["must_save"] = 0
    _SAC_STATS["prefer_save"] = 0
    _SAC_STATS["must_cpu_offload"] = 0
    _SAC_STATS["prefer_cpu_offload"] = 0
    _SAC_STATS["prefer_recompute"] = 0
    _SAC_STATS["ops"] = {}
    for phase_stats in _SAC_STATS["phases"].values():
        phase_stats["calls"] = 0
        phase_stats["must_save"] = 0
        phase_stats["prefer_save"] = 0
        phase_stats["must_cpu_offload"] = 0
        phase_stats["prefer_cpu_offload"] = 0
        phase_stats["prefer_recompute"] = 0
        phase_stats["ops"] = {}


def get_sac_stats():
    return {
        "calls": int(_SAC_STATS["calls"]),
        "must_save": int(_SAC_STATS["must_save"]),
        "prefer_save": int(_SAC_STATS["prefer_save"]),
        "must_cpu_offload": int(_SAC_STATS["must_cpu_offload"]),
        "prefer_cpu_offload": int(_SAC_STATS["prefer_cpu_offload"]),
        "prefer_recompute": int(_SAC_STATS["prefer_recompute"]),
        "ops": dict(_SAC_STATS["ops"]),
        "phases": {
            phase: {
                "calls": int(stats["calls"]),
                "must_save": int(stats["must_save"]),
                "prefer_save": int(stats["prefer_save"]),
                "must_cpu_offload": int(stats["must_cpu_offload"]),
                "prefer_cpu_offload": int(stats["prefer_cpu_offload"]),
                "prefer_recompute": int(stats["prefer_recompute"]),
                "ops": dict(stats["ops"]),
            }
            for phase, stats in _SAC_STATS["phases"].items()
        },
    }


def _ensure_sac_ops():
    """Lazily resolve op handles on first use."""
    global _SAC_ATTENTION_OPS, _SAC_MATMUL_OPS, _SAC_ADD_OPS
    if _SAC_ATTENTION_OPS is not None:
        return

    _SAC_ATTENTION_OPS = set()
    for op_name in (
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_math.default",
        "aten._scaled_dot_product_cudnn_attention.default",
        "aten._flash_attention_forward.default",
        "aten._efficient_attention_forward.default",
    ):
        op = _try_resolve_op(op_name)
        if op is not None:
            _SAC_ATTENTION_OPS.add(op)

    _SAC_MATMUL_OPS = set()
    for op_name in (
        "aten.mm.default",
        "aten.mm.out",
        "aten.bmm.default",
        "aten.bmm.out",
        "aten.addmm.default",
        "aten.addmm.out",
    ):
        op = _try_resolve_op(op_name)
        if op is not None:
            _SAC_MATMUL_OPS.add(op)

    _SAC_ADD_OPS = set()
    for op_name in (
        "aten.add.Tensor",
    ):
        op = _try_resolve_op(op_name)
        if op is not None:
            _SAC_ADD_OPS.add(op)


def _sac_policy_attn_only(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS:
        decision = CheckpointPolicy.MUST_SAVE
    else:
        decision = CheckpointPolicy.PREFER_RECOMPUTE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_attn_prefer_save(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS:
        decision = CheckpointPolicy.PREFER_SAVE
    else:
        decision = CheckpointPolicy.PREFER_RECOMPUTE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_attn_must_save_budget(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS:
        decision = CheckpointPolicy.MUST_SAVE
    else:
        # Under torch.compile, PREFER_SAVE can still be overridden by the
        # AOTAutograd activation memory budget. PREFER_RECOMPUTE is treated by
        # this runtime's partitioner as forced recompute, which is too strong
        # for budgeted SAC.
        decision = CheckpointPolicy.PREFER_SAVE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_budget_only(ctx, op, *args, **kwargs):
    decision = CheckpointPolicy.PREFER_SAVE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_small_matmul_must_save_budget(ctx, op, *args, **kwargs):
    max_out_features = int(os.environ.get("UNSLOTH_SAC_MATMUL_MAX_OUT_FEATURES", "4096"))
    save_matmul = False
    if op in _SAC_MATMUL_OPS:
        out_features = _matmul_output_features(op, args, kwargs)
        save_matmul = out_features is not None and out_features <= max_out_features
    if save_matmul:
        decision = CheckpointPolicy.MUST_SAVE
    else:
        decision = CheckpointPolicy.PREFER_SAVE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_attn_and_small_matmul_must_save_budget(ctx, op, *args, **kwargs):
    max_out_features = int(os.environ.get("UNSLOTH_SAC_MATMUL_MAX_OUT_FEATURES", "4096"))
    save_matmul = False
    if op in _SAC_MATMUL_OPS:
        out_features = _matmul_output_features(op, args, kwargs)
        save_matmul = out_features is not None and out_features <= max_out_features
    if op in _SAC_ATTENTION_OPS or save_matmul:
        decision = CheckpointPolicy.MUST_SAVE
    else:
        decision = CheckpointPolicy.PREFER_SAVE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_attn_cpu_offload(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS:
        decision = CheckpointPolicy.MUST_CPU_OFFLOAD
    else:
        decision = CheckpointPolicy.PREFER_RECOMPUTE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_attn_prefer_cpu_offload(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS:
        decision = CheckpointPolicy.PREFER_CPU_OFFLOAD
    else:
        decision = CheckpointPolicy.PREFER_RECOMPUTE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_attn_and_add(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS or op in _SAC_ADD_OPS:
        decision = CheckpointPolicy.MUST_SAVE
    else:
        decision = CheckpointPolicy.PREFER_RECOMPUTE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _sac_policy_attn_and_matmul(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS or op in _SAC_MATMUL_OPS:
        decision = CheckpointPolicy.MUST_SAVE
    else:
        decision = CheckpointPolicy.PREFER_RECOMPUTE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


def _matmul_output_features(op, args, kwargs):
    try:
        if op is _try_resolve_op("aten.mm.default"):
            return int(args[1].shape[-1])
        if op is _try_resolve_op("aten.mm.out"):
            out = kwargs.get("out") if kwargs else None
            return int(out.shape[-1]) if out is not None else int(args[1].shape[-1])
        if op is _try_resolve_op("aten.bmm.default"):
            return int(args[1].shape[-1])
        if op is _try_resolve_op("aten.bmm.out"):
            out = kwargs.get("out") if kwargs else None
            return int(out.shape[-1]) if out is not None else int(args[1].shape[-1])
        if op is _try_resolve_op("aten.addmm.default"):
            return int(args[2].shape[-1])
        if op is _try_resolve_op("aten.addmm.out"):
            out = kwargs.get("out") if kwargs else None
            return int(out.shape[-1]) if out is not None else int(args[2].shape[-1])
    except Exception:
        return None
    return None


def _sac_policy_attn_and_small_matmul(ctx, op, *args, **kwargs):
    max_out_features = int(os.environ.get("UNSLOTH_SAC_MATMUL_MAX_OUT_FEATURES", "4096"))
    save_matmul = False
    if op in _SAC_MATMUL_OPS:
        out_features = _matmul_output_features(op, args, kwargs)
        save_matmul = out_features is not None and out_features <= max_out_features
    if op in _SAC_ATTENTION_OPS or save_matmul:
        decision = CheckpointPolicy.MUST_SAVE
    else:
        decision = CheckpointPolicy.PREFER_RECOMPUTE
    _record_sac_policy_decision(ctx, op, decision)
    return decision


_SAC_PRESETS = {
    "attn_only": _sac_policy_attn_only,
    "attn_prefer_save": _sac_policy_attn_prefer_save,
    "attn_must_save_budget": _sac_policy_attn_must_save_budget,
    "budget_only": _sac_policy_budget_only,
    "small_matmul_must_save_budget": _sac_policy_small_matmul_must_save_budget,
    "attn_and_small_matmul_must_save_budget": _sac_policy_attn_and_small_matmul_must_save_budget,
    "attn_cpu_offload": _sac_policy_attn_cpu_offload,
    "attn_prefer_cpu_offload": _sac_policy_attn_prefer_cpu_offload,
    "attn_and_add": _sac_policy_attn_and_add,
    "attn_and_matmul": _sac_policy_attn_and_matmul,
    "attn_and_small_matmul": _sac_policy_attn_and_small_matmul,
}


def resolve_sac_context_fn(policy):
    """Resolve a user-provided SAC policy to a context_fn callable (or None).

    Args:
        policy: One of:
            - None → no SAC (returns None)
            - str  → preset name ("attn_only", "attn_prefer_save",
              "attn_cpu_offload", "attn_prefer_cpu_offload",
              "attn_and_add", "attn_and_matmul")
            - list of OpOverloads → ops whose outputs to save
            - callable(ctx, op, *args, **kwargs) → CheckpointPolicy

    Returns:
        A callable suitable for the ``context_fn`` kwarg of
        ``torch.utils.checkpoint.checkpoint``, or None.
    """
    if policy is None:
        return None

    if not _SAC_AVAILABLE:
        raise RuntimeError(
            "Unsloth: SAC requires PyTorch >= 2.4 with "
            "torch.utils.checkpoint.CheckpointPolicy support."
        )

    _ensure_sac_ops()

    allow_cache_entry_mutation = None
    env_allow_mutation = os.environ.get("UNSLOTH_SAC_ALLOW_CACHE_ENTRY_MUTATION")
    if env_allow_mutation is not None:
        allow_cache_entry_mutation = str(env_allow_mutation).strip().lower() in (
            "1", "true", "yes", "on",
        )

    if isinstance(policy, str):
        if policy not in _SAC_PRESETS:
            raise ValueError(
                f"Unsloth: Unknown SAC preset {policy!r}. "
                f"Available: {list(_SAC_PRESETS.keys())}"
            )
        policy_fn = _SAC_PRESETS[policy]
        if allow_cache_entry_mutation is None:
            # Inductor commonly lowers matmuls to `aten.mm.out`; PyTorch SAC
            # otherwise rejects those saved outputs as mutated cache entries.
            allow_cache_entry_mutation = policy == "attn_and_matmul"
    elif isinstance(policy, (list, tuple, set)):
        save_ops = set(policy)
        def policy_fn(ctx, op, *args, **kwargs):
            if op in save_ops:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy):
        policy_fn = policy
    else:
        raise TypeError(
            "Unsloth: sac_policy must be None, a string, a list of ops, "
            f"or a callable, got {type(policy).__name__}"
        )
    if allow_cache_entry_mutation is None:
        allow_cache_entry_mutation = False

    return functools.partial(
        create_selective_checkpoint_contexts,
        policy_fn,
        allow_cache_entry_mutation=allow_cache_entry_mutation,
    )


# Useful for more than just SAC
def _bind_gradient_checkpointing_func(
    model,
    checkpoint_fn,
    use_reentrant,
    context_fn = None,
    offload_backend = None,
):
    """Bind _gradient_checkpointing_func on all modules with the given settings.

    Builds a functools.partial from non-None kwargs and assigns it to every
    module that already has _gradient_checkpointing_func.
    """
    partial_kwargs = {}
    if use_reentrant is not None:
        partial_kwargs["use_reentrant"] = use_reentrant
    # context_fn is only valid for the non-reentrant path. `unsloth_stream` also
    # has a reentrant dispatch path below, so keep that backend bound in both
    # modes.
    if (use_reentrant is False) and context_fn is not None:
        partial_kwargs["context_fn"] = context_fn
    if offload_backend is not None:
        partial_kwargs["offload_backend"] = resolve_gc_offload_backend(offload_backend)

    if partial_kwargs:
        bound = functools.partial(checkpoint_fn, **partial_kwargs)
    else:
        bound = checkpoint_fn

    for module in model.modules():
        if not hasattr(module, "_gradient_checkpointing_func"):
            continue
        module._gradient_checkpointing_func = bound


def _set_unsloth_checkpoint_state(model, use_reentrant, offload_backend):
    use_reentrant = bool(use_reentrant)
    offload_backend = resolve_gc_offload_backend(offload_backend)
    seen = set()
    current = model
    while current is not None and id(current) not in seen:
        current._unsloth_use_reentrant = use_reentrant
        current._unsloth_gc_offload_backend = offload_backend
        seen.add(id(current))
        current = getattr(current, "model", None)
    return use_reentrant, offload_backend


def set_sac_policy(model, policy):
    """Set or clear SAC policy at runtime (no model reload needed).

    Args:
        model: The model (must have been loaded with use_reentrant=False).
        policy: Same as ``sac_policy`` in ``from_pretrained``, or None to disable.

    Raises:
        ValueError: If the model uses reentrant checkpointing.
    """
    _maybe_patch_gc_profile()
    use_reentrant = getattr(model, "_unsloth_use_reentrant", True)
    if use_reentrant and policy is not None:
        raise ValueError(
            "Unsloth: SAC requires use_reentrant=False. "
            "Re-load the model with use_reentrant=False to use SAC."
        )

    context_fn = resolve_sac_context_fn(policy)
    model._unsloth_sac_context_fn = context_fn
    offload_backend = getattr(model, "_unsloth_gc_offload_backend", None)

    checkpoint_fn = torch.utils.checkpoint.checkpoint
    _bind_gradient_checkpointing_func(
        model, checkpoint_fn, use_reentrant, context_fn, offload_backend,
    )


def set_offload_backend(model, backend):
    _maybe_patch_gc_profile()
    backend = resolve_gc_offload_backend(backend)
    use_reentrant = getattr(model, "_unsloth_use_reentrant", True)
    use_reentrant, backend = _set_unsloth_checkpoint_state(
        model, use_reentrant, backend,
    )
    context_fn = getattr(model, "_unsloth_sac_context_fn", None)
    dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
    patch_unsloth_smart_gradient_checkpointing(
        dtype=dtype,
        use_reentrant=use_reentrant,
        offload_backend=backend,
    )
    _bind_gradient_checkpointing_func(
        model, unsloth_checkpoint, use_reentrant, context_fn, backend,
    )
    if backend == "unsloth_stream":
        _install_unsloth_stream_offload_wrapper(model, dtype)
    else:
        _remove_unsloth_stream_offload_wrapper(model)
    return backend


# Initial buffer sizes for gradient checkpointing
INITIAL_CPU_BUFFER_SIZE = 128 * 1024       # Initial size per CPU buffer
INITIAL_GPU_BUFFER_SIZE = 2 * 256 * 2048   # Initial size per GPU buffer
INITIAL_CPU_BUFFER_COUNT = 200             # Initial number of CPU buffers
DOUBLE_BUFFER_HEADROOM = 512 * 1024 * 1024 # 512MB minimum free CUDA memory to enable double buffering

torch_version = torch.__version__
if Version(torch_version) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
pass


def _calculate_n_gradient_checkpoints(
    n_layers : int,
    method   : Optional[Union[str, int]] = "sqrt",
) -> List[int]:
    assert(type(n_layers) is int and n_layers > 0)

    if method is None: method = "sqrt"

    if method == "sqrt":
        n_checkpoints = int(n_layers**0.5)
    elif type(method) is int and method > 0:
        n_checkpoints = int(np.ceil(n_layers / method))
    else:
        raise ValueError("method must be 'sqrt' or an int >0 and <= n_layers.")

    size = n_layers // n_checkpoints
    sizes = np.full(n_checkpoints, size, dtype = int)
    leftovers = n_layers % n_checkpoints
    # We append leftovers from the right
    for k in range(leftovers):
        sizes[n_checkpoints-1-k] += 1
    boundaries = np.hstack((0, np.cumsum(sizes)))
    boundaries = boundaries.tolist()
    return boundaries
pass


def calculate_n_gradient_checkpoints(
    n_layers              : int,
    layers_per_checkpoint : Optional[Union[str, int]] = "sqrt",
) -> List[int]:
    assert(type(n_layers) is int and n_layers > 0)

    if layers_per_checkpoint is None or layers_per_checkpoint == 1:
        return None

    boundaries = _calculate_n_gradient_checkpoints(n_layers, layers_per_checkpoint)

    assert(boundaries[0] == 0 and boundaries[-1] == n_layers)
    assert(min(boundaries) == 0 and max(boundaries) == n_layers)
    assert(np.diff(boundaries).min() >= 0)
    return boundaries
pass


def prepare_n_gradient_checkpoints(
    model                 : Any,
    layers_per_checkpoint : Optional[Union[str, int]] = "sqrt",
    use_reentrant         : Optional[bool] = True,
) -> None:
    """
    Calculates where to place the gradient checkpoints given n_layers.

    Args:
        model: Any LlamaModel with layers.
        layers_per_checkpoint (`Union[str, int]`, *optional*):
            Can either be `sqrt` or an integer for how many layers per checkpoint you want.
            The more, the less memory usage, but can be slower. Default is `sqrt`.
            Choose 1 for Pytorch gradient checkpointing. 2 to wrap 2 layers in 1 module etc.
        use_reentrant (`bool`, *optional*):
            https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L354
            Optimal gradient checkpointing algorithm `use_reentrant=False` which will
            be the default in future Pytorch versions doesn't seem to work??
    """
    _model = None
    if hasattr(model, "layers"):
        _model = model
    elif hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            _model = model.model
    if _model is None:
        raise TypeError("`model` or `model.model` does not have attribute `layers`. Are you sure this is a model?")
    pass

    n_layers = len(_model.layers)
    boundaries = calculate_n_gradient_checkpoints(n_layers, layers_per_checkpoint)
    _model._gradient_checkpointing_boundaries    = boundaries
    _model._gradient_checkpointing_use_reentrant = use_reentrant
pass


class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    All Unsloth Zoo code licensed under LGPLv3
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        ctx.device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking = True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.device, non_blocking = True).detach()
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
    pass
pass


class Unsloth_Gradient_Checkpointer(torch.autograd.Function):
    """
    All Unsloth Zoo code licensed under LGPLv3
    Same as normal gradient checkpointing but cleaner
    """
    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output
    pass

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad_(True)
        with torch.enable_grad():
            (output,) = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,)*len(ctx.args)
    pass
pass


@torch._disable_dynamo
def unsloth_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return Unsloth_Gradient_Checkpointer.apply(function, *args)
pass


def patch_unsloth_gradient_checkpointing():
    print("Unsloth: Patched gradient checkpointing for long context finetuning.")
    import torch.utils
    if torch.utils.checkpoint.checkpoint.__name__ == "unsloth_offloaded_gradient_checkpoint": return
    torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = unsloth_offloaded_gradient_checkpoint
    import transformers.modeling_utils
    transformers.modeling_utils.checkpoint = unsloth_offloaded_gradient_checkpoint
    os.environ["UNSLOTH_PATCHED"] = "1"
pass


def patch_gradient_checkpointing():
    print("Unsloth: Patched gradient checkpointing.")
    import torch.utils
    if torch.utils.checkpoint.checkpoint.__name__ == "unsloth_gradient_checkpoint": return
    torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
    torch.utils.checkpoint.checkpoint = unsloth_gradient_checkpoint
    import transformers.modeling_utils
    transformers.modeling_utils.checkpoint = unsloth_gradient_checkpoint
    os.environ["UNSLOTH_PATCHED"] = "1"
pass


def unpatch_unsloth_gradient_checkpointing():
    import torch.utils
    if hasattr(torch.utils.checkpoint, "_old_checkpoint"):
        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint
        del torch.utils.checkpoint._old_checkpoint
    pass
pass


def unpatch_gradient_checkpointing():
    import torch.utils
    if hasattr(torch.utils.checkpoint, "_old_checkpoint"):
        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint
        del torch.utils.checkpoint._old_checkpoint
    pass
pass


from torch.utils.checkpoint import (
    _DEFAULT_DETERMINISM_MODE,
    _infer_device_type,
    _get_autocast_kwargs,
    _get_device_module,
    get_device_states,
    contextlib,
    DefaultDeviceType,
    noop_context_fn,
)
# Added [device_type] in Torch 2.5!
def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states):
        with device_module.device(device):
            device_module.set_rng_state(state)
pass

global CPU_BUFFERS
global CPU_INDEX
global GPU_BUFFERS
global GPU_BUFFERS_B
global USE_DOUBLE_BUFFER
global BACKWARD_PASS
global EXTRA_STREAMS
global MAIN_STREAMS
global MINIMUM_SIZE
global USE_UNSLOTH_GC
global LAST_GC_INDEX
global FIRST_PASS
global CURRENT_GC_INDEX
global BUFFER_EVENTS_A
global BUFFER_EVENTS_B
global NEXT_BUFFER_SLOT

if DEVICE_TYPE in ("cuda", "hip"):
    torch_gpu_stream = torch.cuda.stream
elif DEVICE_TYPE == "xpu":
    torch_gpu_stream = torch.xpu.stream

CPU_BUFFERS = []
CPU_INDEX = None


def _gc_disable_cpu_offload():
    value = os.environ.get("UNSLOTH_GC_DISABLE_CPU_OFFLOAD", None)
    if value is None:
        return False
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


_pinned_bytes_allocated: int = 0
_cpu_ram_warned: bool = False

def _check_cpu_ram_before_pin(alloc_bytes: int):
    """Check CPU RAM availability before pinned allocation.

    Pinned memory (cudaHostAlloc) locks physical pages and cannot be swapped.
    Exhaustion produces cryptic CUDA errors rather than clean Python MemoryError.
    This checks once per new allocation and warns with actionable context.
    """
    global _cpu_ram_warned, _pinned_bytes_allocated
    if _cpu_ram_warned:
        return
    try:
        import psutil
        mem = psutil.virtual_memory()
    except ImportError:
        return

    avail = mem.available
    total = mem.total
    pct_used = mem.percent
    new_total_pinned = _pinned_bytes_allocated + alloc_bytes

    # Warn if: >85% used AND this alloc would consume >50% of remaining,
    # or >95% used regardless of alloc size
    headroom_ratio = alloc_bytes / avail if avail > 0 else float("inf")
    critical = pct_used > 95 or (pct_used > 85 and headroom_ratio > 0.5)
    if not critical:
        return

    _cpu_ram_warned = True
    warnings.warn(
        f"\nUnsloth: CPU RAM is {pct_used:.0f}% used "
        f"({avail / 2**30:.1f}GB free / {total / 2**30:.1f}GB total). "
        f"Pinned memory allocated by Unsloth so far: {_pinned_bytes_allocated / 2**30:.2f}GB. "
        f"Next allocation: {alloc_bytes / 2**20:.0f}MB. "
        f"Projected pinned memory after this allocation: {new_total_pinned / 2**30:.2f}GB. "
        "Pinned memory cannot be swapped and exhaustion causes cryptic CUDA errors. "
        "To disable CPU offloading: set UNSLOTH_GC_DISABLE_CPU_OFFLOAD=1",
        stacklevel=4,
    )


def _track_pinned_alloc(numel: int, dtype: torch.dtype):
    """Track bytes allocated as pinned memory."""
    global _pinned_bytes_allocated
    n_bytes = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else dtype.itemsize
    alloc_bytes = numel * n_bytes
    _check_cpu_ram_before_pin(alloc_bytes)
    _pinned_bytes_allocated += alloc_bytes


def _unwrap_dtensor(tensor):
    """Unwrap FSDP2 DTensor to its local shard.

    DTensor overrides .copy_(), .to(), etc. to coordinate across ranks.
    For CPU offloading we only need the raw local bytes on this GPU,
    so we extract ._local_tensor to avoid triggering distributed dispatch.
    Plain tensors pass through unchanged.
    """
    if hasattr(tensor, "_local_tensor"):
        return tensor._local_tensor
    return tensor


class UnslothGradientCheckpointer:
    """
    All Unsloth Zoo code licensed under LGPLv3

    Lightweight state holder for stream checkpointing metadata.

    The supported release paths are:
    - original Unsloth reentrant checkpointing, implemented by
      ``UnslothCheckpointFunction`` and the module-level CPU/GPU buffers below;
    - Unsloth stream checkpointing, implemented by
      ``UnslothStreamActivationScheduler`` and ``_UnslothActivationSaveScope``.

    Older class-local saved-tensors pack/unpack implementations were removed
    from the release surface. This class remains only because downstream code
    imports it and the stream scheduler uses ``_minimum_size`` as its staging
    threshold.
    """
    _initialized: bool = False
    _minimum_size: int = 2 * 1024 * 1024 // 2
    _dtype: torch.dtype = None
    _meta_initialized: bool = False

    @classmethod
    def ensure_metadata(cls, dtype: torch.dtype = None):
        if dtype is None:
            if cls._dtype is not None:
                dtype = cls._dtype
            elif DEVICE_TYPE == "cuda":
                major_version, _ = torch.cuda.get_device_capability()
                dtype = torch.bfloat16 if (major_version >= 8) else torch.float16
            else:
                dtype = torch.bfloat16
        cls._dtype = dtype
        n_bytes = torch.finfo(dtype).bits // 8
        # Offload threshold in MiB (min tensor size to qualify). Default 2 MiB.
        # Raising this reduces pack/unpack call count, trading memory savings
        # for throughput (fewer Python + CUDA driver calls per step).
        try:
            _min_mib = float(os.environ.get("UNSLOTH_GC_MIN_OFFLOAD_MB", "2"))
        except ValueError:
            _min_mib = 2.0
        cls._minimum_size = int(_min_mib * 1024 * 1024) // n_bytes
        cls._meta_initialized = True

    @classmethod
    def initialize(cls, dtype: torch.dtype = None, num_devices: int = None):
        cls.ensure_metadata(dtype)
        cls._initialized = True

    @classmethod
    def reset_for_new_training(cls):
        cls.ensure_metadata(cls._dtype)

    @classmethod
    def cleanup(cls):
        cls._initialized = False

    @classmethod
    def begin_checkpoint(cls, dtype=None):
        """Compatibility hook used by the checkpoint dispatcher."""
        if not cls._initialized:
            cls.initialize(dtype)
        else:
            cls.ensure_metadata(dtype)
        return None


class _UnslothActivationTicket:
    """Saved-tensor token for Unsloth's activation stash."""

    __slots__ = ("scheduler", "tag")

    def __init__(self, scheduler, tag):
        self.scheduler = scheduler
        self.tag = tag


class _UnslothActivationBoundary(torch.autograd.Function):
    """Autograd marker that seals and later rewinds one activation region."""

    @staticmethod
    def forward(ctx, tensor, scheduler):
        scheduler.seal_forward_region()
        ctx.scheduler = scheduler
        if getattr(scheduler, "clone_boundary_output", False):
            return tensor.clone()
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        ctx.scheduler.rewind_for_backward_region()
        return grad_output, None


def _get_unique_tensor_key(tensor):
    try:
        return (tensor.untyped_storage().data_ptr() + tensor.storage_offset(), tensor.dtype)
    except Exception:
        return (id(tensor), getattr(tensor, "dtype", None))


class _UnslothHostActivation:
    """Pinned host slot for one stashed activation tensor."""

    __slots__ = (
        "host_tensor",
        "device",
        "device_tensor",
        "host_ready_event",
        "device_ready_event",
    )

    def __init__(self, host_tensor, device, host_ready_event=None):
        self.host_tensor = host_tensor
        self.device = device
        self.device_tensor = None
        self.host_ready_event = host_ready_event
        self.device_ready_event = None


class UnslothStreamActivationScheduler:
    """Layer-window scheduler with direct pinned-host activation staging."""

    def __init__(self, num_host_windows, num_layer_windows):
        self.num_host_windows = max(0, int(num_host_windows))
        self.num_layer_windows = max(1, int(num_layer_windows))
        self.parameter_storage = set()
        self._parameter_storage_cache = {}
        self._d2h_streams = {}
        self._h2d_streams = {}
        self.prefetch_targets = ()
        self.clone_boundary_layers = set()
        self.clone_boundary_output = False
        self.reset_step_windows()

        self.forward_window_map = {}
        constant = 0
        for i in range(self.num_host_windows):
            self.forward_window_map[i] = ((self.num_layer_windows // self.num_host_windows) * (i + 1)) - 1
            if i < (self.num_layer_windows % self.num_host_windows):
                self.forward_window_map[i] += i + 1
                constant = i + 1
            else:
                self.forward_window_map[i] += constant

    def reset_step_windows(self):
        self.current_window = 0
        self.tensor_count_current_window = 0
        self.ticket_state = {}
        self.device_hold_refs = {}
        self.host_windows = {}
        self.active_windows = set()
        self.staged_window_count = 0
        self.prefetch_targets_started = False

    def update_model_parameters(self, module):
        cache_key = id(module)
        cached = self._parameter_storage_cache.get(cache_key)
        if cached is not None:
            self.parameter_storage = cached
            return

        storage = set()
        try:
            params = module.parameters()
        except Exception:
            params = ()
        for param in params:
            try:
                tensor = _unwrap_dtensor(param.data)
                storage.add(tensor.untyped_storage().data_ptr())
            except Exception:
                pass
        self.parameter_storage = storage
        self._parameter_storage_cache[cache_key] = storage

    def _cuda_stream(self, cache, device):
        if DEVICE_TYPE not in ("cuda", "hip"):
            return None
        if getattr(device, "type", None) not in ("cuda", "hip"):
            return None
        try:
            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            stream = cache.get(device_index)
            if stream is None:
                with torch.cuda.device(device_index):
                    stream = torch.cuda.Stream()
                cache[device_index] = stream
            return stream
        except Exception:
            return None

    def _d2h_stream(self, device):
        return self._cuda_stream(self._d2h_streams, device)

    def _h2d_stream(self, device):
        return self._cuda_stream(self._h2d_streams, device)

    def should_stage(self, tensor: torch.Tensor) -> bool:
        if _gc_disable_cpu_offload():
            return False
        tensor = _unwrap_dtensor(tensor)
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            if torch.compiler.is_compiling():
                return False
        if not torch.is_tensor(tensor):
            return False
        if getattr(tensor, "is_meta", False):
            return False
        if tensor.numel() < UnslothGradientCheckpointer._minimum_size:
            return False
        if tensor.device.type == "cpu":
            return False
        if tensor.layout != torch.strided:
            return False
        if (not tensor.is_contiguous()) or (tensor.storage_offset() != 0):
            return False
        try:
            if tensor.untyped_storage().data_ptr() in self.parameter_storage:
                return False
        except Exception:
            pass
        return True

    def stash_tensor(self, tensor: torch.Tensor):
        if not self.should_stage(tensor):
            return tensor
        try:
            max_tensors_per_window = int(os.environ.get("UNSLOTH_GC_STREAM_MAX_TENSORS_PER_WINDOW", "1"))
        except ValueError:
            max_tensors_per_window = 1
        if max_tensors_per_window > 0 and self.tensor_count_current_window >= max_tensors_per_window:
            return tensor
        self.active_windows.add(self.current_window)
        tensor_tag = (self.current_window, self.tensor_count_current_window)
        self.tensor_count_current_window += 1
        self.ticket_state[tensor_tag] = tensor
        if self.current_window < self.num_host_windows:
            self.device_hold_refs[tensor_tag] = tensor
        return _UnslothActivationTicket(self, tensor_tag)

    def _new_host_tensor(self, tensor):
        size = tuple(tensor.size())
        stride = tuple(tensor.stride())
        try:
            return torch.empty_strided(
                size,
                stride,
                dtype=tensor.dtype,
                layout=tensor.layout,
                device="cpu",
                pin_memory=True,
            )
        except TypeError:
            host = torch.empty_strided(
                size,
                stride,
                dtype=tensor.dtype,
                layout=tensor.layout,
                device="cpu",
            )
            return host.pin_memory()

    def _copy_to_host(self, tensor):
        device = tensor.device
        stream = self._d2h_stream(device)
        event = None
        with torch.no_grad():
            host = self._new_host_tensor(tensor)
            if stream is None:
                host.copy_(tensor, non_blocking=False)
            else:
                current_stream = torch.cuda.current_stream(device)
                stream.wait_stream(current_stream)
                with torch.cuda.stream(stream):
                    host.copy_(tensor, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(stream)
        return _UnslothHostActivation(host, device, event)

    def _wait_host_ready(self, slot):
        event = getattr(slot, "host_ready_event", None)
        if event is None:
            return
        if DEVICE_TYPE not in ("cuda", "hip"):
            return
        try:
            torch.cuda.current_stream(slot.device).wait_event(event)
        except Exception:
            pass

    def _restore_slot_to_device(self, slot):
        if slot.device_tensor is not None:
            return False
        stream = self._h2d_stream(slot.device)
        with torch.no_grad():
            if stream is None:
                if slot.host_tensor is None:
                    return False
                slot.device_tensor = slot.host_tensor.to(slot.device, non_blocking=False)
                return True
            current_stream = torch.cuda.current_stream(slot.device)
            stream.wait_stream(current_stream)
            if slot.host_ready_event is not None:
                try:
                    stream.wait_event(slot.host_ready_event)
                except Exception:
                    pass
            with torch.cuda.stream(stream):
                if slot.host_tensor is None:
                    return False
                restored = torch.empty_strided(
                    tuple(slot.host_tensor.size()),
                    tuple(slot.host_tensor.stride()),
                    dtype=slot.host_tensor.dtype,
                    layout=slot.host_tensor.layout,
                    device=slot.device,
                )
                restored.copy_(slot.host_tensor, non_blocking=True)
                slot.device_tensor = restored
                slot.device_ready_event = torch.cuda.Event()
                slot.device_ready_event.record(stream)
        return True

    def _wait_device_ready(self, slot):
        event = getattr(slot, "device_ready_event", None)
        if event is None or DEVICE_TYPE not in ("cuda", "hip"):
            return
        try:
            torch.cuda.current_stream(slot.device).wait_event(event)
        except Exception:
            pass

    def _view_slot(self, slot, shape, stride, requires_grad, dtype):
        if slot.device_tensor is None:
            self._restore_slot_to_device(slot)
        self._wait_device_ready(slot)
        result = slot.device_tensor
        if result is None:
            raise RuntimeError("Unsloth: staged activation was requested before device restore completed")
        if tuple(result.shape) != tuple(shape):
            result = result.view(shape)
        if tuple(result.stride()) != tuple(stride):
            result = result.as_strided(shape, stride)
        if result.dtype != dtype:
            result = result.to(dtype)
        return result

    def restore_tensor(self, token):
        if not isinstance(token, _UnslothActivationTicket) or token.scheduler is not self:
            return token
        state = self.ticket_state.pop(token.tag)
        self.device_hold_refs.pop(token.tag, None)
        if isinstance(state, tuple) and len(state) == 6 and state[0] == "slot":
            _, slot, shape, stride, requires_grad, dtype = state
            return self._view_slot(slot, shape, stride, requires_grad, dtype)
        return state

    def stage_window_to_host(self, window_index):
        if window_index in self.host_windows:
            return
        staged = {}
        for tensor_tag, state in list(self.ticket_state.items()):
            group_id, _ = tensor_tag
            if group_id != window_index or not torch.is_tensor(state):
                continue
            key = _get_unique_tensor_key(state)
            slot = staged.get(key)
            if slot is None:
                slot = self._copy_to_host(state)
                staged[key] = slot
            self.ticket_state[tensor_tag] = (
                "slot",
                slot,
                state.shape,
                state.stride(),
                state.requires_grad,
                state.dtype,
            )
        self.host_windows[window_index] = staged

    def advance_forward_window(self, current_window):
        if self.num_host_windows <= 0 or self.staged_window_count >= self.num_host_windows:
            return
        if current_window == 0:
            self.stage_window_to_host(current_window)
        if self.forward_window_map.get(self.staged_window_count) == current_window:
            for slot in self.host_windows.get(self.staged_window_count, {}).values():
                self._wait_host_ready(slot)
            for tensor_tag in list(self.device_hold_refs.keys()):
                if tensor_tag[0] == self.staged_window_count:
                    self.device_hold_refs[tensor_tag] = None
            if self.staged_window_count < (self.num_host_windows - 1):
                self.stage_window_to_host(self.staged_window_count + 1)
            self.staged_window_count += 1

    def restore_window_to_device(self, window_index):
        mapping = self.host_windows.pop(window_index, None)
        if not mapping:
            return
        for slot in mapping.values():
            self._restore_slot_to_device(slot)

    def prefetch_host_windows_to_device(self):
        if self.num_host_windows <= 0:
            return
        for mapping in list(self.host_windows.values()):
            for slot in mapping.values():
                self._restore_slot_to_device(slot)

    def prefetch_linked_stacks(self):
        if self.prefetch_targets_started:
            return
        self.prefetch_targets_started = True
        for scheduler in self.prefetch_targets:
            if scheduler is not self:
                scheduler.prefetch_host_windows_to_device()

    def seal_forward_region(self):
        if self.current_window not in self.active_windows:
            return
        self.active_windows.discard(self.current_window)
        self.advance_forward_window(self.current_window)
        self.current_window += 1
        self.tensor_count_current_window = 0

    def rewind_for_backward_region(self):
        self.prefetch_linked_stacks()
        self.current_window -= 1
        if self.current_window < 0:
            self.current_window = 0
            return
        if self.num_host_windows <= 0 or self.staged_window_count <= 0:
            return
        idx = self.staged_window_count - 1
        if self.forward_window_map.get(idx) == self.current_window:
            self.restore_window_to_device(idx)
            self.staged_window_count -= 1 if self.staged_window_count > 1 else 0
        if self.current_window == 0:
            self.host_windows.clear()
            self.staged_window_count = 0


class _UnslothActivationSaveScope(torch.autograd.graph.saved_tensors_hooks):
    """Saved-tensor hooks that route tensors through an Unsloth stash."""

    def __init__(self, scheduler, *, enabled=True):
        self._enabled = enabled and not _gc_disable_cpu_offload()
        self._scheduler = scheduler
        super().__init__(self._pack_hook, self._unpack_hook)

    def __enter__(self):
        if not self._enabled:
            return self
        return super().__enter__()

    def __exit__(self, *args):
        if not self._enabled:
            return
        super().__exit__(*args)

    def _pack_hook(self, tensor):
        if not self._enabled:
            return tensor
        return self._scheduler.stash_tensor(tensor)

    def _unpack_hook(self, packed):
        if not self._enabled:
            return packed
        return self._scheduler.restore_tensor(packed)


def _attach_activation_boundary(outputs, scheduler):
    """Attach the activation boundary to the first tensor in a nested output."""
    if torch.is_tensor(outputs):
        return _UnslothActivationBoundary.apply(outputs, scheduler)
    if isinstance(outputs, tuple):
        replaced = False
        new_items = []
        for item in outputs:
            if not replaced:
                new_item = _attach_activation_boundary(item, scheduler)
                replaced = new_item is not item
                new_items.append(new_item)
            else:
                new_items.append(item)
        return tuple(new_items)
    if isinstance(outputs, list):
        replaced = False
        new_items = []
        for item in outputs:
            if not replaced:
                new_item = _attach_activation_boundary(item, scheduler)
                replaced = new_item is not item
                new_items.append(new_item)
            else:
                new_items.append(item)
        return new_items
    return outputs


def _resolve_stream_wrapper_target(model):
    inner_model = model
    while hasattr(inner_model, "model"):
        inner_model = inner_model.model
    return inner_model


def _stream_wrapper_owner_chain(model):
    owners = []
    seen = set()
    current = model
    while current is not None and id(current) not in seen:
        owners.append(current)
        seen.add(id(current))
        current = getattr(current, "model", None)
    return owners


def _delete_local_attr(obj, name):
    if name not in getattr(obj, "__dict__", {}):
        return
    try:
        delattr(obj, name)
    except AttributeError:
        pass


_STREAM_LAYER_ATTRS = ("layers", "blocks", "h")
_STREAM_TEXT_TOKENS = ("language_model", "text_model", "decoder", "transformer", "llm")
_STREAM_VISION_TOKENS = ("vision", "visual", "image", "tower")
_STREAM_AUDIO_TOKENS = ("audio", "speech", "whisper", "sound")
_STREAM_BLOCK_TOKENS = ("layer", "block", "decoder", "encoder")
_STREAM_ROLE_NAMES = ("text", "vision", "audio", "other")


def _env_enabled(name):
    return os.environ.get(name, "") in ("1", "true", "True", "yes", "YES")


def _stream_debug_enabled():
    return (
        _env_enabled("UNSLOTH_GC_STREAM_DEBUG")
        or _env_enabled("UNSLOTH_GC_PROFILE")
        or _env_enabled("UNSLOTH_GC_CHECKPOINT_HIT_COUNT")
    )


def _stream_model_type(model):
    config = getattr(model, "config", None)
    return str(getattr(config, "model_type", "") or "").lower()


def _remove_unsloth_stream_offload_wrapper(model):
    owners = _stream_wrapper_owner_chain(model)
    stream_layers = []
    for owner in owners:
        local_layers = getattr(owner, "__dict__", {}).get("_unsloth_stream_layers", None)
        if local_layers is not None:
            stream_layers.extend(local_layers)
    for layer in stream_layers:
        try:
            if hasattr(layer, "_unsloth_stream_scheduler"):
                delattr(layer, "_unsloth_stream_scheduler")
            if hasattr(layer, "_unsloth_stream_layer_idx"):
                delattr(layer, "_unsloth_stream_layer_idx")
        except Exception:
            pass
    for owner in owners:
        _delete_local_attr(owner, "_unsloth_stream_layers")
        _delete_local_attr(owner, "_unsloth_stream_scheduler")
        _delete_local_attr(owner, "_unsloth_stream_schedulers")
        _delete_local_attr(owner, "_unsloth_stream_groups")


def _is_multimodal_model(model):
    config = getattr(model, "config", None)
    model_type = _stream_model_type(model)
    if any(token in model_type for token in ("vl", "vision", "audio", "speech", "multi", "whisper", "csm")):
        return True
    if any(
        getattr(config, attr, None) is not None
        for attr in ("vision_config", "audio_config", "text_config")
    ):
        return True
    model_name = type(model).__name__.lower()
    return any(token in model_name for token in ("vision", "whisper", "audio", "speech"))


def _stream_role_from_model_family(model, name):
    model_type = _stream_model_type(model)
    lowered = str(name).lower()
    parts = tuple(part for part in lowered.split(".") if part)
    if model_type == "whisper":
        if "encoder" in parts:
            return "audio"
        if "decoder" in parts:
            return "text"
    if model_type == "csm":
        if parts and parts[0] == "codec_model":
            return "audio"
        if parts and parts[0] in ("backbone_model", "depth_decoder"):
            return "text"
    return "other"


def _stream_group_role(name):
    lowered = str(name).lower()
    if any(token in lowered for token in _STREAM_TEXT_TOKENS):
        return "text"
    if any(token in lowered for token in _STREAM_VISION_TOKENS):
        return "vision"
    if any(token in lowered for token in _STREAM_AUDIO_TOKENS):
        return "audio"
    return "other"


def _config_int(config, *names):
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _stream_role_from_config(model, layers):
    config = getattr(model, "config", None)
    if config is None:
        return "other"
    n_layers = len(layers)
    text_config = getattr(config, "text_config", None)
    vision_config = getattr(config, "vision_config", None)
    audio_config = getattr(config, "audio_config", None)

    text_depth = _config_int(
        text_config or config,
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "decoder_layers",
    )
    if text_depth is not None and n_layers == text_depth:
        return "text"

    vision_depth = _config_int(
        vision_config,
        "num_hidden_layers",
        "num_layers",
        "depth",
        "num_blocks",
    )
    if vision_depth is not None and n_layers == vision_depth:
        return "vision"

    audio_depth = _config_int(audio_config, "num_hidden_layers", "num_layers", "depth")
    if audio_depth is not None and n_layers == audio_depth:
        return "audio"

    return "other"


def _stream_role_from_layer_type(layers):
    if not layers:
        return "other"
    names = " ".join(type(layer).__name__.lower() for layer in layers[: min(len(layers), 4)])
    if any(token in names for token in ("vision", "visual", "image", "vit", "patchmerger")):
        return "vision"
    if any(token in names for token in ("audio", "speech", "whisper")):
        return "audio"
    if any(token in names for token in ("decoder", "causal", "language", "llm")):
        return "text"
    return "other"


def _infer_stream_group_role(model, name, layers):
    role = _stream_role_from_model_family(model, name)
    if role != "other":
        return role
    role = _stream_group_role(name)
    if role != "other":
        return role
    role = _stream_role_from_layer_type(layers)
    if role != "other":
        return role
    role = _stream_role_from_config(model, layers)
    if role != "other":
        return role
    if not _is_multimodal_model(model):
        return "text"
    return "other"


def _is_transformer_layer_list(layers):
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) < 2:
        return False
    for layer in layers:
        class_name = type(layer).__name__.lower()
        if any(token in class_name for token in _STREAM_BLOCK_TOKENS):
            return True
    return False


def _stream_stack_filter(model):
    raw = os.environ.get("UNSLOTH_GC_STREAM_STACKS", "auto").strip().lower()
    if raw in ("", "1", "true", "yes"):
        raw = "auto"
    requested = {part.strip() for part in raw.replace(";", ",").split(",") if part.strip()}
    if not requested or "auto" in requested or "default" in requested:
        return {"text", "modalities"} if _is_multimodal_model(model) else {"text"}
    if "all" in requested:
        return {"all"}
    aliases = {
        "language": "text",
        "language_model": "text",
        "decoder": "text",
        "visual": "vision",
        "image": "vision",
        "tower": "vision",
        "modality": "modalities",
        "multimodal": "modalities",
        "modal": "modalities",
    }
    return {aliases.get(item, item) for item in requested}


def _stream_stack_path_overrides():
    raw = os.environ.get("UNSLOTH_GC_STREAM_STACK_PATHS", "").strip()
    if not raw:
        return []
    overrides = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        role = None
        path = item
        if ":" in item:
            maybe_role, maybe_path = item.split(":", 1)
            maybe_role = maybe_role.strip().lower()
            maybe_path = maybe_path.strip()
            if maybe_role in _STREAM_ROLE_NAMES and maybe_path:
                role = maybe_role
                path = maybe_path
        overrides.append((role, path.strip()))
    return overrides


def _stream_path_matches(candidate_name, requested_path):
    candidate_name = str(candidate_name).strip(".")
    requested_path = str(requested_path).strip(".")
    if not candidate_name or not requested_path:
        return False
    return (
        candidate_name == requested_path
        or candidate_name.endswith("." + requested_path)
        or requested_path.endswith("." + candidate_name)
    )


def _log_stream_layer_groups(label, groups):
    if not _env_enabled("UNSLOTH_GC_STREAM_DEBUG"):
        return
    print(f"[unsloth_stream] {label}={len(groups)}", flush=True)
    for group in groups:
        print(
            f"[unsloth_stream] candidate role={group['role']} layers={len(group['layers'])} "
            f"order={group['order']} name={group['name'] or '<root>'} "
            f"has_checkpoint={group['has_checkpoint']}",
            flush=True,
        )


def _find_stream_layer_groups(model):
    target = _resolve_stream_wrapper_target(model)
    candidates = []
    order_counter = [0]

    def add_candidate(name, layers):
        if not _is_transformer_layer_list(layers):
            return
        has_checkpoint = any(hasattr(layer, "_gradient_checkpointing_func") for layer in layers)
        role = _infer_stream_group_role(model, name, layers)
        score = 0
        if has_checkpoint:
            score += 1000
        if role == "text":
            score += 100
        if str(name).lower() in ("", "model", "base_model", "model.model"):
            score += 50
        if role in ("vision", "audio"):
            score += 300 if _is_multimodal_model(model) else -100
        score += min(len(layers), 99)
        candidates.append({
            "name": str(name),
            "role": role,
            "score": score,
            "order": order_counter[0],
            "has_checkpoint": has_checkpoint,
            "layers": list(layers),
        })
        order_counter[0] += 1

    for owner_index, owner in enumerate(_stream_wrapper_owner_chain(model)):
        for attr in _STREAM_LAYER_ATTRS:
            add_candidate(f"owner_{owner_index}.{attr}", getattr(owner, attr, None))
    for attr in _STREAM_LAYER_ATTRS:
        add_candidate(attr if attr != "layers" else "", getattr(target, attr, None))

    for name, module in target.named_modules():
        for attr in _STREAM_LAYER_ATTRS:
            child_path = attr if not name else f"{name}.{attr}"
            add_candidate(child_path, getattr(module, attr, None))
        for child_name, child in module.named_children():
            if isinstance(child, torch.nn.ModuleList):
                child_path = child_name if not name else f"{name}.{child_name}"
                add_candidate(child_path, child)

    unique = {}
    for candidate in candidates:
        layer_ids = tuple(id(layer) for layer in candidate["layers"])
        prev = unique.get(layer_ids)
        if prev is None or candidate["score"] > prev["score"]:
            unique[layer_ids] = candidate
    candidates = list(unique.values())
    if not candidates:
        return []
    _log_stream_layer_groups("candidate_groups", candidates)

    path_overrides = _stream_stack_path_overrides()
    preserve_override_order = False
    if path_overrides:
        selected = []
        used_paths = set()
        preserve_override_order = True
        for override_order, (role_override, requested_path) in enumerate(path_overrides):
            for candidate in candidates:
                if not _stream_path_matches(candidate["name"], requested_path):
                    continue
                candidate = dict(candidate)
                candidate["order"] = override_order
                if role_override is not None and candidate["role"] != role_override:
                    candidate["role"] = role_override
                selected.append(candidate)
                used_paths.add(requested_path)
                break
        if _stream_debug_enabled():
            missing = [path for _, path in path_overrides if path not in used_paths]
            if missing:
                print(
                    f"[unsloth_stream] stack path override did not match: {', '.join(missing)}",
                    flush=True,
                )
        candidates = selected
    else:
        selected_roles = _stream_stack_filter(model)
        if "all" not in selected_roles:
            filtered = []
            for candidate in candidates:
                role = candidate["role"]
                include = role in selected_roles
                include = include or ("modalities" in selected_roles and role in ("vision", "audio"))
                if include:
                    filtered.append(candidate)
            candidates = filtered

    if not candidates:
        return []
    _log_stream_layer_groups("selected_groups", candidates)
    if preserve_override_order:
        return candidates
    return sorted(candidates, key=lambda item: (item["role"] != "text", item["order"]))


def _find_layers_owner_name(model, layers):
    target = _resolve_stream_wrapper_target(model)
    layer_ids = tuple(id(layer) for layer in layers)
    for name, module in target.named_modules():
        for attr in _STREAM_LAYER_ATTRS:
            module_layers = getattr(module, attr, None)
            if not isinstance(module_layers, torch.nn.ModuleList):
                continue
            if tuple(id(layer) for layer in module_layers) == layer_ids:
                return name if attr == "layers" else f"{name}.{attr}" if name else attr
    return ""


def _install_unsloth_stream_offload_wrapper(model, dtype):
    _remove_unsloth_stream_offload_wrapper(model)
    groups = _find_stream_layer_groups(model)
    if not groups:
        if os.environ.get("UNSLOTH_GC_CHECKPOINT_HIT_COUNT", "") in ("1", "true", "True"):
            print("[unsloth_stream] no transformer layers found; falling back to no wrapper", flush=True)
        return
    if not UnslothGradientCheckpointer._initialized:
        UnslothGradientCheckpointer.initialize(dtype)
    config = getattr(model, "config", None)
    is_vlm = bool(
        getattr(config, "vision_config", None) is not None
        or "vl" in str(getattr(config, "model_type", "") or "").lower()
        or "vision" in type(model).__name__.lower()
    )
    deepstack_indexes = getattr(getattr(config, "vision_config", None), "deepstack_visual_indexes", None)
    deepstack_count = len(deepstack_indexes or []) if is_vlm else 0

    stream_layers = []
    stream_schedulers = []
    installed_groups = []
    for group in groups:
        layers = group["layers"]
        skip_default = "2"
        try:
            skip_last_groups = max(0, int(os.environ.get("UNSLOTH_GC_STREAM_SKIP_LAST_WINDOWS", skip_default)))
        except ValueError:
            skip_last_groups = int(skip_default)
        scheduler = UnslothStreamActivationScheduler(
            num_host_windows=max(0, len(layers) - skip_last_groups),
            num_layer_windows=len(layers),
        )
        scheduler.clone_boundary_layers = set(range(deepstack_count)) if group["role"] == "text" else set()
        scheduler.clone_boundary_output = False
        scheduler.disable_nonreentrant_cpu_offload = False
        stream_schedulers.append(scheduler)
        selected_owner = _find_layers_owner_name(model, layers) or group["name"]
        installed_groups.append((group, selected_owner, scheduler))
        for layer_idx, layer in enumerate(layers):
            layer._unsloth_stream_scheduler = scheduler
            layer._unsloth_stream_layer_idx = layer_idx
            stream_layers.append(layer)

    modality_schedulers = [
        scheduler
        for group, _, scheduler in installed_groups
        if group["role"] in ("vision", "audio")
    ]
    prefetch_modalities = os.environ.get("UNSLOTH_GC_STREAM_PREFETCH_MODALITIES", "1") not in ("0", "false", "False")
    if modality_schedulers and prefetch_modalities:
        for group, _, scheduler in installed_groups:
            if group["role"] == "text":
                scheduler.prefetch_targets = tuple(modality_schedulers)

    if os.environ.get("UNSLOTH_GC_CHECKPOINT_HIT_COUNT", "") in ("1", "true", "True"):
        total_layers = sum(len(group["layers"]) for group in groups)
        print(
            f"[unsloth_stream] installing layer-level stream wrapper groups={len(groups)} "
            f"total_layers={total_layers} stacks={os.environ.get('UNSLOTH_GC_STREAM_STACKS', 'auto')} "
            f"vlm={is_vlm} deepstack_count={deepstack_count}",
            flush=True,
        )
        for group, selected_owner, scheduler in installed_groups:
            print(
                f"[unsloth_stream] group role={group['role']} layers={len(group['layers'])} "
                f"host_windows={scheduler.num_host_windows} owner={selected_owner or '<unknown>'} "
                f"has_checkpoint={group['has_checkpoint']} "
                f"clone_boundary_layers={len(scheduler.clone_boundary_layers)}",
                flush=True,
            )

    model._unsloth_stream_scheduler = stream_schedulers[0]
    model._unsloth_stream_schedulers = stream_schedulers
    model._unsloth_stream_groups = [
        {"role": group["role"], "owner": selected_owner, "layers": len(group["layers"])}
        for group, selected_owner, _ in installed_groups
    ]
    model._unsloth_stream_layers = stream_layers


def initialize_unsloth_gradient_checkpointing(dtype = None):
    # All Unsloth Zoo code licensed under LGPLv3
    global CPU_BUFFERS
    global CPU_INDEX
    global GPU_BUFFERS
    global GPU_BUFFERS_B
    global USE_DOUBLE_BUFFER
    global BACKWARD_PASS
    global EXTRA_STREAMS
    global MAIN_STREAMS
    global MINIMUM_SIZE
    global USE_UNSLOTH_GC
    global LAST_GC_INDEX
    global FIRST_PASS
    global CURRENT_GC_INDEX
    global BUFFER_EVENTS_A
    global BUFFER_EVENTS_B
    global NEXT_BUFFER_SLOT
    CPU_BUFFERS = []
    CPU_INDEX = 0

    if dtype is None:
        if DEVICE_TYPE == "cuda":
            major_version, minor_version = torch.cuda.get_device_capability()
            SUPPORTS_BFLOAT16 = (major_version >= 8)
        elif DEVICE_TYPE == "hip":
            SUPPORTS_BFLOAT16 = True
        elif DEVICE_TYPE == "xpu":
            SUPPORTS_BFLOAT16 = True
        dtype = torch.bfloat16 if SUPPORTS_BFLOAT16 else torch.float16
    pass

    _track_pinned_alloc(128*1024 * 200, dtype)
    for i in range(200):
        x = torch.empty(128*1024, dtype = dtype, device = "cpu", pin_memory = True)
        CPU_BUFFERS.append(x)
    pass

    # Allocate buffers to how many GPUs
    n_gpus = torch.cuda.device_count() if DEVICE_TYPE in ("cuda", "hip") else torch.xpu.device_count()
    NEXT_BUFFER_SLOT = [0] * n_gpus
    try:
        GPU_BUFFERS = tuple([torch.empty(INITIAL_GPU_BUFFER_SIZE, dtype = dtype, device = f"{DEVICE_TYPE_TORCH}:{i}") for i in range(n_gpus)])
        # Double buffering: try to allocate buffer B (can be disabled via env var)
        if os.environ.get("UNSLOTH_DISABLE_DOUBLE_BUFFER", "0") == "1":
            GPU_BUFFERS_B = None
            USE_DOUBLE_BUFFER = False
            BUFFER_EVENTS_A = None
            BUFFER_EVENTS_B = None
        else:
            try:
                GPU_BUFFERS_B = tuple([torch.empty(INITIAL_GPU_BUFFER_SIZE, dtype = dtype, device = f"{DEVICE_TYPE_TORCH}:{i}") for i in range(n_gpus)])
                USE_DOUBLE_BUFFER = False # set false first, enabled after first pass if CUDA free memory > DOUBLE_BUFFER_HEADROOM
                # Per-buffer events to prevent race conditions in double buffering.
                # Each event tracks when compute on that buffer finishes
                if DEVICE_TYPE in ("cuda", "hip"):
                    event_ctor = torch.cuda.Event
                elif DEVICE_TYPE == "xpu":
                    event_ctor = torch.xpu.Event
                else:
                    raise RuntimeError(f"Double buffering unsupported on {DEVICE_TYPE}")
                BUFFER_EVENTS_A = tuple([event_ctor() for _ in range(n_gpus)])
                BUFFER_EVENTS_B = tuple([event_ctor() for _ in range(n_gpus)])
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                GPU_BUFFERS_B = None
                USE_DOUBLE_BUFFER = False
                BUFFER_EVENTS_A = None
                BUFFER_EVENTS_B = None
    except Exception as e:
        print("="*10 + "\n")
        print("Unsloth: Your setup does not support `PYTORCH_CUDA_ALLOC_CONF`\n")
        print("Please set `import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '';`\n")
        print("Then re-run Unsloth from the start.")
        print("="*10 + "\n")
        raise

    BACKWARD_PASS = True
    EXTRA_STREAMS = tuple([torch.cuda.Stream() if DEVICE_TYPE_TORCH == "cuda" else torch.xpu.Stream() for i in range(n_gpus)])
    if DEVICE_TYPE in ("cuda", "hip"):
        MAIN_STREAMS  = tuple([torch.cuda.default_stream(torch.device(f"cuda:{i}")) for i in range(n_gpus)])
    elif DEVICE_TYPE == "xpu":
        MAIN_STREAMS  = tuple([torch.xpu.current_stream(torch.device(f"xpu:{i}")) for i in range(n_gpus)])

    # Minimum size to enable Unsloth GC is 2MB -> 32 layers = 64MB
    n_bytes = torch.finfo(dtype).bits // 8
    MINIMUM_SIZE = 2 * 1024 * 1024 // n_bytes
    USE_UNSLOTH_GC = True

    # Disable offloading on the last layer - uses more VRAM and is slower
    # See https://github.com/pytorch/torchtune/pull/1443
    LAST_GC_INDEX = 0
    FIRST_PASS = True
    CURRENT_GC_INDEX = 0
pass


class UnslothCheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        # Check if no requires_grad in inputs
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        ctx._requires_gradient = False
        use_gpu_buffer = False
        disable_cpu_offload = _gc_disable_cpu_offload()

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):

                if i == 0 and arg.requires_grad:
                    global FIRST_PASS
                    global LAST_GC_INDEX
                    if FIRST_PASS:
                        # Save last layer index so next run we do not offload activations
                        # Saves VRAM and saves some time
                        # See https://github.com/pytorch/torchtune/pull/1443
                        LAST_GC_INDEX += 1
                    pass
                    global CURRENT_GC_INDEX
                    CURRENT_GC_INDEX += 1

                    ctx._requires_gradient = True
                    # Unwrap DTensor to local shard for plain memcpy
                    _arg = _unwrap_dtensor(arg)
                    new_size = _arg.numel()

                    global MINIMUM_SIZE
                    global CPU_INDEX
                    if (not disable_cpu_offload) and new_size > MINIMUM_SIZE and ((CURRENT_GC_INDEX != LAST_GC_INDEX) or FIRST_PASS):
                        use_gpu_buffer = True
                        global CPU_BUFFERS
                        global GPU_BUFFERS
                        global GPU_BUFFERS_B
                        global USE_DOUBLE_BUFFER
                        global BACKWARD_PASS
                        global EXTRA_STREAMS
                        global MAIN_STREAMS
                        device = _arg.device
                        device_index = device.index
                        GPU_BUFFER   = GPU_BUFFERS  [device_index]
                        MAIN_STREAM  = MAIN_STREAMS [device_index]
                        EXTRA_STREAM = EXTRA_STREAMS[device_index]

                        # Handle interrupted training runs
                        if BACKWARD_PASS:
                            BACKWARD_PASS = False
                            CPU_INDEX = 0
                            if not FIRST_PASS and not USE_DOUBLE_BUFFER and GPU_BUFFERS_B is not None:
                                try:
                                    if DEVICE_TYPE in ("cuda", "hip"):
                                        free_mem, _ = torch.cuda.mem_get_info(device_index)
                                    elif DEVICE_TYPE == "xpu":
                                        free_mem, _ = torch.xpu.mem_get_info(device_index)
                                    else:
                                        free_mem = 0
                                except Exception as e:
                                    free_mem = 0
                                if free_mem > DOUBLE_BUFFER_HEADROOM:
                                    USE_DOUBLE_BUFFER = True
                                    print(f"Unsloth: Double buffering enabled (parallel H2D + compute) for backward pass.")
                                else:
                                    for j in range(len(GPU_BUFFERS_B)):
                                        GPU_BUFFERS_B[j].resize_(0)
                                    GPU_BUFFERS_B = None
                        pass

                        # Extend buffer size
                        if CPU_INDEX >= len(CPU_BUFFERS):
                            _track_pinned_alloc(new_size, _arg.dtype)
                            x = torch.empty(new_size, dtype = _arg.dtype, device = "cpu", pin_memory = True)
                            CPU_BUFFERS.append(x)
                        pass

                        x = CPU_BUFFERS[CPU_INDEX]
                        shape = _arg.shape
                        if new_size > x.numel(): x.resize_(new_size)
                        if new_size > GPU_BUFFER.numel():
                            try:
                                GPU_BUFFER.resize_(new_size)
                            except RuntimeError as e:
                                if "out of memory" not in str(e).lower():
                                    raise
                                # clear Buffer B and try to resize Single Buffer
                                if GPU_BUFFERS_B is not None:
                                    USE_DOUBLE_BUFFER = False
                                    for j in range(len(GPU_BUFFERS_B)):
                                        GPU_BUFFERS_B[j].resize_(0)
                                    GPU_BUFFERS_B = None
                                    print("Unsloth: Disabled double buffering due to insufficient VRAM.")
                                    GPU_BUFFER.resize_(new_size)
                                else:
                                    raise
                        # resize buffer B as needed if double buffering is enabled, disable and free Buffer B if OOM
                        if USE_DOUBLE_BUFFER:
                            GPU_BUFFER_B = GPU_BUFFERS_B[device_index]
                            if new_size > GPU_BUFFER_B.numel():
                                try:
                                    GPU_BUFFER_B.resize_(new_size)
                                except RuntimeError as e:
                                    if "out of memory" not in str(e).lower():
                                        raise
                                    # OOM - disable double buffering
                                    USE_DOUBLE_BUFFER = False
                                    # Reclaim buffer B
                                    for j in range(len(GPU_BUFFERS_B)):
                                        GPU_BUFFERS_B[j].resize_(0)
                                    GPU_BUFFERS_B = None
                                    print("Unsloth: Disabled double buffering due to insufficient VRAM.")
                        x = x[:new_size].view(shape)

                        # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
                        EXTRA_STREAM.wait_stream(MAIN_STREAM)
                        with torch_gpu_stream(EXTRA_STREAM):
                            x.copy_(_arg, non_blocking = True)

                        global NEXT_BUFFER_SLOT
                        buffer_slot = NEXT_BUFFER_SLOT[device_index]
                        NEXT_BUFFER_SLOT[device_index] ^= 1
                        ctx._saved_metadata = (new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM, buffer_slot,)
                        CPU_INDEX += 1
                        tensor_inputs.append(None)

                        global USE_UNSLOTH_GC
                        if USE_UNSLOTH_GC:
                            print("Unsloth: Will smartly offload gradients to save VRAM!")
                            USE_UNSLOTH_GC = False
                    else:
                        ctx._saved_metadata = (None, None, None, None, None, None, None,)
                        tensor_inputs.append(arg)
                    pass
                else:
                    tensor_inputs.append(arg)
                pass
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
            pass
        pass
        if ctx._requires_gradient: ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)

        if use_gpu_buffer: MAIN_STREAM.wait_stream(EXTRA_STREAM)
        return outputs
    pass


    @staticmethod
    def backward(ctx, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        if not ctx._requires_gradient: return None

        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "When use_reentrant=True, torch.utils.checkpoint is incompatible"
                " with .grad() or passing an `inputs` parameter to .backward()."
                " To resolve this error, you can either set use_reentrant=False,"
                " or call .backward() without passing the `inputs` argument."
            )

        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM, buffer_slot = ctx._saved_metadata
        if CPU_INDEX is not None:
            global GPU_BUFFERS
            global USE_DOUBLE_BUFFER
            global GPU_BUFFERS_B
            global BUFFER_EVENTS_A
            global BUFFER_EVENTS_B
            # Select which buffer to use based on per-device buffer_slot
            if USE_DOUBLE_BUFFER and buffer_slot == 1:
                buffer = GPU_BUFFERS_B[device_index][:new_size].view(shape)
            else:
                buffer = GPU_BUFFERS[device_index][:new_size].view(shape)
            x = CPU_BUFFERS[CPU_INDEX][:new_size].view(shape)

            # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
            if USE_DOUBLE_BUFFER:
                # Wait for the last compute on THIS SPECIFIC buffer to finish
                event_buffer = BUFFER_EVENTS_B if buffer_slot == 1 else BUFFER_EVENTS_A
                EXTRA_STREAM.wait_event(event_buffer[device_index])
            else:
                # Single buffer mode: Must wait for MAIN_STREAM to finish
                EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                buffer.copy_(x, non_blocking = True)
        else:
            # No GPU buffer seen
            if len(tensor_indices) != 0:
                inputs[tensor_indices[0]] = tensors[0]
        pass

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices[1:], start = 1):
            inputs[idx] = tensors[i]
        pass

        global BACKWARD_PASS
        BACKWARD_PASS = True
        global FIRST_PASS
        FIRST_PASS = False
        global CURRENT_GC_INDEX
        CURRENT_GC_INDEX = 0

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()

            detached_inputs = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    detached_inputs.append(inp)
                    continue
                x = inp.detach()
                x.requires_grad = inp.requires_grad
                detached_inputs.append(x)
            pass

            # Wait for GPU buffer to finish
            if CPU_INDEX is not None:
                MAIN_STREAM.wait_stream(EXTRA_STREAM)
                x = buffer.detach()
                x.requires_grad_(True)
                detached_inputs[0] = x
            pass

            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):  # type: ignore[attr-defined]
                outputs = ctx.run_function(*detached_inputs)
            pass
        pass

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        pass

        if len(outputs_with_grad) == 0:
            # raise RuntimeError(
            #     "none of output has requires_grad=True,"
            #     " this checkpoint() is not necessary"
            # )
            pass
        else:
            torch.autograd.backward(outputs_with_grad, args_with_grad)
        pass

        if CPU_INDEX is not None and USE_DOUBLE_BUFFER:
            # Record event after compute finishes so the copy stream knows
            event_buffer = BUFFER_EVENTS_B if buffer_slot == 1 else BUFFER_EVENTS_A
            event_buffer[device_index].record(MAIN_STREAM)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )
        # Clear all memory
        for i in range(len(detached_inputs)):
            detached_inputs[i] = None
            inputs[i] = None
        pass

        return (None, None) + grads
    pass
pass


@torch._disable_dynamo
def _unsloth_checkpoint_reentrant(function, *args, preserve_rng_state=True):
    return UnslothCheckpointFunction.apply(function, preserve_rng_state, *args)

def _resolve_unsloth_scheduler_from_checkpoint_function(function, attr_name):
    bound_module = getattr(function, "__self__", None)
    scheduler = getattr(bound_module, attr_name, None)
    if scheduler is not None:
        return scheduler, bound_module
    inner = getattr(function, "func", None)
    if inner is not None and inner is not function:
        inner_scheduler, inner_module = _resolve_unsloth_scheduler_from_checkpoint_function(inner, attr_name)
        if inner_scheduler is not None:
            return inner_scheduler, inner_module
    for arg in getattr(function, "args", ()) or ():
        scheduler = getattr(arg, attr_name, None)
        if scheduler is not None:
            return scheduler, arg
    return None, bound_module


def _resolve_unsloth_stream_scheduler_from_checkpoint_function(function):
    return _resolve_unsloth_scheduler_from_checkpoint_function(function, "_unsloth_stream_scheduler")


def _unsloth_checkpoint_nonreentrant(function, *args, **kwargs):
    """Non-reentrant checkpoint using native PyTorch checkpoint plus scoped input offload."""
    preserve = kwargs.pop("preserve_rng_state", True)
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    offload_backend_arg = kwargs.pop("offload_backend", None)
    if offload_backend_arg is None:
        offload_backend_arg = os.environ.get("UNSLOTH_GC_OFFLOAD_BACKEND", "unsloth_stream")
    offload_backend = resolve_gc_offload_backend(offload_backend_arg)
    determinism_check = kwargs.pop("determinism_check", _DEFAULT_DETERMINISM_MODE)
    debug = kwargs.pop("debug", False)

    cls = UnslothGradientCheckpointer
    dtype = None
    first_arg = args[0] if args else None
    if torch.is_tensor(first_arg):
        dtype = first_arg.dtype

    cls.begin_checkpoint(dtype)
    old_checkpoint = getattr(torch.utils.checkpoint, "_old_checkpoint", None)
    original_checkpoint = old_checkpoint or torch.utils.checkpoint.checkpoint

    if (not _gc_disable_cpu_offload()) and offload_backend == "unsloth_stream":
        scheduler, bound_module = _resolve_unsloth_stream_scheduler_from_checkpoint_function(function)
        if scheduler is not None and not getattr(scheduler, "disable_nonreentrant_cpu_offload", False):
            if bound_module is not None:
                scheduler.update_model_parameters(bound_module)
                scheduler.clone_boundary_output = (
                    getattr(bound_module, "_unsloth_stream_layer_idx", -1)
                    in getattr(scheduler, "clone_boundary_layers", set())
                )
            checkpoint_group = scheduler.current_window
            with _UnslothActivationSaveScope(scheduler):
                outputs = original_checkpoint(
                    function, *args,
                    use_reentrant=False,
                    preserve_rng_state=preserve,
                    context_fn=context_fn,
                    determinism_check=determinism_check,
                    debug=debug,
                    **kwargs
                )
            if checkpoint_group in scheduler.active_windows:
                return _attach_activation_boundary(outputs, scheduler)
            return outputs

    return original_checkpoint(
        function, *args,
        use_reentrant=False,
        preserve_rng_state=preserve,
        context_fn=context_fn,
        determinism_check=determinism_check,
        debug=debug,
        **kwargs
    )


def _unsloth_checkpoint_reentrant_offload(function, *args, preserve_rng_state=True, offload_backend=None, **kwargs):
    """Reentrant PyTorch checkpoint with Unsloth stream activation offload."""
    offload_backend = resolve_gc_offload_backend(offload_backend)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    cls = UnslothGradientCheckpointer
    dtype = None
    first_arg = args[0] if args else None
    if torch.is_tensor(first_arg):
        dtype = first_arg.dtype
    if not cls._initialized:
        cls.initialize(dtype)
    else:
        cls.ensure_metadata(dtype)
    cls.begin_checkpoint(dtype)

    old_checkpoint = getattr(torch.utils.checkpoint, "_old_checkpoint", None)
    original_checkpoint = old_checkpoint or torch.utils.checkpoint.checkpoint

    if _gc_disable_cpu_offload() or offload_backend != "unsloth_stream":
        return original_checkpoint(
            function, *args,
            use_reentrant=True,
            preserve_rng_state=preserve_rng_state,
        )

    scheduler, bound_module = _resolve_unsloth_stream_scheduler_from_checkpoint_function(function)
    if scheduler is not None:
        if bound_module is not None:
            scheduler.update_model_parameters(bound_module)
            scheduler.clone_boundary_output = (
                getattr(bound_module, "_unsloth_stream_layer_idx", -1)
                in getattr(scheduler, "clone_boundary_layers", set())
            )
        checkpoint_group = scheduler.current_window
        with _UnslothActivationSaveScope(scheduler):
            outputs = original_checkpoint(
                function, *args,
                use_reentrant=True,
                preserve_rng_state=preserve_rng_state,
            )
        if checkpoint_group in scheduler.active_windows:
            return _attach_activation_boundary(outputs, scheduler)
        return outputs

    return original_checkpoint(
        function, *args,
        use_reentrant=True,
        preserve_rng_state=preserve_rng_state,
    )


def unsloth_checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    **kwargs
):
    """Unsloth gradient checkpoint: dispatches to reentrant or non-reentrant.

    No @torch._disable_dynamo -- this is a pure dispatcher so it does not
    create graph breaks for the non-reentrant path.
    """
    if use_reentrant is None:
        use_reentrant = True

    # Auto-force non-reentrant when FSDP2 is the underlying module with real
    # sharding (ws>1). Reentrant UnslothCheckpointFunction holds unshard state
    # across layers and defeats FSDP2 param sharding at backward; the stream
    # non-reentrant path recovers sharding and keeps CPU offload.
    # Override with UNSLOTH_SMART_GC_FSDP2 = "off" / "disable" to keep the
    # auto-override OFF (i.e. stay on whatever use_reentrant was passed).
    if use_reentrant:
        _fsdp2_env = os.environ.get("UNSLOTH_SMART_GC_FSDP2", "auto").strip().lower()
        if _fsdp2_env not in ("off", "disable", "0", "false", "no"):
            if _is_fsdp2_module(function):
                use_reentrant = False

    if use_reentrant:
        preserve = kwargs.pop("preserve_rng_state", True)
        offload_backend = resolve_gc_offload_backend(
            kwargs.pop("offload_backend", None)
        )
        if offload_backend == "unsloth_stream":
            return _unsloth_checkpoint_reentrant_offload(
                function, *args,
                preserve_rng_state=preserve,
                offload_backend=offload_backend,
                **kwargs,
            )
        if kwargs:
            raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
        return _unsloth_checkpoint_reentrant(function, *args, preserve_rng_state=preserve)

    return _unsloth_checkpoint_nonreentrant(function, *args, **kwargs)


def patch_unsloth_smart_gradient_checkpointing(dtype = None, use_reentrant = None, offload_backend = None):
    # All Unsloth Zoo code licensed under LGPLv3
    _maybe_patch_gc_profile()
    effective_use_reentrant = bool(use_reentrant) if use_reentrant is not None else True
    try:
        reentrant_backend = resolve_gc_offload_backend(offload_backend)
    except Exception:
        reentrant_backend = "unsloth_original"
    use_reentrant_offload_backend = (
        effective_use_reentrant
        and reentrant_backend == "unsloth_stream"
    )

    if effective_use_reentrant and not use_reentrant_offload_backend:
        UnslothGradientCheckpointer.cleanup()
        if torch.utils.checkpoint.CheckpointFunction.__name__ != "UnslothCheckpointFunction":
            initialize_unsloth_gradient_checkpointing(dtype)
            torch.utils.checkpoint._old_CheckpointFunction = torch.utils.checkpoint.CheckpointFunction
            torch.utils.checkpoint.CheckpointFunction = UnslothCheckpointFunction
    else:
        UnslothGradientCheckpointer.initialize(dtype)
        if (torch.utils.checkpoint.CheckpointFunction.__name__ == "UnslothCheckpointFunction") and \
            hasattr(torch.utils.checkpoint, "_old_CheckpointFunction"):
            torch.utils.checkpoint.CheckpointFunction = torch.utils.checkpoint._old_CheckpointFunction
            del torch.utils.checkpoint._old_CheckpointFunction

    if torch.utils.checkpoint.checkpoint.__name__ != "unsloth_checkpoint":
        torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
        torch.utils.checkpoint.checkpoint = unsloth_checkpoint

    try:
        import transformers.modeling_utils
        if hasattr(transformers.modeling_utils, "checkpoint") and \
            transformers.modeling_utils.checkpoint.__name__ != "unsloth_checkpoint":
            transformers.modeling_utils._old_checkpoint = transformers.modeling_utils.checkpoint
            transformers.modeling_utils.checkpoint = unsloth_checkpoint
    except Exception:
        pass
pass


def unpatch_unsloth_smart_gradient_checkpointing():
    # All Unsloth Zoo code licensed under LGPLv3
    UnslothGradientCheckpointer.cleanup()

    if (torch.utils.checkpoint.CheckpointFunction.__name__ == "UnslothCheckpointFunction") and \
        hasattr(torch.utils.checkpoint, "_old_CheckpointFunction"):

        torch.utils.checkpoint.CheckpointFunction = torch.utils.checkpoint._old_CheckpointFunction
        del torch.utils.checkpoint._old_CheckpointFunction
        global CPU_BUFFERS
        global GPU_BUFFERS
        global GPU_BUFFERS_B
        global USE_DOUBLE_BUFFER
        global BUFFER_EVENTS_A
        global BUFFER_EVENTS_B
        global NEXT_BUFFER_SLOT
        for i in range(len(CPU_BUFFERS)):
            if hasattr(CPU_BUFFERS[i], "resize_"): CPU_BUFFERS[i].resize_(0)
            if type(CPU_BUFFERS) is list: CPU_BUFFERS[i] = None
        for i in range(len(GPU_BUFFERS)):
            if hasattr(GPU_BUFFERS[i], "resize_"): GPU_BUFFERS[i].resize_(0)
            if type(GPU_BUFFERS) is list: GPU_BUFFERS[i] = None
        if GPU_BUFFERS_B is not None:
            for i in range(len(GPU_BUFFERS_B)):
                if hasattr(GPU_BUFFERS_B[i], "resize_"): GPU_BUFFERS_B[i].resize_(0)
            GPU_BUFFERS_B = None
            USE_DOUBLE_BUFFER = False
        CPU_BUFFERS = None
        GPU_BUFFERS = None
        BUFFER_EVENTS_A = None
        BUFFER_EVENTS_B = None
        NEXT_BUFFER_SLOT = None
        torch.cuda.empty_cache()
        gc.collect()

    if (torch.utils.checkpoint.checkpoint.__name__ == "unsloth_checkpoint") and \
        hasattr(torch.utils.checkpoint, "_old_checkpoint"):

        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint
        del torch.utils.checkpoint._old_checkpoint

    try:
        import transformers.modeling_utils
        if (hasattr(transformers.modeling_utils, "_old_checkpoint") and
            hasattr(transformers.modeling_utils, "checkpoint") and
            transformers.modeling_utils.checkpoint.__name__ == "unsloth_checkpoint"):
            transformers.modeling_utils.checkpoint = transformers.modeling_utils._old_checkpoint
            del transformers.modeling_utils._old_checkpoint
    except Exception:
        pass
pass


def reset_unsloth_gradient_checkpointing_buffers():
    """
    All Unsloth Zoo code licensed under LGPLv3

    Resets CPU_BUFFERS and GPU_BUFFERS to their initial sizes after training.

    This function should be called after trainer.train() completes to free up
    memory that was allocated during training while keeping the buffers ready
    for another potential training run. Unlike unpatch_unsloth_smart_gradient_checkpointing,
    this does NOT destroy the buffers or unpatch the checkpointing - it just resets
    them to their initial state.

    Usage:
        trainer.train()
        reset_unsloth_gradient_checkpointing_buffers()  # Free memory, stay ready
        # Can run trainer.train() again without re-initializing
    """
    global CPU_BUFFERS
    global GPU_BUFFERS
    global CPU_INDEX
    global BACKWARD_PASS
    global LAST_GC_INDEX
    global FIRST_PASS
    global CURRENT_GC_INDEX
    global USE_UNSLOTH_GC
    global NEXT_BUFFER_SLOT
    global GPU_BUFFERS_B
    global USE_DOUBLE_BUFFER
    global BUFFER_EVENTS_A
    global BUFFER_EVENTS_B

    UnslothGradientCheckpointer.reset_for_new_training()

    # Check if buffers exist
    if CPU_BUFFERS is None or GPU_BUFFERS is None:
        return
    if len(CPU_BUFFERS) == 0:
        return

    # Reset CPU buffers to initial size and remove excess buffers
    for i in range(len(CPU_BUFFERS)):
        if i < INITIAL_CPU_BUFFER_COUNT:
            # Resize existing buffers back to initial size
            if CPU_BUFFERS[i] is not None and hasattr(CPU_BUFFERS[i], "resize_"):
                CPU_BUFFERS[i].resize_(INITIAL_CPU_BUFFER_SIZE)
        else:
            # Free excess buffers that were added during training
            if CPU_BUFFERS[i] is not None and hasattr(CPU_BUFFERS[i], "resize_"):
                CPU_BUFFERS[i].resize_(0)
            CPU_BUFFERS[i] = None
    pass

    # Trim the list back to initial count if it grew
    if len(CPU_BUFFERS) > INITIAL_CPU_BUFFER_COUNT:
        del CPU_BUFFERS[INITIAL_CPU_BUFFER_COUNT:]
    pass

    # Reset GPU buffers to initial size
    for i in range(len(GPU_BUFFERS)):
        if GPU_BUFFERS[i] is not None and hasattr(GPU_BUFFERS[i], "resize_"):
            GPU_BUFFERS[i].resize_(INITIAL_GPU_BUFFER_SIZE)
    pass

    # Reset state variables for fresh training run
    CPU_INDEX = 0
    BACKWARD_PASS = True
    LAST_GC_INDEX = 0
    FIRST_PASS = True
    CURRENT_GC_INDEX = 0
    USE_UNSLOTH_GC = True  # Re-enable the "Will smartly offload" message
    if NEXT_BUFFER_SLOT is not None:
        for i in range(len(NEXT_BUFFER_SLOT)):
            NEXT_BUFFER_SLOT[i] = 0

    # Reset double buffering if buffer B still exists, or try to re-allocate
    if os.environ.get("UNSLOTH_DISABLE_DOUBLE_BUFFER", "0") == "1":
        if GPU_BUFFERS_B is not None:
            for i in range(len(GPU_BUFFERS_B)):
                if GPU_BUFFERS_B[i] is not None and hasattr(GPU_BUFFERS_B[i], "resize_"):
                    GPU_BUFFERS_B[i].resize_(0)
            GPU_BUFFERS_B = None
        USE_DOUBLE_BUFFER = False
    elif GPU_BUFFERS_B is not None:
        for i in range(len(GPU_BUFFERS_B)):
            if GPU_BUFFERS_B[i] is not None and hasattr(GPU_BUFFERS_B[i], "resize_"):
                GPU_BUFFERS_B[i].resize_(INITIAL_GPU_BUFFER_SIZE)
        USE_DOUBLE_BUFFER = False
    else:
        try:
            n_gpus = len(GPU_BUFFERS)
            dtype = GPU_BUFFERS[0].dtype
            GPU_BUFFERS_B = tuple([torch.empty(INITIAL_GPU_BUFFER_SIZE, dtype=dtype, device=f"{DEVICE_TYPE_TORCH}:{i}") for i in range(n_gpus)])
            if DEVICE_TYPE in ("cuda", "hip"):
                event_ctor = torch.cuda.Event
            elif DEVICE_TYPE == "xpu":
                event_ctor = torch.xpu.Event
            else:
                raise RuntimeError(f"Double buffering unsupported on {DEVICE_TYPE}")
            BUFFER_EVENTS_A = tuple([event_ctor() for _ in range(n_gpus)])
            BUFFER_EVENTS_B = tuple([event_ctor() for _ in range(n_gpus)])
            USE_DOUBLE_BUFFER = False
        except RuntimeError:
            pass

    # Clean up freed memory
    torch.cuda.empty_cache()
    gc.collect()
pass


def unsloth_offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return unsloth_checkpoint(function, *args, use_reentrant = False, **kwargs)
pass
