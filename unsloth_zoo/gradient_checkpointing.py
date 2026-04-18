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
from typing import Union, Optional, List, Any, Callable, Tuple
import os
import warnings
import gc
import weakref
import time
import atexit
from collections import defaultdict
from contextvars import ContextVar
from .utils import _get_dtype, Version
from .device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
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
    "UnslothOffloadActivations",
    "resolve_sac_context_fn",
    "set_sac_policy",
    "resolve_gc_offload_backend",
    "set_offload_backend",
    "_bind_gradient_checkpointing_func",
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
_GC_OFFLOAD_BACKENDS = {"noop", "hooks", "hooks_prefetch"}
# Ring slot count for prefetch-capable unpack. Must be strictly greater than
# the max prefetch depth you intend to use; with depth D the prefetch of K+D
# runs while main_stream is still consuming slot K, so we need at least D+1
# distinct slots to avoid aliasing. Override via UNSLOTH_GC_PREFETCH_RING_SIZE.
_GC_PREFETCH_RING_SIZE = int(os.environ.get("UNSLOTH_GC_PREFETCH_RING_SIZE", "4"))
_gc_profile_module: ContextVar[Optional[str]] = ContextVar("_gc_profile_module", default=None)
_gc_profile_enabled_cache: Optional[bool] = None
_gc_profile_registered = False
_gc_profile_state = {
    "mode": None,
    "totals": defaultdict(float),
    "module_stats": defaultdict(lambda: defaultdict(float)),
    "shape_stats": defaultdict(lambda: defaultdict(float)),
    "skip_reasons": defaultdict(int),
}


def _gc_profile_enabled() -> bool:
    global _gc_profile_enabled_cache
    if _gc_profile_enabled_cache is None:
        value = os.environ.get("UNSLOTH_GC_PROFILE", "0")
        _gc_profile_enabled_cache = str(value).strip().lower() not in ("0", "false", "no", "off", "")
    return bool(_gc_profile_enabled_cache)


def _gc_profile_module_name() -> str:
    name = _gc_profile_module.get()
    if name:
        return name
    return "<unknown>"


def _gc_profile_resolve_function_module_name(function) -> Optional[str]:
    module = getattr(function, "__self__", None)
    if isinstance(module, torch.nn.Module):
        return getattr(module, "_unsloth_gc_profile_module_name", None) or module.__class__.__name__

    closure = getattr(function, "__closure__", None)
    if closure:
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, torch.nn.Module):
                return getattr(value, "_unsloth_gc_profile_module_name", None) or value.__class__.__name__
            maybe_module = getattr(value, "__self__", None)
            if isinstance(maybe_module, torch.nn.Module):
                return getattr(maybe_module, "_unsloth_gc_profile_module_name", None) or maybe_module.__class__.__name__
    return None


def _gc_profile_shape_key(shape, dtype) -> str:
    dims = "x".join(str(int(x)) for x in shape)
    return f"{dims}:{dtype}"


def _gc_profile_ensure(mode: str) -> bool:
    global _gc_profile_registered
    if not _gc_profile_enabled():
        return False
    if _gc_profile_state["mode"] is None:
        _gc_profile_state["mode"] = mode
    if not _gc_profile_registered:
        atexit.register(_gc_profile_dump_summary)
        _gc_profile_registered = True
    return True


def _gc_profile_record_skip(reason: str) -> None:
    if not _gc_profile_enabled():
        return
    _gc_profile_state["skip_reasons"][reason] += 1


def _gc_profile_record(
    *,
    mode: str,
    module_name: str,
    shape,
    dtype,
    numel: int,
    kind: str,
    duration_s: float = 0.0,
    wait_s: float = 0.0,
    count: int = 1,
    extra_allocated: bool = False,
    pool_hit: bool = False,
    wait_event_fallback: bool = False,
) -> None:
    if not _gc_profile_ensure(mode):
        return

    n_bytes = int(numel) * torch.tensor([], dtype=dtype).element_size()
    totals = _gc_profile_state["totals"]
    totals[f"{kind}_count"] += count
    totals[f"{kind}_bytes"] += n_bytes
    totals[f"{kind}_cpu_s"] += duration_s
    totals[f"{kind}_wait_s"] += wait_s
    if extra_allocated:
        totals["cpu_buffer_allocs"] += 1
    if pool_hit:
        totals["cpu_buffer_pool_hits"] += 1
    if wait_event_fallback:
        totals["wait_stream_fallbacks"] += 1

    module_stats = _gc_profile_state["module_stats"][module_name]
    module_stats[f"{kind}_count"] += count
    module_stats[f"{kind}_bytes"] += n_bytes
    module_stats[f"{kind}_cpu_s"] += duration_s
    module_stats[f"{kind}_wait_s"] += wait_s

    shape_key = _gc_profile_shape_key(shape, dtype)
    shape_stats = _gc_profile_state["shape_stats"][shape_key]
    shape_stats[f"{kind}_count"] += count
    shape_stats[f"{kind}_bytes"] += n_bytes
    shape_stats[f"{kind}_cpu_s"] += duration_s
    shape_stats[f"{kind}_wait_s"] += wait_s


def _gc_profile_top_lines(stats_dict, primary_key: str, extra_keys: Tuple[str, ...], limit: int = 10):
    items = [
        (name, values)
        for name, values in stats_dict.items()
        if values.get(primary_key, 0.0) > 0
    ]
    items.sort(key = lambda item: item[1].get(primary_key, 0.0), reverse = True)
    lines = []
    for name, values in items[:limit]:
        parts = [f"{primary_key}={values.get(primary_key, 0.0):.0f}" if "bytes" in primary_key or "count" in primary_key else f"{primary_key}={values.get(primary_key, 0.0):.6f}s"]
        for key in extra_keys:
            value = values.get(key, 0.0)
            if "bytes" in key or "count" in key:
                parts.append(f"{key}={value:.0f}")
            else:
                parts.append(f"{key}={value:.6f}s")
        lines.append(f"  - {name}: " + ", ".join(parts))
    return lines


def _gc_profile_dump_summary() -> None:
    if not _gc_profile_enabled():
        return
    totals = _gc_profile_state["totals"]
    mode = _gc_profile_state["mode"] or "unknown"
    print(f"Unsloth GC profile summary ({mode}):")
    print(
        "  totals: "
        f"pack_count={totals.get('pack_count', 0):.0f}, "
        f"pack_bytes={totals.get('pack_bytes', 0):.0f}, "
        f"pack_cpu_s={totals.get('pack_cpu_s', 0.0):.6f}, "
        f"unpack_count={totals.get('unpack_count', 0):.0f}, "
        f"unpack_bytes={totals.get('unpack_bytes', 0):.0f}, "
        f"unpack_cpu_s={totals.get('unpack_cpu_s', 0.0):.6f}, "
        f"unpack_wait_s={totals.get('unpack_wait_s', 0.0):.6f}, "
        f"cpu_buffer_pool_hits={totals.get('cpu_buffer_pool_hits', 0):.0f}, "
        f"cpu_buffer_allocs={totals.get('cpu_buffer_allocs', 0):.0f}, "
        f"wait_stream_fallbacks={totals.get('wait_stream_fallbacks', 0):.0f}"
    )
    if _gc_profile_state["skip_reasons"]:
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(_gc_profile_state["skip_reasons"].items())
        )
        print(f"  skips: {reasons}")

    module_lines = _gc_profile_top_lines(
        _gc_profile_state["module_stats"],
        "unpack_bytes",
        ("unpack_count", "unpack_cpu_s"),
    )
    if module_lines:
        print("  top modules by unpack_bytes:")
        for line in module_lines:
            print(line)

    module_lines = _gc_profile_top_lines(
        _gc_profile_state["module_stats"],
        "unpack_cpu_s",
        ("unpack_bytes", "unpack_count"),
    )
    if module_lines:
        print("  top modules by unpack_cpu_s:")
        for line in module_lines:
            print(line)

    module_lines = _gc_profile_top_lines(
        _gc_profile_state["module_stats"],
        "pack_bytes",
        ("pack_count", "pack_cpu_s"),
    )
    if module_lines:
        print("  top modules by pack_bytes:")
        for line in module_lines:
            print(line)

    shape_lines = _gc_profile_top_lines(
        _gc_profile_state["shape_stats"],
        "unpack_bytes",
        ("unpack_count", "unpack_cpu_s"),
    )
    if shape_lines:
        print("  top tensor shapes by unpack_bytes:")
        for line in shape_lines:
            print(line)

    shape_lines = _gc_profile_top_lines(
        _gc_profile_state["shape_stats"],
        "unpack_cpu_s",
        ("unpack_bytes", "unpack_count"),
    )
    if shape_lines:
        print("  top tensor shapes by unpack_cpu_s:")
        for line in shape_lines:
            print(line)


def resolve_gc_offload_backend(backend = None):
    if backend is None:
        backend = os.environ.get("UNSLOTH_GC_OFFLOAD_BACKEND", "hooks")
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
    via retained unshard state across layers, while the non-reentrant path
    (Mode A hooks / Mode B noop) composes cleanly with FSDP2.
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


def _ensure_sac_ops():
    """Lazily resolve op handles on first use."""
    global _SAC_ATTENTION_OPS, _SAC_MATMUL_OPS
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
        "aten.bmm.default",
        "aten.addmm.default",
    ):
        op = _try_resolve_op(op_name)
        if op is not None:
            _SAC_MATMUL_OPS.add(op)


def _sac_policy_attn_only(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _sac_policy_attn_and_matmul(ctx, op, *args, **kwargs):
    if op in _SAC_ATTENTION_OPS or op in _SAC_MATMUL_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


_SAC_PRESETS = {
    "attn_only": _sac_policy_attn_only,
    "attn_and_matmul": _sac_policy_attn_and_matmul,
}


def resolve_sac_context_fn(policy):
    """Resolve a user-provided SAC policy to a context_fn callable (or None).

    Args:
        policy: One of:
            - None → no SAC (returns None)
            - str  → preset name ("attn_only", "attn_and_matmul")
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

    if isinstance(policy, str):
        if policy not in _SAC_PRESETS:
            raise ValueError(
                f"Unsloth: Unknown SAC preset {policy!r}. "
                f"Available: {list(_SAC_PRESETS.keys())}"
            )
        policy_fn = _SAC_PRESETS[policy]
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
            f"Unsloth: sac_policy must be None, a string, a list of ops, "
            f"or a callable, got {type(policy).__name__}"
        )

    return functools.partial(create_selective_checkpoint_contexts, policy_fn)


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
    # context_fn / offload_backend are only valid for the non-reentrant path.
    # Keeping them bound in reentrant mode can route layers back into the
    # non-reentrant runtime after a mode switch.
    if (use_reentrant is False) and context_fn is not None:
        partial_kwargs["context_fn"] = context_fn
    if (use_reentrant is False) and offload_backend is not None:
        partial_kwargs["offload_backend"] = resolve_gc_offload_backend(offload_backend)

    if partial_kwargs:
        bound = functools.partial(checkpoint_fn, **partial_kwargs)
    else:
        bound = checkpoint_fn

    named_modules = {
        id(candidate): (name or candidate.__class__.__name__)
        for name, candidate in model.named_modules()
    }
    for module in model.modules():
        if not hasattr(module, "_gradient_checkpointing_func"):
            continue
        if _gc_profile_enabled():
            module._unsloth_gc_profile_module_name = named_modules.get(id(module), module.__class__.__name__)
        if not _gc_profile_enabled():
            module._gradient_checkpointing_func = bound
            continue

        module_name = named_modules.get(id(module), module.__class__.__name__)

        @functools.wraps(bound)
        def profiled_checkpoint_call(function, *args, __bound = bound, __module_name = module_name, **kwargs):
            token = _gc_profile_module.set(__module_name)
            try:
                return __bound(function, *args, **kwargs)
            finally:
                _gc_profile_module.reset(token)

        module._gradient_checkpointing_func = profiled_checkpoint_call


def set_sac_policy(model, policy):
    """Set or clear SAC policy at runtime (no model reload needed).

    Args:
        model: The model (must have been loaded with use_reentrant=False).
        policy: Same as ``sac_policy`` in ``from_pretrained``, or None to disable.

    Raises:
        ValueError: If the model uses reentrant checkpointing.
    """
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
    global _default_offload_backend
    backend = resolve_gc_offload_backend(backend)
    model._unsloth_gc_offload_backend = backend
    _default_offload_backend = backend
    use_reentrant = getattr(model, "_unsloth_use_reentrant", True)
    context_fn = getattr(model, "_unsloth_sac_context_fn", None)
    checkpoint_fn = torch.utils.checkpoint.checkpoint
    _bind_gradient_checkpointing_func(
        model, checkpoint_fn, use_reentrant, context_fn, backend,
    )
    try:
        from .training_utils import (
            _install_hook_based_offload_wrapper,
            _remove_hook_based_offload_wrapper,
        )
        dtype = getattr(getattr(model, "config", None), "torch_dtype", None)
        if (not use_reentrant) and backend == "hooks":
            _install_hook_based_offload_wrapper(model, dtype)
        else:
            _remove_hook_based_offload_wrapper(model)
    except Exception:
        pass
    return backend


# Initial buffer sizes for gradient checkpointing
INITIAL_CPU_BUFFER_SIZE = 128 * 1024       # Initial size per CPU buffer
INITIAL_GPU_BUFFER_SIZE = 2 * 256 * 2048   # Initial size per GPU buffer
INITIAL_CPU_BUFFER_COUNT = 200             # Initial number of CPU buffers

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
    ContextManager,
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
global BACKWARD_PASS
global EXTRA_STREAMS
global MAIN_STREAMS
global MINIMUM_SIZE
global USE_UNSLOTH_GC
global LAST_GC_INDEX
global FIRST_PASS
global CURRENT_GC_INDEX

if DEVICE_TYPE in ("cuda", "hip"):
    torch_gpu_stream = torch.cuda.stream
elif DEVICE_TYPE == "xpu":
    torch_gpu_stream = torch.xpu.stream

CPU_BUFFERS = []
CPU_INDEX = None
_noop_offload_state: ContextVar[Optional[dict]] = ContextVar("_noop_offload_state", default=None)
_hooks_offload_state: ContextVar[Optional[dict]] = ContextVar("_hooks_offload_state", default=None)
# Module-level default offload backend, set by set_offload_backend().
# Used as fallback when HF's gradient_checkpointing_enable() overwrites the
# per-module partial binding and drops the offload_backend kwarg.
_default_offload_backend: Optional[str] = None
pass


def _patch_noop_save_inputs():
    cls = getattr(torch.utils.checkpoint, "_NoopSaveInputs", None)
    if cls is None:
        return
    if UnslothGradientCheckpointer._original_noop_setup_context is not None:
        return
    UnslothGradientCheckpointer._original_noop_setup_context = cls.setup_context

    def unsloth_setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        state = _noop_offload_state.get()
        if state is None:
            return UnslothGradientCheckpointer._original_noop_setup_context(ctx, inputs, output)

        offloader = state.get("offloader", None)
        if offloader is None:
            return UnslothGradientCheckpointer._original_noop_setup_context(ctx, inputs, output)

        n_inputs = len(inputs)
        # 0 = non-tensor, 1 = saved tensor, 2 = offloaded tensor
        entry_kind = [0] * n_inputs
        entry_index = [-1] * n_inputs
        non_tensor_values = [None] * n_inputs
        saved_tensors = []
        offloaded = []

        for i, o in enumerate(inputs):
            if not isinstance(o, torch.Tensor):
                non_tensor_values[i] = o
                continue

            if offloader.should_offload(o):
                packed = offloader.pack_hook(o)
                if isinstance(packed, PackedCPUBuffer):
                    off_idx = len(offloaded)
                    offloaded.append(packed)
                    entry_kind[i] = 2
                    entry_index[i] = off_idx
                    continue

            saved_idx = len(saved_tensors)
            saved_tensors.append(o)
            entry_kind[i] = 1
            entry_index[i] = saved_idx

        def get_args(saved_tensors_runtime):
            ret = [None] * (n_inputs - 1)
            for i in range(1, n_inputs):
                kind = entry_kind[i]
                if kind == 0:
                    ret[i - 1] = non_tensor_values[i]
                elif kind == 1:
                    ret[i - 1] = saved_tensors_runtime[entry_index[i]]
                else:
                    ret[i - 1] = UnslothGradientCheckpointer.unpack_packed(offloaded[entry_index[i]])
            return ret

        ctx.get_args = get_args
        ctx.save_for_backward(*saved_tensors)

    cls.setup_context = staticmethod(unsloth_setup_context)
pass


def _unpatch_noop_save_inputs():
    cls = getattr(torch.utils.checkpoint, "_NoopSaveInputs", None)
    if cls is None:
        return
    if UnslothGradientCheckpointer._original_noop_setup_context is None:
        return
    cls.setup_context = UnslothGradientCheckpointer._original_noop_setup_context
    UnslothGradientCheckpointer._original_noop_setup_context = None
    _noop_offload_state.set(None)
pass


def _gc_disable_cpu_offload():
    value = os.environ.get("UNSLOTH_GC_DISABLE_CPU_OFFLOAD", None)
    if value is None:
        return False
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")
pass


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
        f"Pinned memory cannot be swapped and exhaustion causes cryptic CUDA errors. "
        f"To disable CPU offloading: set UNSLOTH_GC_DISABLE_CPU_OFFLOAD=1",
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


# In practice you probably don't need to rewrap after restoring from CPU,
# because autograd only needs the local shard for backward and FSDP2
# handles gradient reduction separately. Kept here for completeness
# in case a future codepath requires DTensor metadata on restored tensors.
#
# def _rewrap_dtensor(restored, original):
#     """Re-wrap a plain tensor as DTensor using the original's metadata."""
#     if hasattr(original, "_spec") and hasattr(original, "device_mesh"):
#         try:
#             from torch.distributed.tensor import DTensor
#             return DTensor.from_local(
#                 restored,
#                 device_mesh=original.device_mesh,
#                 placements=original.placements,
#             )
#         except Exception:
#             pass
#     return restored


class UnslothGradientCheckpointer:
    """
    All Unsloth Zoo code licensed under LGPLv3

    Non-reentrant gradient checkpointing with smart CPU offloading.
    """
    _cpu_buffers: List[torch.Tensor] = []
    _cpu_free_buffers: dict = {}
    _gpu_buffers: dict = {}
    # Persistent GPU restore buffer -- mirrors reentrant's GPU_BUFFERS pattern.
    # One buffer per (dtype, device_index), never freed, only grown.
    # Avoids CUDA allocator fragmentation from repeated alloc/free during backward.
    _gpu_restore_persistent: dict = {}  # (dtype, device_index) -> GPU tensor (single-slot mode)
    _gpu_restore_ring: dict = {}        # (dtype, device_index, slot) -> GPU tensor (prefetch mode)
    _pending_unpacks: list = []         # PackedCPUBuffer in pack order awaiting unpack
    _next_pack_idx: int = 0             # monotonic pack counter for ring-slot assignment
    _main_streams: dict = {}
    _extra_streams: dict = {}
    _initialized: bool = False

    _current_gc_index: int = 0
    _last_gc_index: int = 0
    _first_pass: bool = True
    _backward_pass: bool = True
    _minimum_size: int = 2 * 1024 * 1024 // 2
    _use_unsloth_gc_message: bool = True
    _dtype: torch.dtype = None
    _events_supported: Optional[bool] = None
    _meta_initialized: bool = False
    _original_noop_setup_context: Optional[Any] = None

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
        cls._minimum_size = 2 * 1024 * 1024 // n_bytes
        cls._meta_initialized = True

    @classmethod
    def initialize(cls, dtype: torch.dtype = None, num_devices: int = None):
        if cls._initialized:
            return

        if dtype is None:
            if DEVICE_TYPE == "cuda":
                major_version, minor_version = torch.cuda.get_device_capability()
                supports_bfloat16 = (major_version >= 8)
            elif DEVICE_TYPE in ("hip", "xpu"):
                supports_bfloat16 = True
            else:
                supports_bfloat16 = True
            dtype = torch.bfloat16 if supports_bfloat16 else torch.float16

        cls._dtype = dtype
        n_bytes = torch.finfo(dtype).bits // 8
        cls._minimum_size = 2 * 1024 * 1024 // n_bytes
        cls._meta_initialized = True

        _track_pinned_alloc(INITIAL_CPU_BUFFER_SIZE * INITIAL_CPU_BUFFER_COUNT, dtype)
        cls._cpu_buffers = [
            torch.empty(INITIAL_CPU_BUFFER_SIZE, dtype=dtype, device="cpu", pin_memory=True)
            for _ in range(INITIAL_CPU_BUFFER_COUNT)
        ]
        cls._cpu_free_buffers = {
            dtype: [(buf, None, None) for buf in cls._cpu_buffers]
        }

        if num_devices is None:
            num_devices = torch.cuda.device_count() if DEVICE_TYPE in ("cuda", "hip") else torch.xpu.device_count()

        try:
            for device_idx in range(num_devices):
                device = torch.device(f"{DEVICE_TYPE_TORCH}:{device_idx}")
                cls._gpu_buffers[device_idx] = torch.empty(
                    INITIAL_GPU_BUFFER_SIZE, dtype=dtype, device=device,
                )
                if DEVICE_TYPE in ("cuda", "hip"):
                    cls._main_streams[device_idx] = torch.cuda.default_stream(device)
                    cls._extra_streams[device_idx] = torch.cuda.Stream(device)
                elif DEVICE_TYPE == "xpu":
                    cls._main_streams[device_idx] = torch.xpu.current_stream(device)
                    cls._extra_streams[device_idx] = torch.xpu.Stream(device)
        except Exception:
            print("="*10 + "\n")
            print("Unsloth: Your setup does not support `PYTORCH_CUDA_ALLOC_CONF`\n")
            print("Please set `import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '';`\n")
            print("Then re-run Unsloth from the start.")
            print("="*10 + "\n")
            raise

        cls._initialized = True

    @classmethod
    def reset_for_new_training(cls):
        global _pinned_bytes_allocated, _cpu_ram_warned
        if not cls._initialized:
            return

        _pinned_bytes_allocated = 0
        _cpu_ram_warned = False
        cls._cpu_free_buffers = {}
        cls._current_gc_index = 0
        cls._last_gc_index = 0
        cls._first_pass = True
        cls._backward_pass = True
        cls._use_unsloth_gc_message = True

        for i in range(len(cls._cpu_buffers)):
            if i < INITIAL_CPU_BUFFER_COUNT:
                if cls._cpu_buffers[i] is not None and hasattr(cls._cpu_buffers[i], "resize_"):
                    cls._cpu_buffers[i].resize_(INITIAL_CPU_BUFFER_SIZE)
            else:
                if cls._cpu_buffers[i] is not None and hasattr(cls._cpu_buffers[i], "resize_"):
                    cls._cpu_buffers[i].resize_(0)
                cls._cpu_buffers[i] = None

        if len(cls._cpu_buffers) > INITIAL_CPU_BUFFER_COUNT:
            del cls._cpu_buffers[INITIAL_CPU_BUFFER_COUNT:]
        cls._cpu_free_buffers[cls._dtype] = [
            (buf, None, None)
            for buf in cls._cpu_buffers
            if buf is not None
        ]

        for device_idx in cls._gpu_buffers:
            if cls._gpu_buffers[device_idx] is not None and hasattr(cls._gpu_buffers[device_idx], "resize_"):
                cls._gpu_buffers[device_idx].resize_(INITIAL_GPU_BUFFER_SIZE)
        # Keep persistent GPU restore buffers allocated -- they'll be reused next training.

    @classmethod
    def cleanup(cls):
        if not cls._initialized:
            return

        for i in range(len(cls._cpu_buffers)):
            if cls._cpu_buffers[i] is not None and hasattr(cls._cpu_buffers[i], "resize_"):
                cls._cpu_buffers[i].resize_(0)
            cls._cpu_buffers[i] = None
        cls._cpu_buffers = []
        cls._cpu_free_buffers = {}

        for device_idx in list(cls._gpu_buffers.keys()):
            if cls._gpu_buffers[device_idx] is not None and hasattr(cls._gpu_buffers[device_idx], "resize_"):
                cls._gpu_buffers[device_idx].resize_(0)
            cls._gpu_buffers[device_idx] = None
        cls._gpu_buffers = {}
        for key, buf in cls._gpu_restore_persistent.items():
            if buf is not None and hasattr(buf, "resize_"):
                buf.resize_(0)
        cls._gpu_restore_persistent = {}
        cls._main_streams = {}
        cls._extra_streams = {}
        cls._initialized = False

    def __init__(self, is_last_layer: bool = False):
        self.is_last_layer = is_last_layer

    @classmethod
    def begin_checkpoint(cls, dtype=None):
        """Per-call bookkeeping: initialize if needed, track layer index, return offloader."""
        if not cls._initialized:
            cls.initialize(dtype)
        if cls._backward_pass:
            cls._backward_pass = False
            cls._current_gc_index = 0
        if cls._first_pass:
            cls._last_gc_index += 1
        cls._current_gc_index += 1
        is_last_layer = (cls._current_gc_index == cls._last_gc_index) and not cls._first_pass
        return cls(is_last_layer=is_last_layer)

    @classmethod
    def _record_stream_event(cls, stream):
        if cls._events_supported is False:
            return None
        try:
            event = stream.record_event()
            cls._events_supported = True
            return event
        except Exception:
            cls._events_supported = False
            return None

    @classmethod
    def _wait_event(cls, stream, event):
        if event is None:
            return False
        try:
            stream.wait_event(event)
            return True
        except Exception:
            cls._events_supported = False
            return False

    def should_offload(self, tensor: torch.Tensor) -> bool:
        cls = self.__class__
        if _gc_disable_cpu_offload():
            _gc_profile_record_skip("cpu_offload_disabled")
            return False
        # Skip parameter-like tensors (requires_grad=True leaf tensors).
        # Under FSDP, all-gathered params lose nn.Parameter type, but
        # they remain leaf tensors (no grad_fn). Activations always have grad_fn.
        if tensor.requires_grad and tensor.grad_fn is None:
            _gc_profile_record_skip("parameter_like_tensor")
            return False
        # Unwrap FSDP2 DTensors so .numel(), .is_contiguous(), etc.
        # check the local shard, not the virtualized distributed shape.
        tensor = _unwrap_dtensor(tensor)
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            if torch.compiler.is_compiling():
                _gc_profile_record_skip("compiler_is_compiling")
                return False
        if tensor.numel() < cls._minimum_size:
            _gc_profile_record_skip("below_minimum_size")
            return False
        if tensor.device.type == "cpu":
            _gc_profile_record_skip("tensor_on_cpu")
            return False
        if self.is_last_layer:
            _gc_profile_record_skip("last_layer")
            return False
        # Custom packed-buffer restore currently materializes contiguous tensors.
        # Restrict offload to safe tensor layouts to preserve correctness.
        if tensor.layout != torch.strided:
            _gc_profile_record_skip("non_strided_layout")
            return False
        if (not tensor.is_contiguous()) or (tensor.storage_offset() != 0):
            _gc_profile_record_skip("non_contiguous_or_offset")
            return False
        try:
            if tensor._is_view():
                _gc_profile_record_skip("tensor_view")
                return False
        except Exception:
            pass
        return True

    @classmethod
    def _acquire_cpu_buffer(cls, *, numel: int, dtype: torch.dtype, device_index: int):
        pool = cls._cpu_free_buffers.setdefault(dtype, [])
        chosen_idx = None
        chosen_buf = None
        # this whole bit may not be needed but in the future if we move to multi stream
        # this will be helpful infra
        # could be made more efficient so we don't over query events
        for i in range(len(pool) - 1, -1, -1):
            buf, fence_device_index, fence_event = pool[i]
            # Skip other devices FIRST (avoid device switch + query)
            if fence_device_index is not None and fence_device_index != device_index:
                continue

            ready = True
            if fence_event is not None:
                try:
                    if DEVICE_TYPE in ("cuda", "hip") and fence_device_index is not None:
                        with torch.cuda.device(fence_device_index):
                            ready = bool(fence_event.query())
                    elif DEVICE_TYPE == "xpu" and fence_device_index is not None:
                        with torch.xpu.device(fence_device_index):
                            ready = bool(fence_event.query())
                    else:
                        ready = bool(fence_event.query())
                except Exception:
                    ready = False
            if not ready:
                continue
            if fence_device_index is not None and fence_device_index != device_index:
                # Keep per-device stream-fenced buffers isolated.
                continue
            chosen_idx = i
            chosen_buf = buf
            break

        if chosen_idx is not None:
            pool.pop(chosen_idx)
            allocated = False
            if chosen_buf.numel() < numel:
                _track_pinned_alloc(numel, dtype)
                chosen_buf = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
                allocated = True
            return chosen_buf, True, allocated

        _track_pinned_alloc(numel, dtype)
        return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True), False, True

    @classmethod
    def _release_cpu_buffer(
        cls,
        *,
        cpu_buffer: torch.Tensor,
        dtype: torch.dtype,
        device_index: int,
        restore_event,
    ) -> None:
        pool = cls._cpu_free_buffers.setdefault(dtype, [])
        pool.append((cpu_buffer, device_index, restore_event))

    @classmethod
    def _get_gpu_restore_buffer(cls, *, numel: int, dtype: torch.dtype, device_index: int):
        """Get persistent GPU restore buffer, matching reentrant's GPU_BUFFERS pattern.

        One buffer per (dtype, device_index). Never freed, only grown via resize_().
        Eliminates CUDA allocator fragmentation from repeated alloc/free during backward.
        """
        key = (dtype, device_index)
        buf = cls._gpu_restore_persistent.get(key)
        if buf is None:
            buf = torch.empty(numel, dtype=dtype,
                device=f"{DEVICE_TYPE_TORCH}:{device_index}")
            cls._gpu_restore_persistent[key] = buf
        elif buf.numel() < numel:
            buf.resize_(numel)
        return buf

    @classmethod
    def _get_gpu_restore_ring_slot(cls, *, numel: int, dtype: torch.dtype,
                                   device_index: int, slot: int):
        """Get one slot of a per-(dtype, device) ring of GPU restore buffers.
        Used by the prefetch-capable unpack path so consecutive unpacks don't
        alias into a single shared buffer.
        """
        key = (dtype, device_index, slot)
        buf = cls._gpu_restore_ring.get(key)
        if buf is None:
            buf = torch.empty(numel, dtype=dtype,
                device=f"{DEVICE_TYPE_TORCH}:{device_index}")
            cls._gpu_restore_ring[key] = buf
        elif buf.numel() < numel:
            buf.resize_(numel)
        return buf

    @classmethod
    def _issue_h2d_for_pack(cls, packed):
        """Start the H2D copy for ``packed`` on its ring slot, idempotent.

        No main_stream wait; that happens at the consumer (the unpack caller
        that needs the data). Enables a prior unpack to prefetch the next.
        """
        state = packed._state
        if state.get("h2d_issued"):
            return
        original_dtype = packed.dtype
        numel = packed.numel
        device_index = packed.device_index
        shape = packed.shape
        original_stride = packed.stride
        cpu_buffer = state["cpu_buffer"]
        slot = packed.pack_idx % _GC_PREFETCH_RING_SIZE
        gpu_buf = cls._get_gpu_restore_ring_slot(
            numel=numel, dtype=original_dtype, device_index=device_index, slot=slot,
        )
        device = torch.device(f"{DEVICE_TYPE_TORCH}:{device_index}")
        if DEVICE_TYPE in ("cuda", "hip"):
            main_stream = torch.cuda.current_stream(device)
        elif DEVICE_TYPE == "xpu":
            main_stream = torch.xpu.current_stream(device)
        else:
            main_stream = cls._main_streams[device_index]
        extra_stream = cls._extra_streams[device_index]
        extra_stream.wait_stream(main_stream)
        with torch_gpu_stream(extra_stream):
            gpu_buf[:numel].copy_(cpu_buffer[:numel], non_blocking=True)
            result = gpu_buf[:numel].view(shape)
            if tuple(result.stride()) != tuple(original_stride):
                result = result.as_strided(shape, original_stride)
            restore_event = cls._record_stream_event(extra_stream)
        state["restore_event"] = restore_event
        state["result_view"] = result
        state["h2d_issued"] = True
        packed.set_restore_event(restore_event)

    def pack_hook(self, tensor: torch.Tensor):
        cls = self.__class__
        if not self.should_offload(tensor):
            return ("gpu", tensor)

        # Unwrap DTensor to local shard for plain memcpy (avoids distributed dispatch)
        tensor = _unwrap_dtensor(tensor)
        device = tensor.device
        device_index = device.index if device.index is not None else 0
        numel = tensor.numel()
        shape = tensor.shape
        stride = tensor.stride()
        dtype = tensor.dtype
        requires_grad = tensor.requires_grad

        if cls._use_unsloth_gc_message:
            print("Unsloth: Will smartly offload gradients to save VRAM!")
            cls._use_unsloth_gc_message = False

        cpu_buffer, pool_hit, extra_allocated = cls._acquire_cpu_buffer(
            numel=numel,
            dtype=dtype,
            device_index=device_index,
        )

        if DEVICE_TYPE in ("cuda", "hip"):
            main_stream = torch.cuda.current_stream(device)
        elif DEVICE_TYPE == "xpu":
            main_stream = torch.xpu.current_stream(device)
        else:
            main_stream = cls._main_streams[device_index]
        extra_stream = cls._extra_streams[device_index]
        extra_stream.wait_stream(main_stream)
        module_name = _gc_profile_module_name()
        start_time = time.perf_counter() if _gc_profile_enabled() else 0.0
        with torch_gpu_stream(extra_stream):
            try:
                tensor.record_stream(extra_stream)
            except Exception:
                pass
            cpu_buffer[:numel].view(shape).copy_(tensor, non_blocking=True)
        if _gc_profile_enabled():
            _gc_profile_record(
                mode = "nonreentrant_hooks",
                module_name = module_name,
                shape = shape,
                dtype = dtype,
                numel = numel,
                kind = "pack",
                duration_s = time.perf_counter() - start_time,
                extra_allocated = extra_allocated,
                pool_hit = pool_hit,
            )
        packed = PackedCPUBuffer(
            shape,
            stride,
            dtype,
            requires_grad,
            device_index,
            numel,
            cpu_buffer,
            cls,
            module_name,
        )
        # Prefetch-mode bookkeeping: assign pack index and register as pending.
        # Noop for single-slot (non-prefetch) unpack path.
        packed.pack_idx = cls._next_pack_idx
        cls._next_pack_idx += 1
        packed._state["h2d_issued"] = False
        packed._state["result_view"] = None
        cls._pending_unpacks.append(packed)
        return packed

    @classmethod
    def unpack_packed_prefetch(cls, packed):
        """Prefetch-capable unpack: issues H2D for ``packed`` if not already in
        flight, and speculatively starts H2D for the next-N packs in reverse-
        pack order so main_stream's backward compute can overlap the transfers.
        Each pack uses its own ring slot so the copies don't alias.

        Depth tunable via ``UNSLOTH_GC_PREFETCH_DEPTH`` (default 1). Depth must
        be < ring size so prefetched packs don't collide with each other.
        """
        cls._backward_pass = True
        cls._first_pass = False

        # Remove self from pending list (O(n) scan but n≈36 per step).
        try:
            cls._pending_unpacks.remove(packed)
        except ValueError:
            pass

        # Issue H2D for self if not yet done.
        if not packed._state.get("h2d_issued"):
            cls._issue_h2d_for_pack(packed)

        # Prefetch the next-N unpacks (tail of pending list).
        try:
            depth = int(os.environ.get("UNSLOTH_GC_PREFETCH_DEPTH", "1"))
        except ValueError:
            depth = 1
        depth = max(0, min(depth, _GC_PREFETCH_RING_SIZE - 1))
        for k in range(1, depth + 1):
            if k > len(cls._pending_unpacks):
                break
            next_packed = cls._pending_unpacks[-k]
            if not next_packed._state.get("h2d_issued"):
                cls._issue_h2d_for_pack(next_packed)

        # Wait on self's restore event so main_stream sees consistent data.
        device_index = packed.device_index
        device = torch.device(f"{DEVICE_TYPE_TORCH}:{device_index}")
        if DEVICE_TYPE in ("cuda", "hip"):
            main_stream = torch.cuda.current_stream(device)
        elif DEVICE_TYPE == "xpu":
            main_stream = torch.xpu.current_stream(device)
        else:
            main_stream = cls._main_streams[device_index]
        restore_event = packed._state["restore_event"]
        if not cls._wait_event(main_stream, restore_event):
            extra_stream = cls._extra_streams[device_index]
            main_stream.wait_stream(extra_stream)

        result = packed._state["result_view"]
        if result.dtype != packed.dtype:
            result = result.to(packed.dtype)
        if result.requires_grad != packed.requires_grad:
            result.requires_grad_(packed.requires_grad)
        return result

    @classmethod
    def unpack_packed(cls, packed):
        cls._backward_pass = True
        cls._first_pass = False

        shape = packed.shape
        original_stride = packed.stride
        original_dtype = packed.dtype
        original_requires_grad = packed.requires_grad
        device_index = packed.device_index
        numel = packed.numel
        cpu_buffer = packed.cpu_buffer

        device = torch.device(f"{DEVICE_TYPE_TORCH}:{device_index}")
        if DEVICE_TYPE in ("cuda", "hip"):
            main_stream = torch.cuda.current_stream(device)
        elif DEVICE_TYPE == "xpu":
            main_stream = torch.xpu.current_stream(device)
        else:
            main_stream = cls._main_streams[device_index]
        extra_stream = cls._extra_streams[device_index]
        module_name = packed.module_name
        start_time = time.perf_counter() if _gc_profile_enabled() else 0.0
        gpu_buf = cls._get_gpu_restore_buffer(
            numel=numel, dtype=original_dtype, device_index=device_index,
        )
        extra_stream.wait_stream(main_stream)
        with torch_gpu_stream(extra_stream):
            gpu_buf[:numel].copy_(cpu_buffer[:numel], non_blocking=True)
            result = gpu_buf[:numel].view(shape)
            if tuple(result.stride()) != tuple(original_stride):
                result = result.as_strided(shape, original_stride)
            restore_event = cls._record_stream_event(extra_stream)
            packed.set_restore_event(restore_event)
        wait_start = time.perf_counter() if _gc_profile_enabled() else 0.0
        wait_event_fallback = not cls._wait_event(main_stream, restore_event)
        if wait_event_fallback:
            main_stream.wait_stream(extra_stream)
        wait_duration = (time.perf_counter() - wait_start) if _gc_profile_enabled() else 0.0

        if result.dtype != original_dtype:
            result = result.to(original_dtype)
        if result.requires_grad != original_requires_grad:
            result.requires_grad_(original_requires_grad)
        # Restored tensor is a plain torch.Tensor (local shard).
        # No DTensor rewrap needed: autograd uses local shard for
        # backward, and FSDP2 handles gradient reduction separately.
        # result = _rewrap_dtensor(result, original_dtensor)
        if _gc_profile_enabled():
            _gc_profile_record(
                mode = "nonreentrant_hooks",
                module_name = module_name,
                shape = shape,
                dtype = original_dtype,
                numel = numel,
                kind = "unpack",
                duration_s = time.perf_counter() - start_time,
                wait_s = wait_duration,
                wait_event_fallback = wait_event_fallback,
            )
        return result


class PackedCPUBuffer:
    """Packed representation for offloaded activations.

    The backing pinned CPU buffer stays alive for as long as this object stays
    alive, which matches saved_tensors_hooks lifetime requirements. Cleanup is
    tied to object destruction instead of the start of a future forward pass.
    """

    __slots__ = (
        "shape",
        "stride",
        "dtype",
        "requires_grad",
        "device_index",
        "numel",
        "module_name",
        "pack_idx",
        "_state",
        "_finalizer",
        "__weakref__",
    )

    def __init__(
        self,
        shape,
        stride,
        dtype,
        requires_grad,
        device_index,
        numel,
        cpu_buffer,
        owner_cls,
        module_name,
    ):
        self.shape = shape
        self.stride = stride
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.device_index = device_index
        self.numel = numel
        self.module_name = module_name
        self._state = {
            "cpu_buffer": cpu_buffer,
            "dtype": dtype,
            "device_index": device_index,
            "restore_event": None,
            "released": False,
            "owner_cls": owner_cls,
        }
        self._finalizer = weakref.finalize(self, PackedCPUBuffer._finalize, self._state)

    @staticmethod
    def _finalize(state):
        if state["released"]:
            return
        state["released"] = True
        state["owner_cls"]._release_cpu_buffer(
            cpu_buffer=state["cpu_buffer"],
            dtype=state["dtype"],
            device_index=state["device_index"],
            restore_event=state["restore_event"],
        )

    @property
    def cpu_buffer(self):
        return self._state["cpu_buffer"]

    def set_restore_event(self, restore_event):
        if not self._state["released"]:
            self._state["restore_event"] = restore_event


class UnslothOffloadActivations(torch.autograd.graph.saved_tensors_hooks):
    """
    All Unsloth Zoo code licensed under LGPLv3

    Compile-compatible CPU activation offloading via saved_tensors_hooks.

    This operates at the autograd runtime level, AFTER compiled graphs
    produce tensors. saved_tensors_hooks are invisible to torch.compile,
    so this avoids graph breaks entirely.

    Reuses all existing UnslothGradientCheckpointer infrastructure:
    buffer pooling, event-fenced reuse, async stream coordination.
    """

    def __init__(self, *, dtype=None, enabled=True):
        self._enabled = enabled and not _gc_disable_cpu_offload()
        self._dtype = dtype
        self._offloader = None
        self._first_pass = True
        super().__init__(self._pack_hook, self._unpack_hook)

    def __enter__(self):
        if not self._enabled:
            return self
        cls = UnslothGradientCheckpointer
        if not cls._initialized:
            cls.initialize(self._dtype)
        # Reset forward/backward state tracking for this forward pass
        cls._backward_pass = False
        cls._current_gc_index = 0
        if self._first_pass:
            cls._last_gc_index = 0
        # Fresh prefetch bookkeeping per forward pass so pack indices start at 0
        # and we don't accumulate stale PackedCPUBuffers from a previous step
        # (the unpack path clears items, but protect against early exits too).
        cls._pending_unpacks = []
        cls._next_pack_idx = 0
        # Create a fresh offloader instance for this forward pass
        self._offloader = cls(is_last_layer=False)
        return super().__enter__()

    def __exit__(self, *args):
        if not self._enabled:
            return
        super().__exit__(*args)
        self._first_pass = False

    @staticmethod
    def _should_offload(tensor):
        """Mirrors UnslothGradientCheckpointer.should_offload() checks."""
        cls = UnslothGradientCheckpointer
        if not cls._meta_initialized:
            cls.ensure_metadata()
        # To restore the broader experimental behavior where hooks offload every
        # large saved tensor in the checkpointed region, remove the
        # `_hooks_offload_state` / `offloader.is_last_layer` gates below and let
        # `_pack_hook()` use `self._offloader` directly again. That widens
        # offload back to things like large saved weights, which increased
        # Qwen3-VL traffic substantially in benchmarking.
        state = _hooks_offload_state.get()
        if state is None:
            return False
        offloader = state.get("offloader", None)
        if offloader is None:
            return False
        if _gc_disable_cpu_offload():
            return False
        if tensor.requires_grad and tensor.grad_fn is None:
            return False
        # Unwrap FSDP2 DTensors so checks see the local shard
        tensor = _unwrap_dtensor(tensor)
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
            if torch.compiler.is_compiling():
                return False
        if tensor.numel() < cls._minimum_size:
            return False
        if tensor.device.type == "cpu":
            return False
        if offloader.is_last_layer:
            return False
        if tensor.layout != torch.strided:
            return False
        if (not tensor.is_contiguous()) or (tensor.storage_offset() != 0):
            return False
        try:
            if tensor._is_view():
                return False
        except Exception:
            pass
        return True

    def _pack_hook(self, tensor):
        if not self._enabled or self._offloader is None:
            return tensor
        state = _hooks_offload_state.get()
        if state is None:
            return ("gpu", tensor)
        offloader = state.get("offloader", None)
        if offloader is None or not self._should_offload(tensor):
            return ("gpu", tensor)
        return offloader.pack_hook(tensor)

    def _unpack_hook(self, packed):
        if not self._enabled:
            return packed
        if isinstance(packed, PackedCPUBuffer):
            state = _hooks_offload_state.get()
            use_prefetch = (
                state is not None and state.get("prefetch", False)
            )
            if use_prefetch:
                return UnslothGradientCheckpointer.unpack_packed_prefetch(packed)
            return UnslothGradientCheckpointer.unpack_packed(packed)
        if isinstance(packed, tuple) and packed[0] == "gpu":
            return packed[1]
        return packed


def initialize_unsloth_gradient_checkpointing(dtype = None):
    # All Unsloth Zoo code licensed under LGPLv3
    global CPU_BUFFERS
    global CPU_INDEX
    global GPU_BUFFERS
    global BACKWARD_PASS
    global EXTRA_STREAMS
    global MAIN_STREAMS
    global MINIMUM_SIZE
    global USE_UNSLOTH_GC
    global LAST_GC_INDEX
    global FIRST_PASS
    global CURRENT_GC_INDEX
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
    try:
        GPU_BUFFERS = tuple([torch.empty(2*256*2048, dtype = dtype, device = f"{DEVICE_TYPE_TORCH}:{i}") for i in range(n_gpus)])
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
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx._gc_profile_module = _gc_profile_module_name()
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
                        if new_size > GPU_BUFFER.numel(): GPU_BUFFER.resize_(new_size)
                        x = x[:new_size].view(shape)

                        # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
                        pack_start = time.perf_counter() if _gc_profile_enabled() else 0.0
                        EXTRA_STREAM.wait_stream(MAIN_STREAM)
                        with torch_gpu_stream(EXTRA_STREAM):
                            x.copy_(_arg, non_blocking = True)

                        ctx._saved_metadata = (new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM,)
                        if _gc_profile_enabled():
                            ctx._gc_profile_shape = shape
                            ctx._gc_profile_numel = new_size
                            ctx._gc_profile_dtype = _arg.dtype
                            _gc_profile_record(
                                mode = "reentrant",
                                module_name = ctx._gc_profile_module,
                                shape = shape,
                                dtype = arg.dtype,
                                numel = new_size,
                                kind = "pack",
                                duration_s = time.perf_counter() - pack_start,
                            )
                        CPU_INDEX += 1
                        tensor_inputs.append(None)

                        global USE_UNSLOTH_GC
                        if USE_UNSLOTH_GC:
                            print("Unsloth: Will smartly offload gradients to save VRAM!")
                            USE_UNSLOTH_GC = False
                    else:
                        ctx._saved_metadata = (None, None, None, None, None, None,)
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

        new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM = ctx._saved_metadata
        if CPU_INDEX is not None:
            buffer = GPU_BUFFERS[device_index][:new_size].view(shape)
            x = CPU_BUFFERS[CPU_INDEX][:new_size].view(shape)

            # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
            unpack_start = time.perf_counter() if _gc_profile_enabled() else 0.0
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
                wait_start = time.perf_counter() if _gc_profile_enabled() else 0.0
                MAIN_STREAM.wait_stream(EXTRA_STREAM)
                wait_duration = (time.perf_counter() - wait_start) if _gc_profile_enabled() else 0.0
                x = buffer.detach()
                x.requires_grad_(True)
                # Restored tensor is a plain torch.Tensor (local shard).
                # No DTensor rewrap needed: autograd uses local shard for
                # backward, and FSDP2 handles gradient reduction separately.
                # x = _rewrap_dtensor(x, original_dtensor)
                detached_inputs[0] = x
                if _gc_profile_enabled():
                    _gc_profile_record(
                        mode = "reentrant",
                        module_name = ctx._gc_profile_module,
                        shape = getattr(ctx, "_gc_profile_shape", shape),
                        dtype = getattr(ctx, "_gc_profile_dtype", x.dtype),
                        numel = getattr(ctx, "_gc_profile_numel", new_size),
                        kind = "unpack",
                        duration_s = time.perf_counter() - unpack_start,
                        wait_s = wait_duration,
                    )
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
            pass
        else:
            torch.autograd.backward(outputs_with_grad, args_with_grad)
        pass

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


def _unsloth_checkpoint_nonreentrant(function, *args, **kwargs):
    """Non-reentrant checkpoint using native PyTorch checkpoint plus scoped input offload."""
    preserve = kwargs.pop("preserve_rng_state", True)
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    offload_backend = resolve_gc_offload_backend(
        kwargs.pop("offload_backend", None) or _default_offload_backend
    )
    determinism_check = kwargs.pop("determinism_check", _DEFAULT_DETERMINISM_MODE)
    debug = kwargs.pop("debug", False)

    cls = UnslothGradientCheckpointer
    determinism_check = "none"
    dtype = None
    first_arg = args[0] if args else None
    if torch.is_tensor(first_arg):
        dtype = first_arg.dtype

    offloader = cls.begin_checkpoint(dtype)
    old_checkpoint = getattr(torch.utils.checkpoint, "_old_checkpoint", None)
    original_checkpoint = old_checkpoint or torch.utils.checkpoint.checkpoint

    if _gc_disable_cpu_offload() or offload_backend in ("hooks", "hooks_prefetch"):
        token = None
        is_hooks = offload_backend in ("hooks", "hooks_prefetch")
        use_prefetch = offload_backend == "hooks_prefetch"
        if (not _gc_disable_cpu_offload()) and is_hooks:
            token = _hooks_offload_state.set({
                "offloader": offloader,
                "prefetch": use_prefetch,
            })
        try:
            # Enter UnslothOffloadActivations locally so saved_tensors_hooks are
            # active during this checkpoint region. When prepare_model_for_training
            # has already wrapped the top-level forward this is a nested (benign)
            # installation.
            if is_hooks and not _gc_disable_cpu_offload():
                with UnslothOffloadActivations(dtype=dtype):
                    return original_checkpoint(
                        function, *args,
                        use_reentrant=False,
                        preserve_rng_state=preserve,
                        context_fn=context_fn,
                        determinism_check=determinism_check,
                        debug=debug,
                        **kwargs
                    )
            return original_checkpoint(
                function, *args,
                use_reentrant=False,
                preserve_rng_state=preserve,
                context_fn=context_fn,
                determinism_check=determinism_check,
                debug=debug,
                **kwargs
            )
        finally:
            if token is not None:
                _hooks_offload_state.reset(token)

    token = _noop_offload_state.set({
        "offloader": offloader,
    })
    try:
        return original_checkpoint(
            function, *args,
            use_reentrant=False,
            preserve_rng_state=preserve,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            **kwargs
        )
    finally:
        _noop_offload_state.reset(token)


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
    # across layers and defeats FSDP2 param sharding at backward; the non-
    # reentrant Mode A/B paths recover sharding AND keep CPU offload.
    # Override with UNSLOTH_SMART_GC_FSDP2 = "off" / "disable" to keep the
    # auto-override OFF (i.e. stay on whatever use_reentrant was passed).
    if use_reentrant:
        _fsdp2_env = os.environ.get("UNSLOTH_SMART_GC_FSDP2", "auto").strip().lower()
        if _fsdp2_env not in ("off", "disable", "0", "false", "no"):
            if _is_fsdp2_module(function):
                use_reentrant = False

    token = None
    if _gc_profile_enabled():
        module_name = _gc_profile_resolve_function_module_name(function)
        if module_name is not None:
            token = _gc_profile_module.set(module_name)

    try:
        if use_reentrant:
            preserve = kwargs.pop("preserve_rng_state", True)
            if kwargs:
                raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))
            return _unsloth_checkpoint_reentrant(function, *args, preserve_rng_state=preserve)

        return _unsloth_checkpoint_nonreentrant(function, *args, **kwargs)
    finally:
        if token is not None:
            _gc_profile_module.reset(token)
pass


def patch_unsloth_smart_gradient_checkpointing(dtype = None, use_reentrant = None):
    # All Unsloth Zoo code licensed under LGPLv3
    effective_use_reentrant = bool(use_reentrant) if use_reentrant is not None else True

    if effective_use_reentrant:
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
    _patch_noop_save_inputs()
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
        for i in range(len(CPU_BUFFERS)):
            if hasattr(CPU_BUFFERS[i], "resize_"): CPU_BUFFERS[i].resize_(0)
            if type(CPU_BUFFERS) is list: CPU_BUFFERS[i] = None
        for i in range(len(GPU_BUFFERS)):
            if hasattr(GPU_BUFFERS[i], "resize_"): GPU_BUFFERS[i].resize_(0)
            if type(GPU_BUFFERS) is list: GPU_BUFFERS[i] = None
        CPU_BUFFERS = None
        GPU_BUFFERS = None
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
    _unpatch_noop_save_inputs()
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

    # Clean up freed memory
    torch.cuda.empty_cache()
    gc.collect()
pass


def unsloth_offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
    return unsloth_checkpoint(function, *args, use_reentrant = False, **kwargs)
pass

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
