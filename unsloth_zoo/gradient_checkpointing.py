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
from typing import Union, Optional, List, Any, Callable, Tuple
import os
import warnings
import gc
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
]

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


# @torch._disable_dynamo
# def unsloth_offloaded_gradient_checkpoint(function, *args, use_reentrant = None, **kwargs):
#     return Unsloth_Offloaded_Gradient_Checkpointer.apply(function, *args)
# pass


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
    check_backward_validity,
    _infer_device_type,
    _get_autocast_kwargs,
    _get_device_module,
    get_device_states,
    # set_device_states,
    detach_variable,
    contextlib,
    DefaultDeviceType,
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
UNSLOTH_GC_PATCH_USE_REENTRANT = True
UNSLOTH_GC_NONREENTRANT_BACKEND = "hooks"
UNSLOTH_GC_DEBUG_PRINTED = set()
ORIGINAL_NOOP_SETUP_CONTEXT = None
UNSLOTH_NOOP_OFFLOAD_STATE = None


def _patch_noop_save_inputs():
    global ORIGINAL_NOOP_SETUP_CONTEXT
    cls = getattr(torch.utils.checkpoint, "_NoopSaveInputs", None)
    if cls is None:
        return
    if ORIGINAL_NOOP_SETUP_CONTEXT is not None:
        return
    ORIGINAL_NOOP_SETUP_CONTEXT = cls.setup_context

    def unsloth_setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
        state = UNSLOTH_NOOP_OFFLOAD_STATE
        if state is None:
            return ORIGINAL_NOOP_SETUP_CONTEXT(ctx, inputs, output)

        offloader = state.get("offloader", None)
        target_input_index = int(state.get("target_input_index", 2))
        if offloader is None:
            return ORIGINAL_NOOP_SETUP_CONTEXT(ctx, inputs, output)

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

            should_try_offload = (
                i == target_input_index and
                o.requires_grad and
                o.device.type != "cpu"
            )
            if should_try_offload:
                packed = offloader.pack_hook(o)
                if isinstance(packed, tuple) and len(packed) >= 1 and packed[0] in ("cpu", "engine"):
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
                    ret[i - 1] = offloader.unpack_hook(offloaded[entry_index[i]])
            return ret

        ctx.get_args = get_args
        ctx.save_for_backward(*saved_tensors)

    cls.setup_context = staticmethod(unsloth_setup_context)
pass


def _unpatch_noop_save_inputs():
    global ORIGINAL_NOOP_SETUP_CONTEXT
    global UNSLOTH_NOOP_OFFLOAD_STATE
    cls = getattr(torch.utils.checkpoint, "_NoopSaveInputs", None)
    if cls is None:
        return
    if ORIGINAL_NOOP_SETUP_CONTEXT is None:
        return
    cls.setup_context = ORIGINAL_NOOP_SETUP_CONTEXT
    ORIGINAL_NOOP_SETUP_CONTEXT = None
    UNSLOTH_NOOP_OFFLOAD_STATE = None
pass


def _is_truthy_env(name):
    value = os.environ.get(name, None)
    if value is None:
        return False
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")
pass


def _gc_debug(tag, message, once = True):
    if not _is_truthy_env("UNSLOTH_ENABLE_LOGGING"):
        return
    key = (tag, message)
    if once and key in UNSLOTH_GC_DEBUG_PRINTED:
        return
    if once:
        UNSLOTH_GC_DEBUG_PRINTED.add(key)
    print(f"[UNSLOTH_GC][{tag}] {message}")
pass


def _gc_disable_cpu_offload():
    return _is_truthy_env("UNSLOTH_GC_DISABLE_CPU_OFFLOAD")
pass


def _gc_async_mode():
    # Modes:
    # - off   : single-stream safe transport.
    # - pack  : async D2H pack only; restore remains on consumer stream.
    # - full  : async D2H + async H2D on extra stream.
    mode = os.environ.get("UNSLOTH_GC_ASYNC_MODE", "off")
    mode = str(mode).strip().lower()
    if mode in ("off", "none", "0", "false", "no"):
        return "off"
    if mode in ("full", "all", "2"):
        return "full"
    return "pack"
pass


def _parse_nonreentrant_backend(backend):
    if backend is None:
        backend = os.environ.get("UNSLOTH_GC_NONREENTRANT_BACKEND", "hooks")
    backend = str(backend).strip().lower()
    if backend in ("hooks", "saved_tensors_hooks"):
        return "hooks"
    if backend in ("save_on_cpu", "nohooks", "no_hooks"):
        return "save_on_cpu"
    if backend in ("engine", "custom_engine", "nohooks_engine"):
        return "engine"
    if backend in ("engine_v2", "v2_engine", "custom_engine_v2"):
        return "engine_v2"
    raise ValueError("`nonreentrant_backend` must be one of ['hooks', 'save_on_cpu', 'engine', 'engine_v2'].")
pass


def _resolve_nonreentrant_determinism_mode(determinism_check):
    if determinism_check != _DEFAULT_DETERMINISM_MODE:
        return determinism_check
    mode = os.environ.get("UNSLOTH_GC_NONREENTRANT_DETERMINISM_CHECK", "none")
    mode = str(mode).strip().lower()
    if mode in ("default", "none"):
        return mode
    raise ValueError("`UNSLOTH_GC_NONREENTRANT_DETERMINISM_CHECK` must be one of ['default', 'none'].")
pass


def _parse_nonreentrant_offload_policy(prefix):
    policy = os.environ.get(f"{prefix}_POLICY", "auto")
    policy = str(policy).strip().lower()
    if policy in ("auto", "always", "never"):
        return policy
    raise ValueError(f"`{prefix}_POLICY` must be one of ['auto', 'always', 'never'].")
pass


def _nonreentrant_adaptive_offload_should_offload(tensor, prefix):
    policy = _parse_nonreentrant_offload_policy(prefix)
    if policy == "always":
        return True, "policy=always"
    if policy == "never":
        return False, "policy=never"

    # Match reentrant offload behavior: 2MB activation threshold by default.
    min_bytes = int(os.environ.get(f"{prefix}_MIN_BYTES", str(2 * 1024 * 1024)))
    if tensor is None:
        return False, "auto:no_tensor"
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "is_compiling"):
        if torch.compiler.is_compiling():
            return False, "auto:torch_compile_disable_offload_gate"
    try:
        tensor_nbytes = int(getattr(tensor, "nbytes", 0))
    except Exception:
        # Symbolic shapes under torch.compile can make nbytes/numel unavailable.
        # In that case, skip offload for safety rather than raising.
        return False, "auto:symbolic_nbytes_unavailable"
    if tensor_nbytes < min_bytes:
        return False, f"auto:tensor_nbytes<{min_bytes}"
    if DEVICE_TYPE not in ("cuda", "hip"):
        return True, "auto:non_cuda_device"

    # Memory-pressure gating is intentionally disabled for now.
    # In auto mode, once tensor size passes the threshold, offload it.
    return True, f"auto:size_gate_only nbytes={tensor_nbytes}"
pass


def _nonreentrant_save_on_cpu_should_offload(tensor):
    return _nonreentrant_adaptive_offload_should_offload(
        tensor,
        "UNSLOTH_GC_NONREENTRANT_SAVE_ON_CPU",
    )
pass


def _nonreentrant_hooks_should_offload(tensor):
    return _nonreentrant_adaptive_offload_should_offload(
        tensor,
        "UNSLOTH_GC_NONREENTRANT_HOOKS_OFFLOAD",
    )
pass


def _maybe_compose_selective_ac_context_fn(context_fn):
    # Keep SAC plumbing by forwarding the context function through
    # non-reentrant checkpoint calls unchanged.
    return context_fn
pass


@contextlib.contextmanager
def _nonreentrant_perf_context():
    set_early_stop = getattr(torch.utils.checkpoint, "set_checkpoint_early_stop", None)
    if set_early_stop is None:
        yield
        return
    with set_early_stop(True):
        yield
pass


class UnslothGradientCheckpointer:
    """
    All Unsloth Zoo code licensed under LGPLv3

    Non-reentrant gradient checkpointing with smart CPU offloading.
    """
    _cpu_buffers: List[torch.Tensor] = []
    _cpu_free_buffers: dict = {}
    _gpu_buffers: dict = {}
    _main_streams: dict = {}
    _extra_streams: dict = {}
    _initialized: bool = False

    _cpu_buffer_index: int = 0
    _current_gc_index: int = 0
    _last_gc_index: int = 0
    _first_pass: bool = True
    _backward_pass: bool = True
    _minimum_size: int = 2 * 1024 * 1024 // 2
    _use_unsloth_gc_message: bool = True
    _dtype: torch.dtype = None
    _events_supported: Optional[bool] = None
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
        if not cls._initialized:
            return

        cls._cpu_buffer_index = 0
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
        cls._main_streams = {}
        cls._extra_streams = {}
        cls._initialized = False

    def __init__(self, is_last_layer: bool = False):
        self.offloaded_tensors = {}
        self.pack_counter = 0
        self.is_last_layer = is_last_layer

    @classmethod
    def _record_stream_event(cls, stream):
        if cls._events_supported is False:
            return None
        try:
            event = stream.record_event()
            cls._events_supported = True
            return event
        except Exception as e:
            _gc_debug(
                "NONREENTRANT_EVENT",
                f"stream.record_event failed; falling back to wait_stream path. error={type(e).__name__}: {e}",
            )
            cls._events_supported = False
            return None

    @classmethod
    def _wait_event(cls, stream, event):
        if event is None:
            return False
        try:
            stream.wait_event(event)
            return True
        except Exception as e:
            _gc_debug(
                "NONREENTRANT_EVENT",
                f"stream.wait_event failed; falling back to wait_stream path. error={type(e).__name__}: {e}",
            )
            cls._events_supported = False
            return False

    def should_offload(self, tensor: torch.Tensor) -> bool:
        cls = self.__class__
        if _gc_disable_cpu_offload():
            return False
        if tensor.numel() < cls._minimum_size:
            return False
        if tensor.device.type == "cpu":
            return False
        if self.is_last_layer:
            return False
        # Custom packed-buffer restore currently materializes contiguous tensors.
        # Restrict offload to safe tensor layouts to preserve correctness.
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

    @classmethod
    def _acquire_cpu_buffer(cls, *, numel: int, dtype: torch.dtype, device_index: int) -> torch.Tensor:
        pool = cls._cpu_free_buffers.setdefault(dtype, [])
        chosen_idx = None
        chosen_buf = None
        for i in range(len(pool) - 1, -1, -1):
            buf, fence_device_index, fence_event = pool[i]
            ready = True
            if fence_event is not None:
                try:
                    ready = bool(fence_event.query())
                except Exception:
                    ready = True
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
            if chosen_buf.numel() < numel:
                chosen_buf = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)
            return chosen_buf

        return torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True)

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

    def pack_hook(self, tensor: torch.Tensor):
        cls = self.__class__
        if not self.should_offload(tensor):
            return ("gpu", tensor)

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

        cpu_buffer = cls._acquire_cpu_buffer(
            numel=numel,
            dtype=dtype,
            device_index=device_index,
        )

        offload_event = None
        async_mode = _gc_async_mode()
        if async_mode in ("pack", "full"):
            if DEVICE_TYPE in ("cuda", "hip"):
                main_stream = torch.cuda.current_stream(device)
            elif DEVICE_TYPE == "xpu":
                main_stream = torch.xpu.current_stream(device)
            else:
                main_stream = cls._main_streams[device_index]
            extra_stream = cls._extra_streams[device_index]
            extra_stream.wait_stream(main_stream)
            with torch_gpu_stream(extra_stream):
                # Ensure allocator does not recycle `tensor` storage until this stream
                # has finished consuming it for the async D2H copy.
                try:
                    tensor.record_stream(extra_stream)
                except Exception:
                    pass
                cpu_buffer[:numel].view(shape).copy_(tensor, non_blocking=True)
                offload_event = cls._record_stream_event(extra_stream)
        else:
            cpu_buffer[:numel].view(shape).copy_(tensor, non_blocking=True)

        pack_id = self.pack_counter
        self.pack_counter += 1
        self.offloaded_tensors[pack_id] = (
            shape,
            stride,
            dtype,
            requires_grad,
            device_index,
            numel,
            cpu_buffer,
            offload_event,
        )
        return ("cpu", pack_id)

    def unpack_hook(self, packed):
        cls = self.__class__
        cls._backward_pass = True
        cls._first_pass = False

        if packed[0] == "gpu":
            return packed[1]

        _, pack_id = packed
        (
            shape,
            original_stride,
            original_dtype,
            original_requires_grad,
            device_index,
            numel,
            cpu_buffer,
            offload_event,
        ) = self.offloaded_tensors[pack_id]

        async_mode = _gc_async_mode()
        if async_mode == "full":
            device = torch.device(f"{DEVICE_TYPE_TORCH}:{device_index}")
            if DEVICE_TYPE in ("cuda", "hip"):
                main_stream = torch.cuda.current_stream(device)
            elif DEVICE_TYPE == "xpu":
                main_stream = torch.xpu.current_stream(device)
            else:
                main_stream = cls._main_streams[device_index]
            extra_stream = cls._extra_streams[device_index]
            with torch_gpu_stream(extra_stream):
                if not cls._wait_event(extra_stream, offload_event):
                    extra_stream.wait_stream(main_stream)
                result = cpu_buffer[:numel].view(shape).to(
                    device = f"{DEVICE_TYPE_TORCH}:{device_index}",
                    non_blocking = True,
                )
                if tuple(result.stride()) != tuple(original_stride):
                    result = result.as_strided(shape, original_stride)
                restore_event = cls._record_stream_event(extra_stream)
            if not cls._wait_event(main_stream, restore_event):
                main_stream.wait_stream(extra_stream)
        else:
            # For "pack" mode, ensure D2H completion happened before using CPU buffer.
            if offload_event is not None and DEVICE_TYPE in ("cuda", "hip"):
                torch.cuda.current_stream(torch.device(f"{DEVICE_TYPE_TORCH}:{device_index}")).wait_event(offload_event)
            elif offload_event is not None and DEVICE_TYPE == "xpu":
                torch.xpu.current_stream(torch.device(f"{DEVICE_TYPE_TORCH}:{device_index}")).wait_event(offload_event)
            result = cpu_buffer[:numel].view(shape).to(
                device = f"{DEVICE_TYPE_TORCH}:{device_index}",
                non_blocking = True,
            )
            if tuple(result.stride()) != tuple(original_stride):
                result = result.as_strided(shape, original_stride)
            restore_event = None
        cls._release_cpu_buffer(
            cpu_buffer=cpu_buffer,
            dtype=original_dtype,
            device_index=device_index,
            restore_event=restore_event,
        )

        if result.dtype != original_dtype:
            result = result.to(original_dtype)
        if result.requires_grad != original_requires_grad:
            result.requires_grad_(original_requires_grad)
        return result


class UnslothNonReentrantEngineOffloader:
    """
    Hook-free non-reentrant offload controller using _NoopSaveInputs interception.
    This keeps per-checkpoint ownership centralized in torch checkpoint internals.
    """
    def __init__(self, is_last_layer: bool = False):
        self._base = UnslothGradientCheckpointer(is_last_layer=is_last_layer)

    def pack_hook(self, tensor: torch.Tensor):
        # _NoopSaveInputs gives deterministic per-checkpoint ownership/order
        # of packed entries.
        return self._base.pack_hook(tensor)

    def unpack_hook(self, packed):
        return self._base.unpack_hook(packed)

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
        # check_backward_validity(args)
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
                    new_size = arg.numel()

                    global MINIMUM_SIZE
                    global CPU_INDEX
                    if (not disable_cpu_offload) and new_size > MINIMUM_SIZE and ((CURRENT_GC_INDEX != LAST_GC_INDEX) or FIRST_PASS):
                        use_gpu_buffer = True
                        global CPU_BUFFERS
                        global GPU_BUFFERS
                        global BACKWARD_PASS
                        global EXTRA_STREAMS
                        global MAIN_STREAMS
                        device = arg.device
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
                            x = torch.empty(new_size, dtype = arg.dtype, device = "cpu", pin_memory = True)
                            CPU_BUFFERS.append(x)
                        pass

                        x = CPU_BUFFERS[CPU_INDEX]
                        shape = arg.shape
                        if new_size > x.numel(): x.resize_(new_size)
                        if new_size > GPU_BUFFER.numel(): GPU_BUFFER.resize_(new_size)
                        x = x[:new_size].view(shape)

                        # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
                        EXTRA_STREAM.wait_stream(MAIN_STREAM)
                        with torch_gpu_stream(EXTRA_STREAM):
                            x.copy_(arg, non_blocking = True)

                        ctx._saved_metadata = (new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM,)
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
            global GPU_BUFFER
            buffer = GPU_BUFFERS[device_index][:new_size].view(shape)
            x = CPU_BUFFERS[CPU_INDEX][:new_size].view(shape)

            # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
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

            # detached_inputs = detach_variable(tuple(inputs))
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
            pass
            # raise RuntimeError(
            #     "none of output has requires_grad=True,"
            #     " this checkpoint() is not necessary"
            # )
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


from torch.utils.checkpoint import (
    ContextManager,
    _DEFAULT_DETERMINISM_MODE,
    noop_context_fn,
)
def _unsloth_checkpoint_impl(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.5 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    global UNSLOTH_NOOP_OFFLOAD_STATE
    if use_reentrant is None:
        global UNSLOTH_GC_PATCH_USE_REENTRANT
        use_reentrant = UNSLOTH_GC_PATCH_USE_REENTRANT
        if use_reentrant:
            warnings.warn(
                "torch.utils.checkpoint: the use_reentrant parameter should be "
                "passed explicitly. In version 2.5 we will raise an exception "
                "if use_reentrant is not passed. use_reentrant=False is "
                "recommended, but if you need to preserve the current default "
                "behavior, you can pass use_reentrant=True. Refer to docs for more "
                "details on the differences between the two variants.",
                stacklevel=2
            )

    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    if use_reentrant:
        _gc_debug(
            "REENTRANT_PATH",
            "unsloth_checkpoint -> UnslothCheckpointFunction.apply (reentrant autograd path)",
        )
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return UnslothCheckpointFunction.apply(function, preserve, *args)

    cls = UnslothGradientCheckpointer
    determinism_check = _resolve_nonreentrant_determinism_mode(determinism_check)
    context_fn = _maybe_compose_selective_ac_context_fn(context_fn)
    global UNSLOTH_GC_NONREENTRANT_BACKEND
    backend = UNSLOTH_GC_NONREENTRANT_BACKEND
    _gc_debug(
        "NONREENTRANT_PATH",
        f"unsloth_checkpoint -> torch checkpoint(use_reentrant=False), backend={backend}",
        once = False,
    )
    dtype = None
    first_arg = args[0] if args else None
    if torch.is_tensor(first_arg):
        dtype = first_arg.dtype

    if backend in ("hooks", "engine", "engine_v2") and not cls._initialized:
        cls.initialize(dtype)
    elif backend != "hooks" and not cls._meta_initialized:
        cls.ensure_metadata(dtype)

    if cls._backward_pass:
        cls._backward_pass = False
        cls._cpu_buffer_index = 0
        cls._current_gc_index = 0

    if cls._first_pass:
        cls._last_gc_index += 1
    cls._current_gc_index += 1

    is_last_layer = (cls._current_gc_index == cls._last_gc_index) and not cls._first_pass
    should_offload = (
        (not _gc_disable_cpu_offload()) and
        torch.is_tensor(first_arg) and
        first_arg.requires_grad and
        first_arg.numel() > cls._minimum_size and
        (not is_last_layer)
    )

    if backend == "engine":
        offloader = UnslothNonReentrantEngineOffloader(is_last_layer=is_last_layer)
    elif backend == "engine_v2":
        offloader = None
    else:
        offloader = UnslothGradientCheckpointer(is_last_layer=is_last_layer)
    old_checkpoint = getattr(torch.utils.checkpoint, "_old_checkpoint", None)
    original_checkpoint = old_checkpoint if old_checkpoint is not None else torch.utils.checkpoint.checkpoint

    if backend == "engine_v2":
        _gc_debug(
            "NONREENTRANT_ENGINE_V2_PATH",
            "engine_v2 backend -> torch checkpoint(use_reentrant=False), optimized non-reentrant path",
            once = False,
        )
        # Fast path: when offload is disabled, bypass all offload bookkeeping.
        if _gc_disable_cpu_offload():
            with _nonreentrant_perf_context():
                return original_checkpoint(
                    function, *args,
                    use_reentrant=False,
                    preserve_rng_state=preserve,
                    context_fn=context_fn,
                    determinism_check=determinism_check,
                    debug=debug,
                    **kwargs
                )

        allow_offload_v2, offload_reason_v2 = _nonreentrant_save_on_cpu_should_offload(
            first_arg if torch.is_tensor(first_arg) else None
        )
        should_offload_v2 = should_offload and allow_offload_v2
        if should_offload_v2:
            _gc_debug(
                "NONREENTRANT_ENGINE_V2",
                f"engine_v2 offload enabled (optimized _NoopSaveInputs path), reason={offload_reason_v2}",
                once=False,
            )
            offloader_v2 = UnslothNonReentrantEngineOffloader(is_last_layer=is_last_layer)
            previous_state = UNSLOTH_NOOP_OFFLOAD_STATE
            UNSLOTH_NOOP_OFFLOAD_STATE = {
                "offloader": offloader_v2,
                "target_input_index": 2,
            }
            try:
                with _nonreentrant_perf_context():
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
                UNSLOTH_NOOP_OFFLOAD_STATE = previous_state
        else:
            _gc_debug(
                "NONREENTRANT_ENGINE_V2",
                f"engine_v2 offload skipped, reason={offload_reason_v2}",
                once=False,
            )
            with _nonreentrant_perf_context():
                return original_checkpoint(
                    function, *args,
                    use_reentrant=False,
                    preserve_rng_state=preserve,
                    context_fn=context_fn,
                    determinism_check=determinism_check,
                    debug=debug,
                    **kwargs
                )

    if backend in ("save_on_cpu", "engine"):
        allow_offload, offload_reason = _nonreentrant_save_on_cpu_should_offload(first_arg if torch.is_tensor(first_arg) else None)
        should_offload_save_on_cpu = should_offload and (not _gc_disable_cpu_offload()) and allow_offload
        if should_offload_save_on_cpu:
            backend_tag = "NONREENTRANT_ENGINE" if backend == "engine" else "NONREENTRANT_SAVE_ON_CPU"
            backend_msg = "engine backend enabled (ticketed _NoopSaveInputs offload)" if backend == "engine" else "save_on_cpu backend enabled (hook-free _NoopSaveInputs offload)"
            _gc_debug(
                backend_tag,
                f"{backend_msg}, reason={offload_reason}",
                once = False,
            )
            previous_state = UNSLOTH_NOOP_OFFLOAD_STATE
            UNSLOTH_NOOP_OFFLOAD_STATE = {
                "offloader": offloader,
                # _NoopSaveInputs gets: (dummy, kwargs, *args), so first model arg is index 2.
                "target_input_index": 2,
            }
            try:
                with _nonreentrant_perf_context():
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
                UNSLOTH_NOOP_OFFLOAD_STATE = previous_state
        else:
            backend_tag = "NONREENTRANT_ENGINE" if backend == "engine" else "NONREENTRANT_SAVE_ON_CPU"
            backend_msg = "engine backend skipped offload (below threshold / CPU tensor / last layer / no grad / disabled / policy)" if backend == "engine" else "save_on_cpu backend skipped offload (below threshold / CPU tensor / last layer / no grad / disabled / policy)"
            _gc_debug(
                backend_tag,
                f"{backend_msg} reason={offload_reason}",
                once = False,
            )
            with _nonreentrant_perf_context():
                return original_checkpoint(
                    function, *args,
                    use_reentrant=False,
                    preserve_rng_state=preserve,
                    context_fn=context_fn,
                    determinism_check=determinism_check,
                    debug=debug,
                    **kwargs
                )

    allow_hooks_offload, hooks_offload_reason = _nonreentrant_hooks_should_offload(first_arg if torch.is_tensor(first_arg) else None)
    should_offload_hooks = should_offload and allow_hooks_offload
    if should_offload_hooks:
        _gc_debug(
            "NONREENTRANT_OFFLOAD",
            f"non-reentrant checkpoint OFFLOAD enabled for this activation (hooks backend), reason={hooks_offload_reason}",
            once = False,
        )
        with torch.autograd.graph.saved_tensors_hooks(offloader.pack_hook, offloader.unpack_hook):
            with _nonreentrant_perf_context():
                return original_checkpoint(
                    function, *args,
                    use_reentrant=False,
                    preserve_rng_state=preserve,
                    context_fn=context_fn,
                    determinism_check=determinism_check,
                    debug=debug,
                    **kwargs
                )

    _gc_debug(
        "NONREENTRANT_OFFLOAD",
        f"non-reentrant checkpoint OFFLOAD skipped (below threshold / CPU tensor / last layer / no grad / policy), reason={hooks_offload_reason}",
        once = False,
    )
    with _nonreentrant_perf_context():
        return original_checkpoint(
            function, *args,
            use_reentrant=False,
            preserve_rng_state=preserve,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            **kwargs
        )
pass


@torch._disable_dynamo
def _unsloth_checkpoint_nodynamo(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    return _unsloth_checkpoint_impl(
        function,
        *args,
        use_reentrant=use_reentrant,
        context_fn=context_fn,
        determinism_check=determinism_check,
        debug=debug,
        **kwargs,
    )
pass


def _should_allow_dynamo_nonreentrant(use_reentrant: Optional[bool]) -> bool:
    if not _is_truthy_env("UNSLOTH_GC_ALLOW_DYNAMO_NONREENTRANT"):
        return False
    if use_reentrant is None:
        effective_use_reentrant = UNSLOTH_GC_PATCH_USE_REENTRANT
    else:
        effective_use_reentrant = bool(use_reentrant)
    return effective_use_reentrant is False
pass


def unsloth_checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    if _should_allow_dynamo_nonreentrant(use_reentrant):
        _gc_debug(
            "NONREENTRANT_COMPILE_EXPERIMENT",
            "Running non-reentrant checkpoint without torch._disable_dynamo",
            once=False,
        )
        return _unsloth_checkpoint_impl(
            function,
            *args,
            use_reentrant=use_reentrant,
            context_fn=context_fn,
            determinism_check=determinism_check,
            debug=debug,
            **kwargs,
        )
    return _unsloth_checkpoint_nodynamo(
        function,
        *args,
        use_reentrant=use_reentrant,
        context_fn=context_fn,
        determinism_check=determinism_check,
        debug=debug,
        **kwargs,
    )
pass


def _parse_reentrant_mode(use_reentrant):
    if use_reentrant is not None:
        if type(use_reentrant) is not bool:
            raise TypeError("`use_reentrant` must be a boolean or None.")
        return use_reentrant

    env_value = os.environ.get("UNSLOTH_GC_USE_REENTRANT", None)
    if env_value is None:
        return True
    env_value = str(env_value).strip().lower()
    return env_value not in ("0", "false", "no", "off")
pass


def patch_unsloth_smart_gradient_checkpointing(dtype = None, use_reentrant = None, nonreentrant_backend = None):
    # All Unsloth Zoo code licensed under LGPLv3
    global UNSLOTH_GC_PATCH_USE_REENTRANT
    global UNSLOTH_GC_NONREENTRANT_BACKEND
    UNSLOTH_GC_PATCH_USE_REENTRANT = _parse_reentrant_mode(use_reentrant)
    UNSLOTH_GC_NONREENTRANT_BACKEND = _parse_nonreentrant_backend(nonreentrant_backend)
    _gc_debug(
        "PATCH_MODE",
        f"patch_unsloth_smart_gradient_checkpointing selected use_reentrant={UNSLOTH_GC_PATCH_USE_REENTRANT}, nonreentrant_backend={UNSLOTH_GC_NONREENTRANT_BACKEND}",
        once = False,
    )

    if UNSLOTH_GC_PATCH_USE_REENTRANT:
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
    global UNSLOTH_GC_PATCH_USE_REENTRANT
    global UNSLOTH_GC_NONREENTRANT_BACKEND
    UNSLOTH_GC_PATCH_USE_REENTRANT = True
    UNSLOTH_GC_NONREENTRANT_BACKEND = "hooks"
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


@torch._disable_dynamo
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
