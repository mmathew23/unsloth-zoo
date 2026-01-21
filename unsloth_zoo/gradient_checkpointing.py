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
    "UnslothOffloadHooks",
    "unsloth_checkpoint",
    "patch_unsloth_smart_gradient_checkpointing",
    "unpatch_unsloth_smart_gradient_checkpointing",
    "reset_unsloth_gradient_checkpointing_buffers",
    "initialize_unsloth_gradient_checkpointing",
    # Legacy exports for backward compatibility
    "Unsloth_Offloaded_Gradient_Checkpointer",
    "unsloth_offloaded_gradient_checkpoint",
    "patch_unsloth_gradient_checkpointing",
    "unpatch_unsloth_gradient_checkpointing",
    "Unsloth_Gradient_Checkpointer",
    "unsloth_gradient_checkpoint",
    "patch_gradient_checkpointing",
    "unpatch_gradient_checkpointing",
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
    use_reentrant         : Optional[bool] = False,
) -> None:
    """
    Calculates where to place the gradient checkpoints given n_layers.

    All Unsloth Zoo code licensed under LGPLv3

    Args:
        model: Any LlamaModel with layers.
        layers_per_checkpoint (`Union[str, int]`, *optional*):
            Can either be `sqrt` or an integer for how many layers per checkpoint you want.
            The more, the less memory usage, but can be slower. Default is `sqrt`.
            Choose 1 for Pytorch gradient checkpointing. 2 to wrap 2 layers in 1 module etc.
        use_reentrant (`bool`, *optional*):
            This parameter is kept for API compatibility but Unsloth always uses
            non-reentrant checkpointing for better performance and compatibility.
            Default is False (non-reentrant).
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

    # Always use non-reentrant for better performance and compatibility
    use_reentrant = False

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


class UnslothReentrantCheckpointFunction(torch.autograd.Function):
    """
    All Unsloth Zoo code licensed under LGPLv3

    Reentrant gradient checkpointing with smart CPU offloading.
    This is the original implementation that provides optimal memory usage
    by running forward under torch.no_grad() and offloading hidden_states to CPU.
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        ctx._requires_gradient = False
        use_gpu_buffer = False

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                if i == 0 and arg.requires_grad:
                    global FIRST_PASS
                    global LAST_GC_INDEX
                    if FIRST_PASS:
                        LAST_GC_INDEX += 1
                    pass
                    global CURRENT_GC_INDEX
                    CURRENT_GC_INDEX += 1

                    ctx._requires_gradient = True
                    new_size = arg.numel()

                    global MINIMUM_SIZE
                    global CPU_INDEX
                    if new_size > MINIMUM_SIZE and ((CURRENT_GC_INDEX != LAST_GC_INDEX) or FIRST_PASS):
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

                        if BACKWARD_PASS:
                            BACKWARD_PASS = False
                            CPU_INDEX = 0
                        pass

                        if CPU_INDEX >= len(CPU_BUFFERS):
                            x = torch.empty(new_size, dtype = arg.dtype, device = "cpu", pin_memory = True)
                            CPU_BUFFERS.append(x)
                        pass

                        x = CPU_BUFFERS[CPU_INDEX]
                        shape = arg.shape
                        if new_size > x.numel(): x.resize_(new_size)
                        if new_size > GPU_BUFFER.numel(): GPU_BUFFER.resize_(new_size)
                        x = x[:new_size].view(shape)

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

        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        new_size, shape, CPU_INDEX, device_index, MAIN_STREAM, EXTRA_STREAM = ctx._saved_metadata
        if CPU_INDEX is not None:
            global GPU_BUFFERS
            buffer = GPU_BUFFERS[device_index][:new_size].view(shape)
            x = CPU_BUFFERS[CPU_INDEX][:new_size].view(shape)

            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                buffer.copy_(x, non_blocking = True)
        else:
            if len(tensor_indices) != 0:
                inputs[tensor_indices[0]] = tensors[0]
        pass

        for i, idx in enumerate(tensor_indices[1:], start = 1):
            inputs[idx] = tensors[i]
        pass

        global BACKWARD_PASS
        BACKWARD_PASS = True
        global FIRST_PASS
        FIRST_PASS = False
        global CURRENT_GC_INDEX
        CURRENT_GC_INDEX = 0

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

            if CPU_INDEX is not None:
                MAIN_STREAM.wait_stream(EXTRA_STREAM)
                x = buffer.detach()
                x.requires_grad_(True)
                detached_inputs[0] = x
            pass

            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)
            pass
        pass

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

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
        for i in range(len(detached_inputs)):
            detached_inputs[i] = None
            inputs[i] = None
        pass

        return (None, None) + grads
    pass
pass


class UnslothOffloadHooks:
    """
    All Unsloth Zoo code licensed under LGPLv3
    CPU offloading via saved_tensors_hooks for non-reentrant checkpointing.

    This class implements smart CPU offloading using PyTorch's saved_tensors_hooks
    mechanism, which is compatible with non-reentrant gradient checkpointing.

    IMPORTANT: Only ONE tensor per checkpoint region is offloaded (the first large
    tensor, which is typically the hidden_states). This matches the behavior of
    the original reentrant implementation for optimal memory usage.
    """

    def __init__(self, min_size, dtype, device_index=0, cpu_buffer_index=0):
        """
        Args:
            min_size: Minimum tensor size (in elements) to trigger offloading
            dtype: Tensor dtype for CPU buffers
            device_index: GPU device index for stream management
            cpu_buffer_index: Index into CPU_BUFFERS for this checkpoint region
        """
        self.min_size = min_size
        self.dtype = dtype
        self.device_index = device_index
        self.cpu_buffer_index = cpu_buffer_index
        self.offload_data = None  # Store (shape, device, original_dtype) for the one offloaded tensor
        self.has_offloaded = False  # Only offload ONE tensor per checkpoint region

    def pack_hook(self, tensor):
        """Called when saving tensor for backward - offload only the first large tensor to CPU"""
        # All Unsloth Zoo code licensed under LGPLv3

        # Only offload ONE tensor per checkpoint region (like the old implementation)
        if self.has_offloaded:
            return tensor

        # Skip small tensors
        if tensor.numel() < self.min_size:
            return tensor

        # Skip tensors on CPU
        if tensor.device.type == "cpu":
            return tensor

        # Debug: uncomment to see what tensors are being considered
        if os.environ.get("UNSLOTH_DEBUG_GC", "0") == "1":
            print(f"[DEBUG pack_hook] OFFLOADING tensor shape={tensor.shape}, numel={tensor.numel()}, requires_grad={tensor.requires_grad}")

        global EXTRA_STREAMS
        global MAIN_STREAMS
        global CPU_BUFFERS
        global CPU_INDEX
        global USE_UNSLOTH_GC

        device = tensor.device
        device_index = device.index if device.index is not None else 0
        new_size = tensor.numel()

        # Print message once
        if USE_UNSLOTH_GC:
            print("Unsloth: Will smartly offload gradients to save VRAM!")
            USE_UNSLOTH_GC = False

        # Get streams for async transfer
        if EXTRA_STREAMS is not None and device_index < len(EXTRA_STREAMS):
            MAIN_STREAM = MAIN_STREAMS[device_index]
            EXTRA_STREAM = EXTRA_STREAMS[device_index]
        else:
            MAIN_STREAM = None
            EXTRA_STREAM = None

        # Reuse CPU buffer from pre-allocated pool (like old implementation)
        if CPU_BUFFERS is not None and self.cpu_buffer_index < len(CPU_BUFFERS):
            cpu_buffer = CPU_BUFFERS[self.cpu_buffer_index]
            if new_size > cpu_buffer.numel():
                cpu_buffer.resize_(new_size)
            cpu_tensor = cpu_buffer[:new_size]
        else:
            # Extend buffer pool if needed
            cpu_tensor = torch.empty(new_size, dtype=self.dtype, device="cpu", pin_memory=True)
            if CPU_BUFFERS is not None:
                CPU_BUFFERS.append(cpu_tensor)

        # Use streams for async copy if available
        if EXTRA_STREAM is not None:
            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                cpu_tensor.copy_(tensor.view(-1), non_blocking=True)
            MAIN_STREAM.wait_stream(EXTRA_STREAM)
        else:
            cpu_tensor.copy_(tensor.view(-1), non_blocking=True)

        # Store metadata for unpack
        self.offload_data = (tensor.shape, tensor.device, tensor.dtype, new_size)
        self.has_offloaded = True

        # Return a marker that this tensor was offloaded
        return "OFFLOADED"

    def unpack_hook(self, data):
        """Called when restoring tensor for backward - fetch from CPU"""
        # All Unsloth Zoo code licensed under LGPLv3
        if data != "OFFLOADED":
            return data

        shape, device, original_dtype, new_size = self.offload_data
        device_index = device.index if device.index is not None else 0

        global EXTRA_STREAMS
        global MAIN_STREAMS
        global GPU_BUFFERS
        global CPU_BUFFERS

        # Get streams for async transfer
        if EXTRA_STREAMS is not None and device_index < len(EXTRA_STREAMS):
            MAIN_STREAM = MAIN_STREAMS[device_index]
            EXTRA_STREAM = EXTRA_STREAMS[device_index]
        else:
            MAIN_STREAM = None
            EXTRA_STREAM = None

        # Get CPU buffer
        cpu_tensor = CPU_BUFFERS[self.cpu_buffer_index][:new_size]

        # Use GPU buffer if available (like old implementation)
        if GPU_BUFFERS is not None and device_index < len(GPU_BUFFERS):
            buffer = GPU_BUFFERS[device_index]
            if new_size > buffer.numel():
                buffer.resize_(new_size)
            gpu_tensor = buffer[:new_size]
        else:
            gpu_tensor = torch.empty(new_size, dtype=self.dtype, device=device)

        # Use streams for async copy if available
        if EXTRA_STREAM is not None:
            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            MAIN_STREAM.wait_stream(EXTRA_STREAM)
        else:
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)

        # Reshape and cast back to original dtype if needed
        result = gpu_tensor.view(shape)
        if result.dtype != original_dtype:
            result = result.to(original_dtype)
        return result
pass


from torch.utils.checkpoint import (
    ContextManager,
    _DEFAULT_DETERMINISM_MODE,
    _checkpoint_without_reentrant_generator,
    noop_context_fn,
)


class _UnslothOffloadedCheckpointFunction(torch.autograd.Function):
    """
    All Unsloth Zoo code licensed under LGPLv3

    Non-reentrant gradient checkpointing with smart CPU offloading.
    Similar to the reentrant approach but compatible with DDP.

    Key differences from reentrant:
    - Does NOT use torch.no_grad() during forward (records autograd graph)
    - But achieves similar memory by offloading input to CPU
    """

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, run_function, preserve_rng_state, cpu_buffer_index, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        global CPU_BUFFERS, GPU_BUFFERS, EXTRA_STREAMS, MAIN_STREAMS, USE_UNSLOTH_GC

        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(ctx.device_type)

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Handle first argument (hidden_states) specially - offload to CPU if cpu_buffer_index >= 0
        # Note: args may be empty if all arguments are passed as kwargs (e.g., vision models)
        hidden_states = args[0] if len(args) > 0 else None
        ctx._requires_gradient = hidden_states.requires_grad if torch.is_tensor(hidden_states) else False
        should_offload = cpu_buffer_index >= 0 and hidden_states is not None

        if ctx._requires_gradient and torch.is_tensor(hidden_states) and should_offload:
            device = hidden_states.device
            device_index = device.index if device.index is not None else 0
            new_size = hidden_states.numel()
            shape = hidden_states.shape

            # Print message once
            if USE_UNSLOTH_GC:
                print("Unsloth: Will smartly offload gradients to save VRAM!")
                USE_UNSLOTH_GC = False

            MAIN_STREAM = MAIN_STREAMS[device_index]
            EXTRA_STREAM = EXTRA_STREAMS[device_index]
            GPU_BUFFER = GPU_BUFFERS[device_index]

            # Ensure buffers are large enough
            cpu_buffer = CPU_BUFFERS[cpu_buffer_index]
            if new_size > cpu_buffer.numel():
                cpu_buffer.resize_(new_size)
            if new_size > GPU_BUFFER.numel():
                GPU_BUFFER.resize_(new_size)

            # Async copy to CPU
            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                cpu_buffer[:new_size].view(shape).copy_(hidden_states, non_blocking=True)

            ctx._saved_metadata = (new_size, shape, cpu_buffer_index, device_index, MAIN_STREAM, EXTRA_STREAM)

            # Save other tensor args normally, but NOT hidden_states
            ctx.inputs = []
            ctx.tensor_indices = []
            tensor_inputs = []
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    ctx.tensor_indices.append(i)
                    if i == 0:
                        # Don't save hidden_states - we offloaded it
                        tensor_inputs.append(None)
                    else:
                        tensor_inputs.append(arg)
                    ctx.inputs.append(None)
                else:
                    ctx.inputs.append(arg)
            ctx.save_for_backward(*[t for t in tensor_inputs if t is not None])

            # Wait for copy to complete
            MAIN_STREAM.wait_stream(EXTRA_STREAM)
        else:
            ctx._saved_metadata = (None, None, None, None, None, None)
            # Save all tensors normally (no offloading)
            ctx.inputs = []
            ctx.tensor_indices = []
            tensor_inputs = []
            for i, arg in enumerate(args):
                if torch.is_tensor(arg):
                    ctx.tensor_indices.append(i)
                    tensor_inputs.append(arg)
                    ctx.inputs.append(None)
                else:
                    ctx.inputs.append(arg)
            ctx.save_for_backward(*tensor_inputs)

        # Run forward under no_grad - this is key for memory efficiency
        # Intermediate activations are NOT saved, only the input is saved (to CPU)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, *args):
        # All Unsloth Zoo code licensed under LGPLv3
        global CPU_BUFFERS, GPU_BUFFERS, BACKWARD_PASS, FIRST_PASS, CURRENT_GC_INDEX

        # Note: Even if _requires_gradient=False (no positional tensor args),
        # we still need to recompute because the function may have tensors
        # that need gradients bound in kwargs (e.g., vision models with partial)

        # Reset state
        BACKWARD_PASS = True
        FIRST_PASS = False
        CURRENT_GC_INDEX = 0

        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        saved_tensors = ctx.saved_tensors

        new_size, shape, cpu_buffer_index, device_index, MAIN_STREAM, EXTRA_STREAM = ctx._saved_metadata

        # Restore hidden_states from CPU if it was offloaded
        if cpu_buffer_index is not None:
            buffer = GPU_BUFFERS[device_index][:new_size].view(shape)
            cpu_data = CPU_BUFFERS[cpu_buffer_index][:new_size].view(shape)

            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                buffer.copy_(cpu_data, non_blocking=True)
            MAIN_STREAM.wait_stream(EXTRA_STREAM)

            # Fill in other saved tensors
            saved_idx = 0
            for i, idx in enumerate(tensor_indices):
                if idx == 0:
                    continue  # Skip hidden_states, handled separately
                inputs[idx] = saved_tensors[saved_idx]
                saved_idx += 1
        else:
            # No offloading - fill in all saved tensors
            for i, idx in enumerate(tensor_indices):
                inputs[idx] = saved_tensors[i]

        # Restore RNG state
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device_type):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states, device_type=ctx.device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=ctx.device_type, **ctx.device_autocast_kwargs
            ) if torch.amp.is_autocast_available(ctx.device_type) else contextlib.nullcontext()

            # Detach inputs
            detached_inputs = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    detached_inputs.append(inp)
                else:
                    x = inp.detach()
                    x.requires_grad = inp.requires_grad
                    detached_inputs.append(x)

            # Set up hidden_states from GPU buffer
            if cpu_buffer_index is not None:
                x = buffer.detach()
                x.requires_grad_(True)
                detached_inputs[0] = x

            # Recompute forward
            with torch.enable_grad(), device_autocast_ctx, torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # Backward
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])

        if outputs_with_grad:
            torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        # Clean up
        for i in range(len(detached_inputs)):
            detached_inputs[i] = None
            inputs[i] = None

        return (None, None, None) + grads


@torch._disable_dynamo
def unsloth_checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    r"""Checkpoint a model or part of the model with smart CPU offloading.

    All Unsloth Zoo code licensed under LGPLv3

    This is Unsloth's optimized gradient checkpointing implementation.
    By default uses non-reentrant style (compatible with DDP) with smart
    CPU offloading for memory efficiency.

    Args:
        function: The function to checkpoint.
        preserve_rng_state(bool, optional): Default: ``True``
        use_reentrant(bool): If True, uses reentrant style. Default: True for
            optimal memory, but set False for DDP compatibility.
        args: Inputs to the function.

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    preserve = kwargs.pop("preserve_rng_state", True)

    # Debug: confirm this function is being called
    debug_gc = os.environ.get("UNSLOTH_DEBUG_GC", "0") == "1"
    if debug_gc:
        print(f"[DEBUG unsloth_checkpoint] CALLED with {len(args)} args")


    global MINIMUM_SIZE, CPU_BUFFERS, CPU_INDEX, BACKWARD_PASS
    global FIRST_PASS, LAST_GC_INDEX, CURRENT_GC_INDEX

    # Determine dtype from first arg
    dtype = None
    first_arg = args[0] if args else None
    if torch.is_tensor(first_arg):
        dtype = first_arg.dtype

    # Initialize buffers if needed
    if CPU_BUFFERS is None or len(CPU_BUFFERS) == 0:
        initialize_unsloth_gradient_checkpointing(dtype)

    # Determine if we should offload
    should_offload = False
    current_cpu_index = 0

    if (torch.is_tensor(first_arg) and
        first_arg.requires_grad and
        first_arg.numel() > MINIMUM_SIZE):

        if FIRST_PASS:
            LAST_GC_INDEX += 1
        CURRENT_GC_INDEX += 1

        if BACKWARD_PASS:
            BACKWARD_PASS = False
            CPU_INDEX = 0

        # Skip last layer optimization
        if (CURRENT_GC_INDEX != LAST_GC_INDEX) or FIRST_PASS:
            should_offload = True
            current_cpu_index = CPU_INDEX
            CPU_INDEX += 1

            if current_cpu_index >= len(CPU_BUFFERS):
                CPU_BUFFERS.append(torch.empty(
                    first_arg.numel(), dtype=dtype, device="cpu", pin_memory=True
                ))

    debug_gc = os.environ.get("UNSLOTH_DEBUG_GC", "0") == "1"

    # Always use our offloading implementation for consistency
    # The should_offload logic determines IF we actually copy to CPU, but we always
    # use the same code path
    if debug_gc:
        print(f"[DEBUG] Using _UnslothOffloadedCheckpointFunction, should_offload={should_offload}, cpu_buffer_index={current_cpu_index}")

    return _UnslothOffloadedCheckpointFunction.apply(
        function, preserve, current_cpu_index if should_offload else -1, *args
    )
pass


def patch_unsloth_smart_gradient_checkpointing(dtype = None):
    """
    All Unsloth Zoo code licensed under LGPLv3

    Patches torch.utils.checkpoint.checkpoint to use Unsloth's optimized
    checkpointing with smart CPU offloading.
    """
    # Initialize buffers for CPU offloading
    initialize_unsloth_gradient_checkpointing(dtype)

    # Patch CheckpointFunction for reentrant mode (used by default in transformers)
    if torch.utils.checkpoint.CheckpointFunction.__name__ != "UnslothReentrantCheckpointFunction":
        torch.utils.checkpoint._old_CheckpointFunction = torch.utils.checkpoint.CheckpointFunction
        torch.utils.checkpoint.CheckpointFunction = UnslothReentrantCheckpointFunction

    # Patch torch.utils.checkpoint.checkpoint
    if torch.utils.checkpoint.checkpoint.__name__ != "unsloth_checkpoint":
        torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
        torch.utils.checkpoint.checkpoint = unsloth_checkpoint

    # Also patch the checkpoint reference in transformers.modeling_utils
    # This is needed because transformers imports checkpoint at module load time
    # and gradient_checkpointing_enable uses that captured reference
    try:
        import transformers.modeling_utils
        if hasattr(transformers.modeling_utils, 'checkpoint'):
            if transformers.modeling_utils.checkpoint.__name__ != "unsloth_checkpoint":
                transformers.modeling_utils._old_checkpoint = transformers.modeling_utils.checkpoint
                transformers.modeling_utils.checkpoint = unsloth_checkpoint
    except:
        pass
pass


def unpatch_unsloth_smart_gradient_checkpointing():
    """
    All Unsloth Zoo code licensed under LGPLv3

    Restores the original torch.utils.checkpoint.checkpoint function and
    cleans up CPU/GPU buffers.
    """
    # Clean up buffers
    global CPU_BUFFERS
    global GPU_BUFFERS
    if CPU_BUFFERS is not None:
        for i in range(len(CPU_BUFFERS)):
            if CPU_BUFFERS[i] is not None and hasattr(CPU_BUFFERS[i], "resize_"):
                CPU_BUFFERS[i].resize_(0)
            if type(CPU_BUFFERS) is list:
                CPU_BUFFERS[i] = None
        CPU_BUFFERS = None
    if GPU_BUFFERS is not None:
        for i in range(len(GPU_BUFFERS)):
            if GPU_BUFFERS[i] is not None and hasattr(GPU_BUFFERS[i], "resize_"):
                GPU_BUFFERS[i].resize_(0)
            if type(GPU_BUFFERS) is list:
                GPU_BUFFERS[i] = None
        GPU_BUFFERS = None
    torch.cuda.empty_cache()
    gc.collect()

    if (torch.utils.checkpoint.checkpoint.__name__ == "unsloth_checkpoint") and \
        hasattr(torch.utils.checkpoint, "_old_checkpoint"):

        torch.utils.checkpoint.checkpoint = torch.utils.checkpoint._old_checkpoint
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
    """
    All Unsloth Zoo code licensed under LGPLv3

    Legacy wrapper that redirects to unsloth_checkpoint.
    This function is kept for backward compatibility.
    """
    return unsloth_checkpoint(function, *args, use_reentrant=False, **kwargs)
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
