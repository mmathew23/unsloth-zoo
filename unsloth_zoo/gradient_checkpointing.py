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


class UnslothOffloadHooks:
    """
    All Unsloth Zoo code licensed under LGPLv3
    CPU offloading via saved_tensors_hooks for non-reentrant checkpointing.

    This class implements smart CPU offloading using PyTorch's saved_tensors_hooks
    mechanism, which is compatible with non-reentrant gradient checkpointing.
    Large tensors are automatically offloaded to pinned CPU memory during the
    forward pass and restored during the backward pass using async transfers.
    """

    def __init__(self, min_size, dtype, device_index=0):
        """
        Args:
            min_size: Minimum tensor size (in elements) to trigger offloading
            dtype: Tensor dtype for CPU buffers
            device_index: GPU device index for stream management
        """
        self.min_size = min_size
        self.dtype = dtype
        self.device_index = device_index
        self.cpu_cache = {}  # Maps key -> (cpu_tensor, shape, device, original_dtype)
        self.counter = 0
        self._printed_message = False

    def pack_hook(self, tensor):
        """Called when saving tensor for backward - offload large tensors to CPU"""
        # All Unsloth Zoo code licensed under LGPLv3
        if tensor.numel() < self.min_size:
            return tensor

        global EXTRA_STREAMS
        global MAIN_STREAMS

        device = tensor.device
        device_index = device.index if device.index is not None else 0

        # Print message once
        global USE_UNSLOTH_GC
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

        # Create pinned CPU tensor and async copy
        cpu_tensor = torch.empty(
            tensor.numel(),
            dtype=self.dtype if self.dtype is not None else tensor.dtype,
            device="cpu",
            pin_memory=True
        )

        # Use streams for async copy if available
        if EXTRA_STREAM is not None:
            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                cpu_tensor.copy_(tensor.view(-1), non_blocking=True)
            MAIN_STREAM.wait_stream(EXTRA_STREAM)
        else:
            cpu_tensor.copy_(tensor.view(-1), non_blocking=True)

        key = self.counter
        self.counter += 1
        self.cpu_cache[key] = (cpu_tensor, tensor.shape, tensor.device, tensor.dtype)
        return key

    def unpack_hook(self, data):
        """Called when restoring tensor for backward - fetch from CPU"""
        # All Unsloth Zoo code licensed under LGPLv3
        if not isinstance(data, int):
            return data

        cpu_tensor, shape, device, original_dtype = self.cpu_cache.pop(data)
        device_index = device.index if device.index is not None else 0

        global EXTRA_STREAMS
        global MAIN_STREAMS
        global GPU_BUFFERS

        # Get streams for async transfer
        if EXTRA_STREAMS is not None and device_index < len(EXTRA_STREAMS):
            MAIN_STREAM = MAIN_STREAMS[device_index]
            EXTRA_STREAM = EXTRA_STREAMS[device_index]
        else:
            MAIN_STREAM = None
            EXTRA_STREAM = None

        # Use GPU buffer if available, otherwise allocate new tensor
        if GPU_BUFFERS is not None and device_index < len(GPU_BUFFERS):
            buffer = GPU_BUFFERS[device_index]
            if cpu_tensor.numel() > buffer.numel():
                buffer.resize_(cpu_tensor.numel())
            gpu_tensor = buffer[:cpu_tensor.numel()]
        else:
            gpu_tensor = torch.empty(
                cpu_tensor.numel(),
                dtype=self.dtype if self.dtype is not None else original_dtype,
                device=device
            )

        # Use streams for async copy if available
        if EXTRA_STREAM is not None:
            EXTRA_STREAM.wait_stream(MAIN_STREAM)
            with torch_gpu_stream(EXTRA_STREAM):
                gpu_tensor.copy_(cpu_tensor, non_blocking=True)
            MAIN_STREAM.wait_stream(EXTRA_STREAM)
        else:
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)

        # Cast back to original dtype if needed and reshape
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

    This is Unsloth's optimized gradient checkpointing implementation that uses
    non-reentrant checkpointing with smart CPU offloading via saved_tensors_hooks.
    Large activation tensors are automatically offloaded to pinned CPU memory
    during the forward pass and restored during backward, reducing GPU VRAM usage.

    The non-reentrant implementation provides several advantages:
    - Early stopping: Recomputation stops as soon as all needed activations are computed
    - Full backward support: Works with torch.autograd.grad and .backward(inputs=...)
    - Better torch.compile compatibility
    - No requirement for inputs to have requires_grad=True

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple.
        preserve_rng_state(bool, optional): Omit stashing and restoring
            the RNG state during each checkpoint. Default: ``True``
        use_reentrant(bool): This parameter is ignored. Unsloth always uses
            non-reentrant checkpointing for better performance and compatibility.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. Default is ``"default"``.
        debug(bool, optional): If ``True``, error messages will include
            a trace of the operators ran during the original forward computation.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    # All Unsloth Zoo code licensed under LGPLv3
    # Always use non-reentrant for better performance and compatibility
    # The use_reentrant parameter is kept for API compatibility but ignored

    preserve = kwargs.pop("preserve_rng_state", True)

    # Determine dtype from args for CPU buffer allocation
    dtype = None
    device_index = 0
    for arg in args:
        if torch.is_tensor(arg):
            dtype = arg.dtype
            if arg.device.index is not None:
                device_index = arg.device.index
            break

    global MINIMUM_SIZE
    global CPU_BUFFERS

    # Initialize buffers if not already done
    if CPU_BUFFERS is None or len(CPU_BUFFERS) == 0:
        initialize_unsloth_gradient_checkpointing(dtype)

    # Create offload hooks for CPU offloading of large tensors
    hooks = UnslothOffloadHooks(
        min_size=MINIMUM_SIZE if MINIMUM_SIZE is not None else 2 * 1024 * 1024 // 2,  # 2MB default for bfloat16
        dtype=dtype,
        device_index=device_index
    )

    # Use saved_tensors_hooks to intercept tensor saves for CPU offloading
    with torch.autograd.graph.saved_tensors_hooks(hooks.pack_hook, hooks.unpack_hook):
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            pass
    return ret
pass


def patch_unsloth_smart_gradient_checkpointing(dtype = None):
    """
    All Unsloth Zoo code licensed under LGPLv3

    Patches torch.utils.checkpoint.checkpoint to use Unsloth's optimized
    non-reentrant checkpointing with smart CPU offloading.
    """
    # Initialize buffers for CPU offloading
    initialize_unsloth_gradient_checkpointing(dtype)

    if torch.utils.checkpoint.checkpoint.__name__ != "unsloth_checkpoint":
        torch.utils.checkpoint._old_checkpoint = torch.utils.checkpoint.checkpoint
        torch.utils.checkpoint.checkpoint = unsloth_checkpoint
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
