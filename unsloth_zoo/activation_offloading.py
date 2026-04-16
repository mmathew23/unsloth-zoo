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

import contextlib

_TRL_AO_NOOP_MANAGER_CLS = None  # None = unchecked, False = unavailable, or the class


def _get_noop_manager():
    global _TRL_AO_NOOP_MANAGER_CLS
    if _TRL_AO_NOOP_MANAGER_CLS is not None:
        return _TRL_AO_NOOP_MANAGER_CLS
    try:
        from trl.models.activation_offloading import NoOpManager
        _TRL_AO_NOOP_MANAGER_CLS = NoOpManager
    except Exception:
        _TRL_AO_NOOP_MANAGER_CLS = False
    return _TRL_AO_NOOP_MANAGER_CLS


@contextlib.contextmanager
def _pop_default_hooks_temporarily():
    """Pop ALL active saved_tensors_default_hooks for the body, then re-push.

    `torch.func.{grad, vjp, jacrev, hessian}` rejects ANY active saved-tensor
    hooks. `disable_saved_tensors_hooks` only flips a flag and does not pop
    already-pushed hooks, so it is insufficient when OffloadActivations is
    currently active on the stack.
    """
    import os
    import torch
    ag = torch._C._autograd
    debug = os.environ.get("UNSLOTH_AO_DEBUG", "") == "1"
    popped: list = []
    try:
        # Drain the entire stack so torch.func sees no active hooks.
        while True:
            try:
                top = ag._top_saved_tensors_default_hooks(False)
            except RuntimeError:
                break
            if top is None:
                break
            pack, unpack = top
            try:
                ag._pop_saved_tensors_default_hooks()
            except RuntimeError:
                break
            popped.append((pack, unpack))
        if debug:
            print(f"[ao_disable] popped {len(popped)} hook(s)", flush=True)
        yield
    finally:
        # Re-push in reverse order to restore original stack ordering.
        for pack, unpack in reversed(popped):
            ag._push_saved_tensors_default_hooks(pack, unpack)
        if debug:
            print(f"[ao_disable] restored {len(popped)} hook(s)", flush=True)


def maybe_disable_trl_activation_offloading(trainer):
    """
    Some Unsloth fused autograd paths call `torch.func.grad_and_value`, which
    does not support any active saved-tensor hooks (the rejection is on
    presence, not behavior — even no-op hooks fail). The branch wires this
    wrapper into call sites that pass `trainer=None` (e.g. llama.py:1549),
    so the trainer arg is unreliable. We instead key off the live state of
    the saved-tensor-hook stack: if anything is pushed, pop it for the body
    and restore it after. When the stack is empty (AO disabled, or nothing
    active for any other reason), this is a no-op.
    """
    import torch
    try:
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except RuntimeError:
        top = None
    if top is None:
        return contextlib.nullcontext()
    return _pop_default_hooks_temporarily()
