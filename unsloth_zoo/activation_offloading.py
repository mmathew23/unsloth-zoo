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

"""Activation-offloading helpers.

TRL's `OffloadActivations` installs PyTorch saved-tensor-default-hooks. Some
Unsloth fused autograd paths call `torch.func.{grad, vjp, jacrev, hessian}`,
which reject ANY active saved-tensor hooks (rejection is on presence, not
behavior — no-op hooks still fail). `maybe_disable_trl_activation_offloading`
returns a context manager that pops any active hooks for the duration of the
wrapped region and restores them on exit.

Debugging
---------
Set ``UNSLOTH_AO_DEBUG=1`` to print one line each time the wrapper pops and
restores hooks, e.g. during `unsloth_fused_ce_loss` forward.
"""

import contextlib
import os

import torch


# Capability check for the private saved-tensors-default-hooks stack API.
# These underscore-prefixed symbols have been stable across torch 2.x, but we
# still detect at import so a future rename fails at module import with a
# clear message rather than deep inside backward.
_AG = getattr(torch._C, "_autograd", None)
_HOOK_STACK_API_AVAILABLE = _AG is not None and all(
    hasattr(_AG, name) for name in (
        "_top_saved_tensors_default_hooks",
        "_pop_saved_tensors_default_hooks",
        "_push_saved_tensors_default_hooks",
    )
)

_AO_DEBUG = os.environ.get("UNSLOTH_AO_DEBUG", "") in ("1", "true", "True")


@contextlib.contextmanager
def _pop_default_hooks_temporarily():
    """Pop all active saved-tensors-default-hooks for the body, then re-push.

    `torch.autograd.graph.disable_saved_tensors_hooks` only flips a flag and
    does not pop already-pushed hooks, so it is insufficient when
    `OffloadActivations` is currently active on the stack.
    """
    ag = torch._C._autograd
    popped: list = []
    try:
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
        if _AO_DEBUG:
            print(f"[ao_disable] popped {len(popped)} hook(s)", flush=True)
        yield
    finally:
        for pack, unpack in reversed(popped):
            ag._push_saved_tensors_default_hooks(pack, unpack)
        if _AO_DEBUG:
            print(f"[ao_disable] restored {len(popped)} hook(s)", flush=True)


def maybe_disable_trl_activation_offloading(_trainer=None):
    """Return a context manager that disables active saved-tensor hooks.

    Parameters
    ----------
    _trainer
        Ignored. Accepted for backward compatibility with existing call sites
        (e.g. `unsloth_fused_ce_loss(trainer=...)` in unsloth's llama/mistral
        modules, which hardcode `trainer=None`). The wrapper keys off live
        hook-stack state, not this arg, so passing ``None`` is fine.

    Returns
    -------
    contextlib.AbstractContextManager
        - `nullcontext` when no default hooks are pushed, or when the private
          torch hook-stack API is unavailable (so AO is effectively a no-op).
        - A pop-and-restore context manager when hooks are active.

    Raises
    ------
    RuntimeError
        If hooks are active but the private hook-stack API is unavailable on
        this torch build — we cannot safely run `torch.func` under active
        hooks, and silently returning `nullcontext` would crash the backward
        pass with a confusing "torch.func don't yet support saved tensor
        hooks" error deep in autograd.
    """
    del _trainer  # explicit "I know this is unused"
    if not _HOOK_STACK_API_AVAILABLE:
        # No way to inspect the stack. Assume empty and no-op. If hooks ARE
        # active, the caller's torch.func region will raise with its own
        # clear message — we don't want to double-raise here and break users
        # who aren't using activation offloading at all.
        return contextlib.nullcontext()

    try:
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except RuntimeError:
        top = None
    if top is None:
        return contextlib.nullcontext()
    return _pop_default_hooks_temporarily()
