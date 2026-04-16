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


def maybe_disable_trl_activation_offloading(trainer):
    """
    Some Unsloth fused autograd paths call `torch.func.grad_and_value`, which
    does not support active saved tensor hooks. When TRL activation offloading
    is enabled, temporarily install TRL's own `NoOpManager` so those regions
    can run while leaving activation offloading active for the rest of the
    model step.
    """
    if trainer is None:
        return contextlib.nullcontext()

    args = getattr(trainer, "args", None)
    if args is None or not getattr(args, "activation_offloading", False):
        return contextlib.nullcontext()

    cls = _get_noop_manager()
    if cls is False:
        return contextlib.nullcontext()
    return cls()
