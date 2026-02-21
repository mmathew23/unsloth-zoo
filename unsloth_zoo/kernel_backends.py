# Unsloth Zoo - Kernel backend selection helpers
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

from __future__ import annotations

import os
from functools import lru_cache

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_VALID_KERNEL_BACKENDS = frozenset(("unsloth", "fla"))
_VALID_ROTARY_BACKENDS = frozenset(("unsloth", "fla"))


def _normalize_backend(
    value: str | None,
    allowed: frozenset[str],
    default: str,
) -> str:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    return default


@lru_cache(maxsize = 1)
def get_kernel_backend() -> str:
    return _normalize_backend(
        os.environ.get("UNSLOTH_KERNEL_BACKEND"),
        allowed = _VALID_KERNEL_BACKENDS,
        default = "unsloth",
    )


@lru_cache(maxsize = 1)
def get_rotary_kernel_backend() -> str:
    requested = _normalize_backend(
        os.environ.get("UNSLOTH_ROTARY_KERNEL_BACKEND"),
        allowed = _VALID_ROTARY_BACKENDS,
        default = "",
    )
    if requested:
        return requested

    legacy_toggle = os.environ.get("UNSLOTH_FLA_ROTARY", "").strip().lower()
    if legacy_toggle in _TRUE_VALUES:
        return "fla"

    base_backend = get_kernel_backend()
    if base_backend in _VALID_ROTARY_BACKENDS:
        return base_backend
    return "unsloth"


def clear_kernel_backend_cache() -> None:
    get_kernel_backend.cache_clear()
    get_rotary_kernel_backend.cache_clear()


__all__ = [
    "clear_kernel_backend_cache",
    "get_kernel_backend",
    "get_rotary_kernel_backend",
]
