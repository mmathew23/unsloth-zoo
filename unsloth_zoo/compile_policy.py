# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

__all__ = [
    "UNSLOTH_COMPILE_BACKEND",
    "DIRECT_TORCH_COMPILE_SOURCE_BACKENDS",
    "_is_triton_importable",
    "_normalize_compile_backend",
    "_detect_compile_backend",
    "torch_compile_uses_direct_source",
    "get_torch_compile_decorator_source",
    "get_torch_compile_import_source",
]

import importlib
import os


def _is_triton_importable() -> bool:
    try:
        importlib.import_module("triton")
    except Exception:
        return False
    return True


def _normalize_compile_backend(backend: str | None) -> str:
    if backend is None:
        return ""
    return str(backend).strip().lower().replace("-", "_")


def _detect_compile_backend() -> str:
    explicit = _normalize_compile_backend(
        os.environ.get("UNSLOTH_TORCH_COMPILE_BACKEND", "")
    )
    if explicit:
        return explicit
    if _is_triton_importable():
        return "inductor"
    return "aot_eager"


UNSLOTH_COMPILE_BACKEND: str = _detect_compile_backend()
DIRECT_TORCH_COMPILE_SOURCE_BACKENDS = frozenset()


def torch_compile_uses_direct_source(backend = None) -> bool:
    return False


def _resolve_source_backend(backend = None) -> str:
    global_backend = _normalize_compile_backend(UNSLOTH_COMPILE_BACKEND)
    backend = _normalize_compile_backend(
        global_backend if backend is None else backend
    )
    if not backend:
        backend = global_backend
    return backend


def get_torch_compile_decorator_source(
    fullgraph = True,
    dynamic = True,
    backend = None,
    options_name = "torch_compile_options",
    wrapper_name = "_unsloth_torch_compile",
) -> str:
    _resolve_source_backend(backend)
    return f"@{wrapper_name}(fullgraph = {fullgraph}, dynamic = {dynamic})"


def get_torch_compile_import_source(
    backend = None,
    wrapper_name = "_unsloth_torch_compile",
) -> str:
    _resolve_source_backend(backend)
    return (
        "from unsloth_zoo.temporary_patches.common import "
        f"torch_compile as {wrapper_name}\n"
    )
