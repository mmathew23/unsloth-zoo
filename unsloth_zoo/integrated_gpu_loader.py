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

"""Integrated GPU loader patch for unified-memory hosts.

Wraps two transformers symbols when running on an integrated-memory GPU
(NVIDIA GB10 / Spark, where ``torch.cuda.get_device_properties(0).is_integrated == 1``)
and transformers >= ``_MIN_VERSION``:

Design constraints:
  * Patch ONLY when the hardware/env/version gates pass.
  * NEVER fall back to the original loader after possible model mutation.
  * Stream ONLY pre-quantized checkpoints.
  * DECLINE streaming when weight converters require cross-shard fusion.

Override the auto-detection with ``UNSLOTH_INTEGRATED_GPU_LOADER=1``
(force on) or ``=0`` (force off).
"""

from __future__ import annotations

import functools
import gc
import inspect
import logging
import os
import re
import threading
from collections import ChainMap
from typing import Any

logger = logging.getLogger(__name__)


_PATCH_FLAG_ATTR = "_unsloth_integrated_loader_patched"

# Serialise patch installs across threads. The body of
# apply_integrated_gpu_loader_patches reads-then-mutates module-level
# transformers attributes (Patch A on accel_int / mu, Patch B on
# PreTrainedModel); without a lock two concurrent calls can both pass
# the _is_unsloth_routed check and double-wrap.
_INSTALL_LOCK = threading.Lock()

# Reset the install lock in any forked child so a fork() that happens
# while a parent thread held _INSTALL_LOCK can't deadlock the child.
# multiprocessing's "fork" start method (Linux default through 3.13)
# inherits lock state across the fork boundary; the holder thread does
# not exist in the child, so the lock would otherwise stay locked
# forever.
if hasattr(os, "register_at_fork"):

    def _reset_install_lock_after_fork() -> None:
        global _INSTALL_LOCK
        _INSTALL_LOCK = threading.Lock()

    os.register_at_fork(after_in_child=_reset_install_lock_after_fork)


# 4.57.x and earlier streams shards correctly and doesn't have the
# device_map regression. The relevant load-path rewrite lands in 5.x;
# the early 5.x line saw frequent churn in the symbols we depend on, so
# in practice the patch only activates on >= 5.5 (5.0.x..5.4.x will
# typically fail _check_symbols and decline). Keep the floor at 5.0 so
# the symbol check is the source of truth, not a string compare.
_MIN_VERSION = "5.0.0"


# ---------------------------------------------------------------------------
# Compatibility contract with transformers internals.
# These constants document the surface area we monkey-patch against;
# `_check_symbols` validates each one at install time.
# ---------------------------------------------------------------------------

_LOAD_PRETRAINED_MODEL_PARAMS = (
    "model",
    "state_dict",
    "checkpoint_files",
    "load_config",
    "expected_keys",
)

_CONVERT_AND_LOAD_REQUIRED_KWARGS = frozenset(
    {
        "model",
        "state_dict",
        "load_config",
        "tp_plan",
        "disk_offload_index",
    }
)

_GET_DEVICE_MAP_REQUIRED_KWARGS = frozenset(
    {
        "model",
        "device_map",
        "max_memory",
        "hf_quantizer",
    }
)

_LOAD_STATE_DICT_INFO_FIELDS = frozenset(
    {
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
        "conversion_errors",
    }
)

_LOAD_STATE_DICT_CONFIG_FIELDS = frozenset(
    {
        "device_map",
        "disable_mmap",
        "weights_only",
        "hf_quantizer",
        "weight_mapping",
        "sharded_metadata",
        "disk_offload_folder",
        "dtype",
    }
)

_ACCELERATE_DISK_OFFLOAD_REQUIRED_KWARGS = frozenset(
    {
        "model",
        "disk_offload_folder",
        "checkpoint_files",
        "device_map",
        "sharded_metadata",
        "dtype",
        "weight_mapping",
    }
)

_STRING_DEVICE_MAPS = ("auto", "sequential", "balanced", "balanced_low_0")


# ---------------------------------------------------------------------------
# Hardware + environment detection
# ---------------------------------------------------------------------------


_TRUTHY_OVERRIDES = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSY_OVERRIDES = frozenset({"0", "false", "no", "off", "n", "f"})

# One-shot tracker for the unrecognised-value warning so a typo'd
# override doesn't spam logs once per from_pretrained.
_UNRECOGNISED_OVERRIDES_WARNED: set[str] = set()


def _integrated_gpu_override() -> bool | None:
    """Parse ``UNSLOTH_INTEGRATED_GPU_LOADER``. Returns ``True`` (force on),
    ``False`` (force off), or ``None`` (no override). Accepts the usual
    truthy/falsy spellings (``true``/``false``/``yes``/``no``/``on``/``off``,
    case-insensitive); unrecognised non-empty values warn once per
    distinct value and fall through to auto-detect.
    """
    raw = os.environ.get("UNSLOTH_INTEGRATED_GPU_LOADER", "")
    value = raw.strip().lower()
    if not value:
        return None
    if value in _TRUTHY_OVERRIDES:
        return True
    if value in _FALSY_OVERRIDES:
        return False
    if value not in _UNRECOGNISED_OVERRIDES_WARNED:
        _UNRECOGNISED_OVERRIDES_WARNED.add(value)
        logger.warning(
            "Unsloth: UNSLOTH_INTEGRATED_GPU_LOADER=%r not recognised; "
            "ignoring (auto-detect will run). Accepted values: %s.",
            raw,
            sorted(_TRUTHY_OVERRIDES | _FALSY_OVERRIDES),
        )
    return None


def _detect_integrated_gpu(index: int = 0) -> bool:
    """Hardware-only probe: True iff CUDA device ``index`` reports
    ``is_integrated == 1`` (NVIDIA GB10 / Spark). Ignores the env override.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(index)
        return getattr(props, "is_integrated", 0) == 1
    except Exception:
        return False


def _is_integrated_gpu(index: int = 0) -> bool:
    override = _integrated_gpu_override()
    if override is not None:
        return override
    return _detect_integrated_gpu(index)


def _transformers_at_least_min() -> tuple[bool, str]:
    try:
        from importlib.metadata import version

        from packaging.version import Version

        v = Version(version("transformers"))
    except Exception as exc:
        return False, f"version probe failed: {exc!r}"
    if v < Version(_MIN_VERSION):
        return False, f"transformers={v} < {_MIN_VERSION}"
    return True, ""


# ---------------------------------------------------------------------------
# Symbol-shape probes (composed by `_check_symbols`)
# ---------------------------------------------------------------------------


def _signature_has_leading_params(fn, expected: tuple[str, ...]) -> tuple[bool, str]:
    """``fn``'s first ``len(expected)`` params must equal ``expected`` in
    order, AND every param after that must have a default. We re-call
    ``fn`` positionally with ``len(expected)`` args, so trailing
    optional params are harmless but trailing required params would be
    silently dropped.

    ``expected`` must be non-empty -- an empty tuple would degenerate
    into "every param must have a default", which is not a useful gate.
    Raised as ``ValueError`` (not ``assert``) so ``python -O`` cannot
    strip the contract check.
    """
    if not expected:
        raise ValueError("expected must be non-empty")

    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError) as exc:
        return False, f"{fn!r} not introspectable: {exc!r}"

    actual = tuple(p.name for p in params)
    if actual[: len(expected)] != expected:
        return False, (
            f"leading params changed: got {actual[: len(expected)]}, "
            f"expected {expected} (full signature: {actual})"
        )
    for p in params[len(expected) :]:
        if p.default is inspect.Parameter.empty and p.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return False, (
                f"trailing required param {p.name!r} would be dropped "
                f"by our positional call (full signature: {actual})"
            )
    return True, ""


def _signature_accepts_kwargs(fn, required: frozenset[str]) -> tuple[bool, str]:
    """``fn`` must be callable with exactly the keyword arguments in
    ``required``.

    Three failure modes, each fail-closed:
      * a required positional-only parameter (cannot be passed by name);
      * a required name we'd pass that ``fn`` doesn't have AND ``fn``
        has no ``**kwargs`` to absorb it;
      * an additional REQUIRED parameter on ``fn`` that we don't pass
        (positional-or-keyword or keyword-only with no default). A
        ``**kwargs`` parameter does NOT silence this check, since extras
        in **kwargs cannot satisfy a required named parameter.

    Positional-only parameters with defaults are tolerated (they may
    legitimately exist alongside our keyword call as long as they're
    optional).
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError) as exc:
        return False, f"{fn!r} not introspectable: {exc!r}"

    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    # 1) positional-only blockers: required names that cannot be passed
    #    by name, AND positional-only params with no default that we'd
    #    skip past entirely with our keyword call.
    positional_only_blockers = {
        name
        for name, param in params.items()
        if param.kind == inspect.Parameter.POSITIONAL_ONLY
        and (name in required or param.default is inspect.Parameter.empty)
    }
    if positional_only_blockers:
        return False, (
            f"positional-only params incompatible with keyword call "
            f"{positional_only_blockers} (got {tuple(params.keys())})"
        )

    # 2) named-presence: every required name must be findable, unless
    #    **kwargs is present to absorb it.
    if not has_varkw:
        missing = required - set(params)
        if missing:
            return False, (
                f"missing keyword-compatible params {missing} "
                f"(got {tuple(params.keys())})"
            )

    # 3) extra-required: positional-or-keyword / keyword-only params
    #    without defaults that we don't pass. **kwargs does NOT cover
    #    these -- they are required NAMES.
    extra_required = {
        name
        for name, param in params.items()
        if name not in required
        and param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    if extra_required:
        return False, (
            f"has additional required params {extra_required} "
            f"that this patch does not pass"
        )

    return True, ""


def _dataclass_has_fields(cls, required: frozenset[str]) -> tuple[bool, str]:
    fields_attr = getattr(cls, "__dataclass_fields__", None)
    if fields_attr is None:
        return False, f"{cls.__name__} is no longer a dataclass"
    actual = set(fields_attr.keys())
    missing = required - actual
    if missing:
        return False, f"{cls.__name__} missing fields {missing} (got {actual})"
    return True, ""


def _check_symbols() -> tuple[bool, str]:
    try:
        from safetensors import safe_open  # noqa: F401
        from transformers.core_model_loading import (  # noqa: F401
            Concatenate,
            MergeModulelist,
            WeightConverter,
            convert_and_load_state_dict_in_model,
        )
        from transformers.integrations.accelerate import _get_device_map
        from transformers.modeling_utils import (  # noqa: F401  install-time existence probes
            LoadStateDictConfig,
            PreTrainedModel,
            _is_on_hf_mount,
            _load_state_dict_into_zero3_model,
            _safe_load_bytes,
            accelerate_disk_offload,
            is_deepspeed_zero3_enabled,
            load_state_dict,
        )
        from transformers.quantizers.base import HfQuantizer
        from transformers.utils.loading_report import LoadStateDictInfo
        from transformers.utils.quantization_config import (  # noqa: F401
            QuantizationMethod,
        )
    except Exception as e:
        return False, f"import failed: {e!r}"

    checks: list[tuple[bool, str]] = [
        _signature_has_leading_params(
            PreTrainedModel._load_pretrained_model, _LOAD_PRETRAINED_MODEL_PARAMS
        ),
        _signature_accepts_kwargs(
            convert_and_load_state_dict_in_model, _CONVERT_AND_LOAD_REQUIRED_KWARGS
        ),
        _signature_accepts_kwargs(_get_device_map, _GET_DEVICE_MAP_REQUIRED_KWARGS),
        _signature_accepts_kwargs(
            accelerate_disk_offload, _ACCELERATE_DISK_OFFLOAD_REQUIRED_KWARGS
        ),
        _dataclass_has_fields(LoadStateDictInfo, _LOAD_STATE_DICT_INFO_FIELDS),
        _dataclass_has_fields(LoadStateDictConfig, _LOAD_STATE_DICT_CONFIG_FIELDS),
    ]
    for ok, reason in checks:
        if not ok:
            return False, reason

    # Property + attribute checks that don't fit the helpers above.
    if not hasattr(LoadStateDictConfig, "is_quantized"):
        return False, "LoadStateDictConfig has no is_quantized property"
    if not isinstance(PreTrainedModel.__dict__.get("tp_plan"), property):
        return False, "PreTrainedModel.tp_plan is no longer a property"
    # Attribute names read at runtime by helpers in this module. A
    # silent rename here would otherwise just fail-closed on the
    # affected code path with no install-time signal.
    for attr in ("source_patterns", "operations"):
        if not hasattr(WeightConverter, attr):
            return False, f"WeightConverter has no {attr!r} attribute"
    # ``pre_quantized`` is an instance attribute set in
    # ``HfQuantizer.__init__`` (``self.pre_quantized = kwargs.pop(...)``),
    # so it is not visible on the class. Probe the __init__ source for
    # the LHS assignment specifically -- a plain substring would also
    # match the kwargs.pop key, so ``self.X = kwargs.pop("pre_quantized")``
    # with a renamed attribute X would pass and silently disable
    # streaming. The regex matches ``self.pre_quantized =`` only.
    try:
        src = inspect.getsource(HfQuantizer.__init__)
    except (OSError, TypeError) as exc:
        return False, f"could not inspect HfQuantizer.__init__: {exc!r}"
    if not re.search(r"self\.pre_quantized\s*=", src):
        return False, "HfQuantizer.__init__ no longer assigns self.pre_quantized"

    return True, ""


def _should_patch() -> bool:
    """Binary decision: install the patches, or leave transformers alone.

    Declines on discrete GPUs, on transformers < ``_MIN_VERSION``, and on
    any signature/symbol mismatch from ``_check_symbols``. Mismatches log
    a WARNING so future upstream churn is visible at import time.
    """
    if not _is_integrated_gpu():
        return False
    ok, reason = _transformers_at_least_min()
    if not ok:
        logger.debug("Unsloth: integrated_gpu_loader will NOT patch (%s).", reason)
        return False
    ok, reason = _check_symbols()
    if not ok:
        logger.warning(
            "Unsloth: integrated_gpu_loader will NOT patch this transformers "
            "version because %s. Original loader will run unchanged.",
            reason,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _has_disk_offload(device_map: Any) -> bool:
    return isinstance(device_map, dict) and "disk" in device_map.values()


def _is_safetensors_file(path: Any) -> bool:
    """Tolerant of pathlib.Path / os.PathLike inputs."""
    return str(path).endswith(".safetensors")


def _as_set(value: Any) -> set:
    return set(value or ())


# Not thread-safe; load runs on the main thread, so a race here at
# worst produces a duplicate warning, never corruption.
_PAGE_CACHE_FAIL_LOGGED = False


def _drop_file_page_cache(path: str) -> None:
    """Best-effort ``posix_fadvise(POSIX_FADV_DONTNEED)``. No-op where
    unsupported. The first failure on a given process logs a one-shot
    WARNING so users on filesystems where fadvise doesn't release pages
    (e.g. some NFS configs) can see why streaming peak memory isn't
    dropping as expected; subsequent failures are silent to avoid spam."""
    global _PAGE_CACHE_FAIL_LOGGED
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        if not _PAGE_CACHE_FAIL_LOGGED:
            logger.warning(
                "Unsloth: integrated_gpu_loader: os.posix_fadvise unavailable; "
                "page cache will not be dropped between safetensors shards. "
                "Peak memory may not reduce as expected.",
            )
            _PAGE_CACHE_FAIL_LOGGED = True
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        if not _PAGE_CACHE_FAIL_LOGGED:
            logger.warning(
                "Unsloth: integrated_gpu_loader: open(%s, O_RDONLY) failed (%r); "
                "skipping page-cache drop.",
                path,
                exc,
            )
            _PAGE_CACHE_FAIL_LOGGED = True
        return
    try:
        try:
            fadvise(fd, 0, 0, dontneed)
        except OSError as exc:
            if not _PAGE_CACHE_FAIL_LOGGED:
                logger.warning(
                    "Unsloth: integrated_gpu_loader: posix_fadvise(DONTNEED) "
                    "failed on %s (%r); peak memory may not reduce. Common "
                    "cause: an NFS-mounted checkpoint on a kernel whose NFS "
                    "client ignores fadvise. Subsequent failures will be "
                    "silent.",
                    path,
                    exc,
                )
                _PAGE_CACHE_FAIL_LOGGED = True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _merge_loading_infos(model, per_shard_infos, expected_keys=None):
    """Combine per-shard ``LoadStateDictInfo`` into one matching a single-call
    invocation. Relies on the upstream contract that per-shard
    ``LoadStateDictInfo.missing_keys`` is ``expected_keys − keys_loaded
    _by_this_shard``; intersecting across shards therefore yields the
    truly-missing set (mathematically equivalent to ``expected_keys −
    union(loaded_per_shard)``). Revisit if upstream's per-shard
    contract changes. Other fields are unioned;
    ``conversion_errors`` merges first-write-wins via ``ChainMap``.

    ``expected_keys`` (optional): if provided, the empty/no-shard
    fallback uses this set instead of rebuilding from
    ``model.state_dict()``."""
    from transformers.utils.loading_report import LoadStateDictInfo

    if not per_shard_infos:
        keys = (
            set(expected_keys)
            if expected_keys is not None
            else set(model.state_dict().keys())
        )
        return LoadStateDictInfo(
            missing_keys=keys,
            unexpected_keys=set(),
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
        )

    missing = _as_set(per_shard_infos[0].missing_keys)
    unexpected: set = set()
    mismatched: set = set()
    errors: list[str] = []
    convs_chain: list[dict[str, str]] = []
    for li in per_shard_infos:
        missing &= _as_set(li.missing_keys)
        unexpected |= _as_set(li.unexpected_keys)
        mismatched |= _as_set(li.mismatched_keys)
        errors.extend(li.error_msgs or [])
        if li.conversion_errors:
            convs_chain.append(li.conversion_errors)
    convs = dict(ChainMap(*convs_chain)) if convs_chain else {}

    return LoadStateDictInfo(
        missing_keys=missing,
        unexpected_keys=unexpected,
        mismatched_keys=mismatched,
        error_msgs=errors,
        conversion_errors=convs,
    )


def _has_cross_shard_fusion(load_config) -> bool:
    """True if ``weight_mapping`` contains a multi-source ``WeightConverter``
    whose operations need every matching source tensor live in one
    ``convert()`` call.

    Hazardous ops (verified by inspection of
    ``transformers.core_model_loading``): ``MergeModulelist`` (fuses N
    tensors along dim 0) and ``Concatenate`` (concatenates source
    patterns). Both iterate ``source_patterns`` against ``input_dict`` and
    silently produce a wrong-shape merged tensor when sources are split
    across shards.

    Single-source converters (PermuteForRope, Transpose, Chunk, etc.) are
    safe under per-shard splits because each call independently reproduces
    the transform.

    Fail-safe on import failure: if ``WeightConverter`` or the hazardous
    op classes can't be imported, return True so the patch declines
    streaming rather than silently streaming a fusion load.
    ``_check_symbols`` also probes these imports at install time so a
    rename declines patches entirely.
    """
    try:
        from transformers.core_model_loading import (
            Concatenate,
            MergeModulelist,
            WeightConverter,
        )
    except Exception:
        return True
    hazardous = (MergeModulelist, Concatenate)
    wm = getattr(load_config, "weight_mapping", None) or []
    for entry in wm:
        if not isinstance(entry, WeightConverter):
            continue
        try:
            sources = getattr(entry, "source_patterns", None)
            ops = getattr(entry, "operations", None)
            if sources is None or ops is None:
                # Unexpected shape on a class we KNOW is a WeightConverter
                # -> something changed in the upstream contract. Fail
                # closed.
                return True
            if len(sources) <= 1:
                continue
            if any(isinstance(op, hazardous) for op in ops):
                return True
        except Exception:
            # Any failure to inspect a converter -> assume it could be a
            # cross-shard fusion and decline streaming.
            return True
    return False


def _is_pre_quantized_load(load_config) -> bool:
    """True iff the checkpoint is already fully quantized.

    Streaming is safe only when each shard's keys map directly onto model
    params. On-the-fly quantization (fp16/bf16 checkpoint + bnb config at
    load time) needs every source tensor live in one
    ``convert_and_load_state_dict_in_model`` call, so the original loader
    must run.

    Practical consequence: ``unsloth/gpt-oss-120b-unsloth-bnb-4bit`` and
    similar pre-quantized repos hit the streaming path. ``unsloth/Llama-3.1-8B``
    + ``BitsAndBytesConfig(load_in_4bit=True)`` does NOT — only Patch A
    (device_map coercion) helps that case.
    """
    try:
        hf_q = getattr(load_config, "hf_quantizer", None)
        if hf_q is None:
            return False
        return bool(getattr(hf_q, "pre_quantized", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Patch A: routed _get_device_map
# ---------------------------------------------------------------------------


def _build_routed_get_device_map(original):
    @functools.wraps(original)
    def _routed_get_device_map(model, device_map, max_memory, hf_quantizer):
        # Use keyword calls everywhere so the runtime contract matches
        # what `_signature_accepts_kwargs(_GET_DEVICE_MAP_REQUIRED_KWARGS)`
        # validates at install time. If upstream ever reorders parameters,
        # a positional call could pass the wrong values without the probe
        # noticing.
        def call_original(_max_memory):
            return original(
                model=model,
                device_map=device_map,
                max_memory=_max_memory,
                hf_quantizer=hf_quantizer,
            )

        if not (
            _is_integrated_gpu()
            and hf_quantizer is not None
            and isinstance(device_map, str)
            and device_map in _STRING_DEVICE_MAPS
        ):
            return call_original(max_memory)

        # Single-GPU branch coerces to {"": cur}; multi-GPU branch keeps
        # balanced/sequential intent and only zeros the cpu bucket. Never
        # collapse a real multi-GPU placement to a single device.
        import torch

        try:
            device_count = torch.cuda.device_count()
        except Exception:
            device_count = 1

        if device_count <= 1:
            # ``current_device()`` (not hard-coded 0) so each DDP rank
            # picks up its own ``CUDA_VISIBLE_DEVICES`` index.
            try:
                idx = torch.cuda.current_device()
            except Exception:
                idx = 0
            coerced = {"": idx}
            # Re-validate so we surface real quantizer/environment problems
            # (mismatched dtype, missing kernels, etc.). The original
            # CPU-scatter ValueError cannot fire against this dict, so
            # legitimate errors are the only thing this catches.
            try:
                hf_quantizer.validate_environment(device_map=coerced)
            except Exception as exc:
                logger.warning(
                    "Unsloth: integrated_gpu_loader: hf_quantizer.validate_environment "
                    "rejected coerced device_map=%r (%r). Falling back to original "
                    "_get_device_map.",
                    coerced,
                    exc,
                )
                return call_original(max_memory)
            return coerced

        # Multi-GPU integrated. Force cpu=0 so the original infer path
        # places everything on GPUs; quantizer validators that reject
        # CPU-tagged maps then pass. Through normal unsloth workflow no
        # caller passes a non-zero ``cpu`` here, and on integrated-memory
        # hosts host RAM is the same pool as GPU RAM, so zeroing it is
        # the correct policy. (A user-supplied non-zero ``cpu`` is
        # silently overwritten -- intentional.)
        try:
            adjusted = {} if max_memory is None else dict(max_memory)
            for i in range(device_count):
                if i not in adjusted and str(i) not in adjusted:
                    adjusted[i] = torch.cuda.get_device_properties(i).total_memory
            adjusted["cpu"] = 0
        except Exception:
            adjusted = max_memory
        return call_original(adjusted)

    _routed_get_device_map._is_unsloth_routed = True
    return _routed_get_device_map


# ---------------------------------------------------------------------------
# Patch B: routed _load_pretrained_model + streaming variant
# ---------------------------------------------------------------------------


class _StreamingMutationState:
    """Records side-effects the streaming loader has applied to ``model`` /
    ``load_config`` so the caller can decide whether falling back to the
    original loader is safe. Once a mutation MIGHT have started, falling
    back risks double-applying disk_offload registrations or skipping
    meta->real init for already-converted params.

    The flag is set BEFORE the mutating call (not after), so a partial
    mutation from a raise mid-call is correctly reflected.
    """

    __slots__ = (
        "disk_offload_called",
        "model_mutation_started",
        "shards_completed",
        "in_memory_called",
        "zero3_called",
    )

    def __init__(self):
        self.disk_offload_called = False
        self.model_mutation_started = False
        self.shards_completed = 0
        self.in_memory_called = False
        self.zero3_called = False

    def mark_model_mutation_started(self) -> None:
        self.model_mutation_started = True

    def mark_shard_completed(self) -> None:
        self.shards_completed += 1

    def is_clean(self) -> bool:
        return (
            not self.disk_offload_called
            and not self.model_mutation_started
            and not self.in_memory_called
            and not self.zero3_called
        )


def _convert_one_shard(
    *,
    model,
    shard_state_dict,
    load_config,
    disk_offload_index,
    mutation_state: _StreamingMutationState,
):
    from transformers.core_model_loading import convert_and_load_state_dict_in_model

    mutation_state.mark_model_mutation_started()
    loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
        model=model,
        state_dict=shard_state_dict,
        load_config=load_config,
        tp_plan=model.tp_plan,
        disk_offload_index=disk_offload_index,
    )
    mutation_state.mark_shard_completed()
    return loading_info, disk_offload_index


def _streaming_load_pretrained_model(
    model,
    state_dict,
    checkpoint_files,
    load_config,
    expected_keys=None,
    mutation_state: _StreamingMutationState | None = None,
):
    """Integrated-GPU variant of ``PreTrainedModel._load_pretrained_model``.

    Skips ``caching_allocator_warmup`` and processes shards one at a time
    (safetensors via mmap'd ``safe_open`` + ``posix_fadvise`` between
    shards; ``.bin`` via ``load_state_dict`` + ``gc.collect`` between
    shards). Per-shard ``LoadStateDictInfo`` results are merged via
    ``_merge_loading_infos`` so the returned object matches a single-call
    invocation.

    ``mutation_state`` (optional) records side-effects so the caller can
    refuse to fall back to the original loader after partial mutation.
    """
    if mutation_state is None:
        mutation_state = _StreamingMutationState()
    from safetensors import safe_open
    from transformers.modeling_utils import (
        _is_on_hf_mount,
        _safe_load_bytes,
        accelerate_disk_offload,
        is_deepspeed_zero3_enabled,
    )
    from transformers.utils.loading_report import LoadStateDictInfo

    disable_mmap = bool(getattr(load_config, "disable_mmap", False))
    weights_only = bool(getattr(load_config, "weights_only", True))
    is_quantized = load_config.is_quantized

    # Materialise expected_keys exactly like the original loader.
    expected_keys = (
        list(model.state_dict().keys()) if expected_keys is None else expected_keys
    )

    disk_offload_index = None
    if _has_disk_offload(getattr(load_config, "device_map", None)):
        # Mark BEFORE the call: if accelerate_disk_offload partially
        # registers state and then raises, is_clean() must reflect that
        # so the routed entry refuses the unsafe original-loader fallback.
        # (Note: under the current routing, _streaming_decline_reason
        # already declines disk-offload loads, so this branch is mostly
        # defensive -- keeping it correct in case the gating ever
        # changes.)
        mutation_state.disk_offload_called = True
        disk_offload_index = accelerate_disk_offload(
            model=model,
            disk_offload_folder=load_config.disk_offload_folder,
            checkpoint_files=checkpoint_files,
            device_map=load_config.device_map,
            sharded_metadata=load_config.sharded_metadata,
            dtype=load_config.dtype,
            weight_mapping=load_config.weight_mapping,
        )

    # ---------- deepspeed-zero3 branch ----------
    if is_deepspeed_zero3_enabled() and not is_quantized:
        from transformers.modeling_utils import (
            _load_state_dict_into_zero3_model,
            load_state_dict,
        )

        if state_dict is None:
            merged: dict = {}
            for ckpt_file in checkpoint_files:
                merged.update(
                    load_state_dict(
                        ckpt_file,
                        map_location="cpu",
                        weights_only=weights_only,
                        disable_mmap=disable_mmap,
                    )
                )
            state_dict = merged
        mutation_state.zero3_called = True
        error_msgs, missing_keys = _load_state_dict_into_zero3_model(
            model, state_dict, load_config
        )
        return (
            LoadStateDictInfo(
                missing_keys=missing_keys,
                unexpected_keys=set(),
                mismatched_keys=set(),
                conversion_errors={},
                error_msgs=error_msgs,
            ),
            disk_offload_index,
        )

    # ---------- in-memory state_dict branch ----------
    if state_dict is not None:
        mutation_state.in_memory_called = True
        loading_info, disk_offload_index = _convert_one_shard(
            model=model,
            shard_state_dict=state_dict,
            load_config=load_config,
            disk_offload_index=disk_offload_index,
            mutation_state=mutation_state,
        )
        return loading_info, disk_offload_index

    # ---------- shard-streaming branches ----------
    if not checkpoint_files:
        # Mirror the original ValueError. Handles both None and [].
        raise ValueError("Neither a state dict nor checkpoint files were found.")

    per_shard_infos: list = []

    if not _is_safetensors_file(checkpoint_files[0]):
        # .bin path: read whole file into memory per shard, gc between.
        from transformers.modeling_utils import load_state_dict

        for ckpt_file in checkpoint_files:
            shard_state_dict = load_state_dict(
                ckpt_file,
                map_location="cpu",
                weights_only=weights_only,
                disable_mmap=disable_mmap,
            )
            li, disk_offload_index = _convert_one_shard(
                model=model,
                shard_state_dict=shard_state_dict,
                load_config=load_config,
                disk_offload_index=disk_offload_index,
                mutation_state=mutation_state,
            )
            per_shard_infos.append(li)
            del shard_state_dict
            gc.collect()
        return _merge_loading_infos(
            model, per_shard_infos, expected_keys
        ), disk_offload_index

    # safetensors path. Branch on disable_mmap / hf-mount: when either is
    # true, the original loader reads the file into memory via
    # _safe_load_bytes (mmap interaction with hf-mount FUSE deadlocks
    # under parallel page-faults). We mirror that branch here so users
    # with disable_mmap=True or a checkpoint on hf-mount get the same
    # semantics they would on the original loader -- still streamed one
    # shard at a time, just without mmap.
    for file in checkpoint_files:
        if disable_mmap or _is_on_hf_mount(file):
            with open(file, "rb") as _fh:
                shard_state_dict = _safe_load_bytes(_fh.read())
            li, disk_offload_index = _convert_one_shard(
                model=model,
                shard_state_dict=shard_state_dict,
                load_config=load_config,
                disk_offload_index=disk_offload_index,
                mutation_state=mutation_state,
            )
            per_shard_infos.append(li)
            shard_state_dict.clear()
            del shard_state_dict
            gc.collect()
            # _fh.read() populated the kernel page cache; drop it so peak
            # resident memory falls between shards on hf-mount and on
            # disable_mmap=True loads (mirrors the mmap branch below).
            _drop_file_page_cache(file)
            continue
        # try/finally so the page-cache drop and slice cleanup run even
        # if _convert_one_shard raises mid-conversion. mutation_state
        # already records the partial mutation; this just keeps unwind
        # peak memory bounded.
        try:
            with safe_open(file, framework="pt", device="cpu") as file_pointer:
                shard_state_dict: dict[str, Any] = {}
                try:
                    for k in file_pointer.keys():
                        shard_state_dict[k] = file_pointer.get_slice(k)
                    li, disk_offload_index = _convert_one_shard(
                        model=model,
                        shard_state_dict=shard_state_dict,
                        load_config=load_config,
                        disk_offload_index=disk_offload_index,
                        mutation_state=mutation_state,
                    )
                    per_shard_infos.append(li)
                finally:
                    # Drop refs to the safetensors slices BEFORE the
                    # file closes so the mmap region has no live
                    # tensors when posix_fadvise runs.
                    shard_state_dict.clear()
                    del shard_state_dict
        finally:
            _drop_file_page_cache(file)
            gc.collect()

    return _merge_loading_infos(
        model, per_shard_infos, expected_keys
    ), disk_offload_index


def _streaming_decline_reason(load_config) -> str | None:
    """Why we should NOT stream this load. Returns ``None`` when streaming
    is safe; otherwise a short string suitable for debug logging.
    """
    if not _is_integrated_gpu():
        return "not an integrated GPU"
    if _has_disk_offload(getattr(load_config, "device_map", None)):
        return "disk offload is enabled"
    if not _is_pre_quantized_load(load_config):
        return "load is not pre-quantized"
    if _has_cross_shard_fusion(load_config):
        return "weight mapping contains cross-shard fusion"
    return None


def _build_routed_load_pretrained_model(original):
    @functools.wraps(original)
    def _routed_load_pretrained_model(
        model, state_dict, checkpoint_files, load_config, expected_keys=None
    ):
        def run_original():
            return original(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )

        reason = _streaming_decline_reason(load_config)
        if reason is not None:
            logger.debug(
                "Unsloth: integrated_gpu_loader: using original loader: %s",
                reason,
            )
            return run_original()

        ms = _StreamingMutationState()
        try:
            return _streaming_load_pretrained_model(
                model,
                state_dict,
                checkpoint_files,
                load_config,
                expected_keys,
                mutation_state=ms,
            )
        except Exception as exc:
            if ms.is_clean():
                # Streaming raised before any mutation (e.g. an import
                # inside the streaming function failed). Falling back to
                # the original loader is safe -- the model is still in
                # its meta state.
                logger.warning(
                    "Unsloth: integrated_gpu_loader streaming raised %r "
                    "before model mutation; falling back to original loader.",
                    exc,
                )
                return run_original()
            # Partial mutation: the original loader cannot be trusted to
            # recover (it would re-call accelerate_disk_offload, double-
            # register hooks, or skip meta->real init for params whose
            # _is_hf_initialized was already set by streaming). Re-raise
            # so the user sees a clean error instead of a downstream
            # crash on garbage state.
            logger.warning(
                "Unsloth: integrated_gpu_loader streaming raised %r AFTER "
                "partial model mutation (disk_offload_called=%s, "
                "model_mutation_started=%s, shards_completed=%d, "
                "in_memory_called=%s, zero3_called=%s). Re-raising to avoid "
                "unsafe fallback. Set UNSLOTH_INTEGRATED_GPU_LOADER=0 to "
                "force the original loader from the start.",
                exc,
                ms.disk_offload_called,
                ms.model_mutation_started,
                ms.shards_completed,
                ms.in_memory_called,
                ms.zero3_called,
            )
            raise

    _routed_load_pretrained_model._is_unsloth_routed = True
    return staticmethod(_routed_load_pretrained_model)


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def apply_integrated_gpu_loader_patches() -> bool:
    """Idempotently install the routed wrappers when the gate fires.

    Concurrent callers are serialised by ``_INSTALL_LOCK`` so two
    threads cannot both pass the marker check and double-wrap.
    """
    with _INSTALL_LOCK:
        if not _should_patch():
            return False

        try:
            import transformers
            import transformers.integrations.accelerate as accel_int
            import transformers.modeling_utils as mu
        except Exception as exc:
            logger.debug("integrated_gpu_loader: transformers import failed: %s", exc)
            return False

        # Warn loudly if the user forced patches on via the override
        # despite being on a real discrete GPU. The streaming path is
        # correct on dGPU but slower than upstream's
        # caching_allocator_warmup; Patch A's multi-GPU branch forces
        # ``max_memory["cpu"]=0`` which on a real dGPU silently breaks
        # legitimate auto/balanced placements that include host-RAM
        # offload.
        if _integrated_gpu_override() is True and not _detect_integrated_gpu():
            try:
                import torch

                n = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except Exception:
                n = 0
            logger.warning(
                "Unsloth: integrated_gpu_loader patches FORCED on by "
                "UNSLOTH_INTEGRATED_GPU_LOADER=1 even though this device is NOT "
                "integrated (visible_cuda_devices=%d). The patches are designed "
                "for unified-memory hosts (NVIDIA GB10 / Spark). On a real "
                "discrete GPU the streaming loader works but is slower than the "
                "original; Patch A's max_memory['cpu']=0 will also override any "
                "legitimate CPU-offload device_map='auto' you may want. Unset "
                "the env var to gate off automatically.",
                n,
            )

        # Patch A: re-arm only when the marker is missing. The
        # ``_PATCH_FLAG_ATTR`` on PreTrainedModel guards Patch B only;
        # the device-map wrapper has its own marker so a third-party
        # reload of ``transformers.integrations.accelerate`` re-arms
        # without touching Patch B's flag.
        if not getattr(accel_int._get_device_map, "_is_unsloth_routed", False):
            accel_int._get_device_map = _build_routed_get_device_map(
                accel_int._get_device_map
            )
        # ``transformers.modeling_utils`` imports ``_get_device_map``
        # locally at module load. ALWAYS rebind ``mu._get_device_map``
        # to the current ``accel_int._get_device_map`` -- not just when
        # mu's binding lacks ``_is_unsloth_routed``. Otherwise after
        # ``importlib.reload(transformers.integrations.accelerate)`` we
        # could end up with mu pointing at a stale routed closure that
        # closes over the OLD original, while accel_int has the new
        # wrapper closing over the new original.
        if hasattr(mu, "_get_device_map"):
            mu._get_device_map = accel_int._get_device_map

        # Patch B is gated by both the class-level _PATCH_FLAG_ATTR
        # (below) and the per-binding _is_unsloth_routed marker (the
        # check that follows). The redundancy survives a partial reload
        # that clears one but not the other.
        if getattr(mu.PreTrainedModel, _PATCH_FLAG_ATTR, False):
            return True

        if not getattr(
            mu.PreTrainedModel._load_pretrained_model,
            "_is_unsloth_routed",
            False,
        ):
            mu.PreTrainedModel._load_pretrained_model = (
                _build_routed_load_pretrained_model(
                    mu.PreTrainedModel._load_pretrained_model
                )
            )

        setattr(mu.PreTrainedModel, _PATCH_FLAG_ATTR, True)

        logger.info(
            "Unsloth: installed integrated-GPU loader patches "
            "(transformers=%s, detected_integrated=%s, override=%r)",
            transformers.__version__,
            _detect_integrated_gpu(),
            _integrated_gpu_override(),
        )
        return True
