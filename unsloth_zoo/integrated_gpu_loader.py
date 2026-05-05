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

from __future__ import annotations

import functools
import gc
import logging
import os
from collections import ChainMap
from typing import Any

logger = logging.getLogger(__name__)


_PATCH_FLAG_ATTR = "_unsloth_integrated_loader_patched"


# 4.57.x and earlier streams shards correctly and doesn't have the
# device_map regression. The relevant load-path rewrite lands in 5.x;
# the early 5.x line saw frequent churn in the symbols we depend on, so
# in practice the patch only activates on >= 5.5 (5.0.x..5.4.x will
# typically fail _check_symbols and decline). Keep the floor at 5.0 so
# the symbol check is the source of truth, not a string compare.
_MIN_VERSION = "5.0.0"


def _is_integrated_gpu(index: int = 0) -> bool:
    override = os.environ.get("UNSLOTH_INTEGRATED_GPU_LOADER", "").strip()
    if override == "1":
        return True
    if override == "0":
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return getattr(torch.cuda.get_device_properties(index), "is_integrated", 0) == 1
    except Exception:
        return False


def _transformers_at_least_min() -> bool:
    """True if transformers >= _MIN_VERSION (no upper bound)."""
    try:
        from importlib.metadata import version

        from packaging.version import Version

        return Version(version("transformers")) >= Version(_MIN_VERSION)
    except Exception:
        return False


def _check_symbols() -> tuple[bool, str]:
    """Verify every transformers symbol/signature we depend on still has
    the shape we expect. Returns (ok, reason).
    """
    try:
        import inspect

        from safetensors import safe_open  # noqa: F401
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.integrations.accelerate import _get_device_map
        from transformers.modeling_utils import (
            PreTrainedModel,
            accelerate_disk_offload,  # noqa: F401
            is_deepspeed_zero3_enabled,  # noqa: F401
            load_state_dict,  # noqa: F401
        )
        from transformers.utils.loading_report import LoadStateDictInfo
        from transformers.utils.quantization_config import (
            QuantizationMethod,  # noqa: F401
        )
    except Exception as e:
        return False, f"import failed: {e!r}"

    # 1. _load_pretrained_model must be a staticmethod with our 5-arg shape.
    try:
        sig = inspect.signature(PreTrainedModel._load_pretrained_model)
    except (TypeError, ValueError) as e:
        return False, f"_load_pretrained_model not introspectable: {e!r}"
    expected_lpm = (
        "model",
        "state_dict",
        "checkpoint_files",
        "load_config",
        "expected_keys",
    )
    if tuple(sig.parameters.keys()) != expected_lpm:
        return False, (
            f"_load_pretrained_model signature changed: "
            f"got {tuple(sig.parameters.keys())}, expected {expected_lpm}"
        )

    # 2. convert_and_load_state_dict_in_model must accept the kwargs we pass.
    try:
        sig2 = inspect.signature(convert_and_load_state_dict_in_model)
    except (TypeError, ValueError) as e:
        return False, f"convert_and_load_state_dict_in_model not introspectable: {e!r}"
    needed_calsdim = {
        "model",
        "state_dict",
        "load_config",
        "tp_plan",
        "disk_offload_index",
    }
    actual_calsdim = set(sig2.parameters.keys())
    if not needed_calsdim.issubset(actual_calsdim):
        return False, (
            f"convert_and_load_state_dict_in_model missing kwargs "
            f"{needed_calsdim - actual_calsdim} (got {actual_calsdim})"
        )

    # 3. LoadStateDictInfo must be a dataclass with exactly the fields we
    #    read+write during the per-shard merge.
    needed_fields = {
        "missing_keys",
        "unexpected_keys",
        "mismatched_keys",
        "error_msgs",
        "conversion_errors",
    }
    fields_attr = getattr(LoadStateDictInfo, "__dataclass_fields__", None)
    if fields_attr is None:
        return False, "LoadStateDictInfo no longer a dataclass"
    fields = set(fields_attr.keys())
    if not needed_fields.issubset(fields):
        return False, (
            f"LoadStateDictInfo missing fields {needed_fields - fields} (got {fields})"
        )

    # 4. _get_device_map signature.
    try:
        sig3 = inspect.signature(_get_device_map)
    except (TypeError, ValueError) as e:
        return False, f"_get_device_map not introspectable: {e!r}"
    needed_gdm = {"model", "device_map", "max_memory", "hf_quantizer"}
    actual_gdm = set(sig3.parameters.keys())
    if not needed_gdm.issubset(actual_gdm):
        return False, (
            f"_get_device_map missing kwargs {needed_gdm - actual_gdm} "
            f"(got {actual_gdm})"
        )

    # 5. accelerate_disk_offload accepts the 7 positional args we pass.
    #    A *args-style signature passes the check (VAR_POSITIONAL absorbs
    #    arbitrary positional arity); we only fail if there are NO *args
    #    AND fewer than 7 named positional/keyword slots.
    try:
        sig4 = inspect.signature(accelerate_disk_offload)
    except (TypeError, ValueError) as e:
        return False, f"accelerate_disk_offload not introspectable: {e!r}"
    needed_ado = 7  # model, folder, files, device_map, sharded_metadata, dtype, weight_mapping
    has_var_positional = any(
        p.kind == inspect.Parameter.VAR_POSITIONAL
        for p in sig4.parameters.values()
    )
    named_positional = [
        p
        for p in sig4.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if not has_var_positional and len(named_positional) < needed_ado:
        return False, (
            f"accelerate_disk_offload arity changed: got "
            f"{tuple(sig4.parameters.keys())}, expected at least {needed_ado} positional"
        )

    # 6. LoadStateDictConfig fields we consume by name.
    try:
        from transformers.modeling_utils import LoadStateDictConfig
    except Exception as e:
        return False, f"LoadStateDictConfig import failed: {e!r}"
    needed_lsdc_fields = {
        "device_map",
        "disable_mmap",
        "weights_only",
        "hf_quantizer",
        "weight_mapping",
        "sharded_metadata",
        "disk_offload_folder",
        "dtype",
    }
    fields_attr = getattr(LoadStateDictConfig, "__dataclass_fields__", None)
    if fields_attr is None:
        return False, "LoadStateDictConfig no longer a dataclass"
    lsdc_fields = set(fields_attr.keys())
    if not needed_lsdc_fields.issubset(lsdc_fields):
        return False, (
            f"LoadStateDictConfig missing fields "
            f"{needed_lsdc_fields - lsdc_fields} (got {lsdc_fields})"
        )
    # is_quantized is a @property, not a dataclass field.
    if not hasattr(LoadStateDictConfig, "is_quantized"):
        return False, "LoadStateDictConfig has no is_quantized property"

    # 7. tp_plan must be a property on PreTrainedModel (returns _tp_plan or
    #    _ep_plan depending on config.distributed_config.enable_expert_parallel).
    if not isinstance(PreTrainedModel.__dict__.get("tp_plan"), property):
        return False, "PreTrainedModel.tp_plan is no longer a property"

    # 8. HfQuantizer base class import. ``pre_quantized`` is an instance
    #    attribute set in ``__init__``, so we cannot probe for it on the
    #    class; ``_is_pre_quantized_load`` does a ``getattr(..., False)``
    #    at runtime which covers a future removal gracefully.
    try:
        from transformers.quantizers.base import HfQuantizer  # noqa: F401
    except Exception as e:
        return False, f"HfQuantizer import failed: {e!r}"

    # 9. WeightConverter + the hazardous fusion ops we gate against. If a
    #    future transformers renames any of these, ``_has_cross_shard_fusion``
    #    fails open (returns True conservatively). Decline patches entirely
    #    so the user runs the original loader.
    try:
        from transformers.core_model_loading import (  # noqa: F401
            Concatenate,
            MergeModulelist,
            WeightConverter,
        )
    except Exception as e:
        return False, f"WeightConverter / fusion ops import failed: {e!r}"

    return True, ""


def _required_symbols_present() -> bool:
    ok, _reason = _check_symbols()
    return ok


def _should_patch() -> bool:
    """Binary decision: install the patches, or leave transformers alone."""
    if not _is_integrated_gpu():
        return False
    if not _transformers_at_least_min():
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


def _merge_loading_infos(model, per_shard_infos):
    from transformers.utils.loading_report import LoadStateDictInfo

    if not per_shard_infos:
        all_keys = set(model.state_dict().keys())
        return LoadStateDictInfo(
            missing_keys=all_keys,
            unexpected_keys=set(),
            mismatched_keys=set(),
            error_msgs=[],
            conversion_errors={},
        )

    missing = set(per_shard_infos[0].missing_keys)
    for li in per_shard_infos[1:]:
        missing &= li.missing_keys

    unexpected = set()
    mismatched = set()
    errors: list[str] = []
    convs_chain: list[dict[str, str]] = []
    for li in per_shard_infos:
        unexpected |= li.unexpected_keys
        mismatched |= li.mismatched_keys
        errors.extend(li.error_msgs)
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


# ---------------------------------------------------------------------------
# Patch A: routed _get_device_map
# ---------------------------------------------------------------------------

_STRING_DEVICE_MAPS = ("auto", "sequential", "balanced", "balanced_low_0")


def _build_routed_get_device_map(original):
    @functools.wraps(original)
    def _routed_get_device_map(model, device_map, max_memory, hf_quantizer):
        if not (
            _is_integrated_gpu()
            and hf_quantizer is not None
            and isinstance(device_map, str)
            and device_map in _STRING_DEVICE_MAPS
        ):
            return original(model, device_map, max_memory, hf_quantizer)

        # Single-GPU branch coerces to {"": cur}; multi-GPU branch keeps
        # balanced/sequential intent and only zeros the cpu bucket. Never
        # collapse a real multi-GPU placement to a single device.
        try:
            import torch

            device_count = torch.cuda.device_count()
        except Exception:
            device_count = 1

        if device_count <= 1:
            # ``current_device()`` (not hard-coded 0) so each DDP rank
            # picks up its own ``CUDA_VISIBLE_DEVICES`` index.
            try:
                import torch

                idx = torch.cuda.current_device()
            except Exception:
                idx = 0
            coerced = {"": idx}
            # Re-validate so we surface real quantizer/environment problems
            # (mismatched dtype, missing kernels, etc.) instead of pretending
            # the coercion fixed them. The original CPU-scatter ValueError
            # cannot fire against this dict, so legitimate errors are the
            # only thing this catches.
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
                return original(model, device_map, max_memory, hf_quantizer)
            return coerced

        # Multi-GPU integrated. Force cpu=0 so the original infer path
        # places everything on GPUs; quantizer validators that reject
        # CPU-tagged maps then pass.
        try:
            import torch

            adjusted = {} if max_memory is None else dict(max_memory)
            for i in range(device_count):
                if i not in adjusted and str(i) not in adjusted:
                    adjusted[i] = torch.cuda.get_device_properties(i).total_memory
            adjusted["cpu"] = 0
        except Exception:
            adjusted = max_memory
        return original(model, device_map, adjusted, hf_quantizer)

    _routed_get_device_map._is_unsloth_routed = True
    return _routed_get_device_map


# ---------------------------------------------------------------------------
# Patch B: routed _load_pretrained_model + streaming variant
# ---------------------------------------------------------------------------


class _StreamingMutationState:
    """Records side-effects the streaming loader has applied to ``model`` /
    ``load_config`` so the caller can decide whether falling back to the
    original loader is safe. Once a mutation has happened, falling back
    risks double-applying disk_offload registrations or skipping meta->real
    init for already-converted params."""

    __slots__ = (
        "disk_offload_called",
        "shards_processed",
        "in_memory_called",
        "zero3_called",
    )

    def __init__(self):
        self.disk_offload_called = False
        self.shards_processed = 0
        self.in_memory_called = False
        self.zero3_called = False

    def is_clean(self) -> bool:
        return (
            not self.disk_offload_called
            and self.shards_processed == 0
            and not self.in_memory_called
            and not self.zero3_called
        )


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
    from transformers.core_model_loading import convert_and_load_state_dict_in_model
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

    # Materialise expected_keys exactly like the original (5.5.0:4190).
    expected_keys = (
        list(model.state_dict().keys()) if expected_keys is None else expected_keys
    )

    disk_offload_index = None
    if _has_disk_offload(getattr(load_config, "device_map", None)):
        disk_offload_index = accelerate_disk_offload(
            model,
            load_config.disk_offload_folder,
            checkpoint_files,
            load_config.device_map,
            load_config.sharded_metadata,
            load_config.dtype,
            load_config.weight_mapping,
        )
        mutation_state.disk_offload_called = True

    if is_deepspeed_zero3_enabled() and not is_quantized:
        from transformers.modeling_utils import (
            _load_state_dict_into_zero3_model,
            load_state_dict,
        )

        if state_dict is None:
            merged = {}
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
        # Mark BEFORE the mutating call: if it raises mid-way, the model is
        # partially mutated and the caller must NOT fall back to original.
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

    # State dict given in memory: still benefits from skipping warmup but no shard streaming.
    if state_dict is not None:
        # Mark BEFORE the mutating call: if convert_and_load raises after
        # partially writing params, is_clean() must reflect that so the
        # routed entry refuses the unsafe original-loader fallback.
        mutation_state.in_memory_called = True
        loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
            model=model,
            state_dict=state_dict,
            load_config=load_config,
            tp_plan=model.tp_plan,
            disk_offload_index=disk_offload_index,
        )
        return loading_info, disk_offload_index

    # No checkpoint files at all -> mirror the original ValueError.
    if checkpoint_files is None:
        raise ValueError("Neither a state dict nor checkpoint files were found.")

    # .bin path: process one shard at a time, no mmap to release; gc between.
    if not checkpoint_files[0].endswith(".safetensors"):
        from transformers.modeling_utils import load_state_dict

        per_shard_infos: list = []
        for ckpt_file in checkpoint_files:
            shard_state_dict = load_state_dict(
                ckpt_file,
                map_location="cpu",
                weights_only=weights_only,
                disable_mmap=disable_mmap,
            )
            li, disk_offload_index = convert_and_load_state_dict_in_model(
                model=model,
                state_dict=shard_state_dict,
                load_config=load_config,
                tp_plan=model.tp_plan,
                disk_offload_index=disk_offload_index,
            )
            per_shard_infos.append(li)
            mutation_state.shards_processed += 1
            del shard_state_dict
            gc.collect()
        return _merge_loading_infos(model, per_shard_infos), disk_offload_index

    # safetensors streaming path. Branch on disable_mmap / hf-mount: when
    # either is true, the original loader reads the file into memory via
    # _safe_load_bytes (mmap interaction with hf-mount FUSE deadlocks under
    # parallel page-faults). We mirror that branch here so users with
    # `disable_mmap=True` or a checkpoint on hf-mount get the same semantics
    # they would on the original loader -- still streamed one shard at a
    # time, just without mmap.
    per_shard_infos = []
    for file in checkpoint_files:
        if disable_mmap or _is_on_hf_mount(file):
            with open(file, "rb") as _fh:
                shard_state_dict = _safe_load_bytes(_fh.read())
            li, disk_offload_index = convert_and_load_state_dict_in_model(
                model=model,
                state_dict=shard_state_dict,
                load_config=load_config,
                tp_plan=model.tp_plan,
                disk_offload_index=disk_offload_index,
            )
            per_shard_infos.append(li)
            mutation_state.shards_processed += 1
            shard_state_dict.clear()
            del shard_state_dict
            gc.collect()
            continue
        with safe_open(file, framework="pt", device="cpu") as file_pointer:
            shard_state_dict: dict[str, Any] = {}
            for k in file_pointer.keys():
                shard_state_dict[k] = file_pointer.get_slice(k)
            li, disk_offload_index = convert_and_load_state_dict_in_model(
                model=model,
                state_dict=shard_state_dict,
                load_config=load_config,
                tp_plan=model.tp_plan,
                disk_offload_index=disk_offload_index,
            )
            per_shard_infos.append(li)
            mutation_state.shards_processed += 1
            # Drop our refs to the safetensors slices BEFORE the file closes
            # so the mmap region has no live tensors when posix_fadvise runs.
            shard_state_dict.clear()
            del shard_state_dict
        _drop_file_page_cache(file)
        gc.collect()

    loading_info = _merge_loading_infos(model, per_shard_infos)
    return loading_info, disk_offload_index


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

    Conservative-on-import-failure: if WeightConverter or the hazardous
    op classes can't be imported (e.g. a future transformers rename),
    return True so the patch declines streaming rather than silently
    streaming a fusion load. ``_check_symbols`` also probes these
    imports at install time so a rename declines patches entirely.
    """
    try:
        from transformers.core_model_loading import (
            Concatenate,
            MergeModulelist,
            WeightConverter,
        )
    except Exception:
        return True  # fail-safe: decline streaming rather than risk wrong tensors
    hazardous = (MergeModulelist, Concatenate)
    wm = getattr(load_config, "weight_mapping", None) or []
    for entry in wm:
        if not isinstance(entry, WeightConverter):
            continue
        try:
            sources = getattr(entry, "source_patterns", []) or []
            if len(sources) <= 1:
                # A single source pattern can't be split across shards
                # in a way that affects fusion semantics.
                continue
            ops = getattr(entry, "operations", []) or []
            if any(isinstance(op, hazardous) for op in ops):
                return True
        except Exception:
            continue
    return False


def _is_pre_quantized_load(load_config) -> bool:
    """True iff the checkpoint is already fully quantized.

    Streaming is safe only when each shard's keys map directly onto model
    params. On-the-fly quantization (fp16/bf16 checkpoint + bnb config at
    load time) and any path with cross-shard fusion converters needs every
    source tensor live in one ``convert_and_load_state_dict_in_model``
    call, so the original loader must run.

    Practical consequence: ``unsloth/gpt-oss-120b-unsloth-bnb-4bit`` and
    similar pre-quantized repos hit the streaming path. ``unsloth/Llama-3.1-8B``
    + ``BitsAndBytesConfig(load_in_4bit=True)`` does NOT -- only Patch A
    (device_map coercion) helps that case. See
    ``docs/integrated_gpu_loader.md`` for the per-architecture matrix.
    """
    try:
        hf_q = getattr(load_config, "hf_quantizer", None)
        if hf_q is None:
            return False
        return bool(getattr(hf_q, "pre_quantized", False))
    except Exception:
        return False


def _build_routed_load_pretrained_model(original):
    @functools.wraps(original)
    def _routed_load_pretrained_model(
        model, state_dict, checkpoint_files, load_config, expected_keys=None
    ):
        # Cheap gates first; do nothing on a discrete GPU.
        if not _is_integrated_gpu():
            return original(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )
        # Disk offload uses the original path; streaming doesn't apply.
        if _has_disk_offload(getattr(load_config, "device_map", None)):
            return original(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )
        if not _is_pre_quantized_load(load_config):
            return original(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )
        # Cross-shard fusion converters (MergeModulelist) need every source
        # tensor live in one convert() call; per-shard streaming would split
        # them and produce wrong-shape merged tensors. Decline streaming so
        # these loads run through the original loader unchanged.
        if _has_cross_shard_fusion(load_config):
            return original(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )
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
        except Exception as e:
            if ms.is_clean():
                # Streaming raised before mutating model state (e.g. an
                # import inside the streaming function failed, or the
                # in-memory state_dict path raised before any per-shard
                # processing started). Falling back to the original loader
                # is safe -- the model is still in its meta state.
                logger.warning(
                    "Unsloth: integrated_gpu_loader streaming raised %r "
                    "before any model mutation; falling back to the "
                    "original _load_pretrained_model.",
                    e,
                )
                return original(
                    model,
                    state_dict,
                    checkpoint_files,
                    load_config,
                    expected_keys,
                )
            # Partial mutation: the original loader cannot be trusted to
            # recover (it would re-call accelerate_disk_offload, double-
            # register hooks, or skip meta->real init for params whose
            # _is_hf_initialized flag was already set by streaming). Re-
            # raise with an annotation so the user sees a clean error
            # instead of a downstream forward-pass crash on garbage state.
            logger.warning(
                "Unsloth: integrated_gpu_loader streaming raised %r AFTER "
                "partial model mutation (disk_offload_called=%s, "
                "shards_processed=%d). Re-raising to avoid double-applying "
                "the original loader on a partially-converted model. Set "
                "UNSLOTH_INTEGRATED_GPU_LOADER=0 to force the original "
                "loader from the start.",
                e,
                ms.disk_offload_called,
                ms.shards_processed,
            )
            raise

    _routed_load_pretrained_model._is_unsloth_routed = True
    return staticmethod(_routed_load_pretrained_model)


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _hardware_actually_integrated() -> bool:
    """Detect-only: report whether the device REPORTS as integrated, ignoring
    the ``UNSLOTH_INTEGRATED_GPU_LOADER`` override. Used to decide whether
    to surface a "you forced this on a discrete GPU" warning at install."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return getattr(torch.cuda.get_device_properties(0), "is_integrated", 0) == 1
    except Exception:
        return False


def apply_integrated_gpu_loader_patches() -> bool:
    """Idempotently install the routed wrappers when the gate fires.

    Safe to call multiple times -- the ``_PATCH_FLAG_ATTR`` flag on
    ``PreTrainedModel`` short-circuits subsequent invocations. If a
    third-party module reloads ``transformers.modeling_utils`` later
    (rare; not done by accelerate, peft, or transformers itself), the
    reloaded ``PreTrainedModel`` is a fresh class object and our
    staticmethod patch is on the old one. To re-arm after a reload, the
    caller can simply invoke this function again -- the flag check looks
    up the new class. The same is true of
    ``transformers.integrations.accelerate._get_device_map``.
    """
    if not _should_patch():
        return False

    try:
        import transformers
        import transformers.integrations.accelerate as accel_int
        import transformers.modeling_utils as mu
    except Exception as exc:
        logger.debug("integrated_gpu_loader: transformers import failed: %s", exc)
        return False

    # Warn loudly if the user forced patches on via the override despite
    # being on a real discrete GPU. The streaming path is correct on dGPU
    # but slower than upstream's caching_allocator_warmup; Patch A's
    # multi-GPU branch sets ``max_memory["cpu"]=0`` which silently breaks
    # legitimate auto/balanced placements that include host RAM offload.
    override = os.environ.get("UNSLOTH_INTEGRATED_GPU_LOADER", "").strip()
    if override == "1" and not _hardware_actually_integrated():
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

    if getattr(mu.PreTrainedModel, _PATCH_FLAG_ATTR, False):
        return True  # already installed

    # Patch A
    if not getattr(accel_int._get_device_map, "_is_unsloth_routed", False):
        accel_int._get_device_map = _build_routed_get_device_map(
            accel_int._get_device_map
        )
    # ``transformers/modeling_utils.py`` imports the symbol locally at module
    # load (line ~4113). Always rebind ``mu._get_device_map`` to the routed
    # version, EVEN if accel_int was already routed -- this matters after
    # ``importlib.reload(transformers.modeling_utils)`` resets ``mu`` to the
    # original. Outside the inner ``if`` so re-arm works.
    if hasattr(mu, "_get_device_map") and not getattr(
        mu._get_device_map, "_is_unsloth_routed", False
    ):
        mu._get_device_map = accel_int._get_device_map

    # Patch B
    if not getattr(
        mu.PreTrainedModel._load_pretrained_model, "_is_unsloth_routed", False
    ):
        mu.PreTrainedModel._load_pretrained_model = _build_routed_load_pretrained_model(
            mu.PreTrainedModel._load_pretrained_model
        )

    setattr(mu.PreTrainedModel, _PATCH_FLAG_ATTR, True)

    logger.info(
        "Unsloth: installed integrated-GPU loader patches (transformers=%s, is_integrated=1)",
        transformers.__version__,
    )
    return True
