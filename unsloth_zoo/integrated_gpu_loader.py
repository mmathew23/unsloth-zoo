"""Integrated-GPU loader patches for unified-memory devices.

Wraps two transformers symbols when running on an integrated-memory GPU
(``torch.cuda.get_device_properties(0).is_integrated == 1``, e.g. NVIDIA
GB10 / Spark) and transformers >= ``_MIN_VERSION``:

* ``transformers.integrations.accelerate._get_device_map`` -> coerce
  string device_maps so ``infer_auto_device_map`` doesn't scatter modules
  to ``"cpu"`` (which on unified memory is the same pool the GPUs see).
* ``transformers.modeling_utils.PreTrainedModel._load_pretrained_model``
  -> stream pre-quantized safetensors shards one at a time instead of
  pre-allocating the full footprint and mmap'ing every shard upfront.

On any other configuration the originals run unchanged.
``apply_integrated_gpu_loader_patches()`` is idempotent.

Override the auto-detection with ``UNSLOTH_INTEGRATED_GPU_LOADER=1``
(force on) or ``=0`` (force off).

Background, regressions, and empirical results: see
``docs/integrated_gpu_loader.md``.
"""
from __future__ import annotations

import functools
import gc
import logging
import os
from collections import ChainMap
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# detection + version gate
# ---------------------------------------------------------------------------

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

    Each individual contract is checked with a specific failure reason so
    that future breakage shows up in logs as something actionable rather
    than a silent decline.
    """
    try:
        import inspect
        from transformers.modeling_utils import (
            PreTrainedModel,
            accelerate_disk_offload,  # noqa: F401
            is_deepspeed_zero3_enabled,  # noqa: F401
            load_state_dict,  # noqa: F401
        )
        from transformers.core_model_loading import convert_and_load_state_dict_in_model
        from transformers.utils.loading_report import LoadStateDictInfo
        from transformers.utils.quantization_config import QuantizationMethod  # noqa: F401
        from transformers.integrations.accelerate import _get_device_map
        from safetensors import safe_open  # noqa: F401
    except Exception as e:
        return False, f"import failed: {e!r}"

    # 1. _load_pretrained_model must be a staticmethod with our 5-arg shape.
    try:
        sig = inspect.signature(PreTrainedModel._load_pretrained_model)
    except (TypeError, ValueError) as e:
        return False, f"_load_pretrained_model not introspectable: {e!r}"
    expected_lpm = ("model", "state_dict", "checkpoint_files", "load_config", "expected_keys")
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
    needed_calsdim = {"model", "state_dict", "load_config", "tp_plan", "disk_offload_index"}
    actual_calsdim = set(sig2.parameters.keys())
    if not needed_calsdim.issubset(actual_calsdim):
        return False, (
            f"convert_and_load_state_dict_in_model missing kwargs "
            f"{needed_calsdim - actual_calsdim} (got {actual_calsdim})"
        )

    # 3. LoadStateDictInfo must be a dataclass with exactly the fields we
    #    read+write during the per-shard merge.
    needed_fields = {
        "missing_keys", "unexpected_keys", "mismatched_keys",
        "error_msgs", "conversion_errors",
    }
    fields_attr = getattr(LoadStateDictInfo, "__dataclass_fields__", None)
    if fields_attr is None:
        return False, "LoadStateDictInfo no longer a dataclass"
    fields = set(fields_attr.keys())
    if not needed_fields.issubset(fields):
        return False, (
            f"LoadStateDictInfo missing fields {needed_fields - fields} "
            f"(got {fields})"
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

    return True, ""


def _required_symbols_present() -> bool:
    """Backward-compat alias used by tests and external callers."""
    ok, _reason = _check_symbols()
    return ok


def _should_patch() -> bool:
    """Binary decision: install the patches, or leave transformers alone.

    Declines on discrete GPUs, on transformers < ``_MIN_VERSION``, and on
    any signature/symbol mismatch from ``_check_symbols``. Mismatches log
    a WARNING so future upstream churn is visible at import time.
    """
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
    """``device_map`` can be ``None``, a string ("auto", "balanced"), or a
    dict. ``"disk" in device_map.values()`` only makes sense in the dict
    case; on a string it raises AttributeError before our outer try/except
    catches it.
    """
    return isinstance(device_map, dict) and "disk" in device_map.values()


def _drop_file_page_cache(path: str) -> None:
    """Best-effort ``posix_fadvise(POSIX_FADV_DONTNEED)`` on ``path``. No-op
    where unsupported. Errors are swallowed -- this is a memory hint."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            fadvise(fd, 0, 0, dontneed)
        except OSError:
            pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _merge_loading_infos(model, per_shard_infos):
    """Combine per-shard ``LoadStateDictInfo`` objects so the result matches
    a single-call invocation. ``missing_keys`` is intersected across shards
    (each shard seeds with the full model state_dict and removes what it
    loaded). The other fields are unioned; ``conversion_errors`` merges
    first-write-wins via ``ChainMap``."""
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
                    "_get_device_map.", coerced, exc,
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

def _streaming_load_pretrained_model(model, state_dict, checkpoint_files, load_config, expected_keys=None):
    """Integrated-GPU variant of ``PreTrainedModel._load_pretrained_model``.

    Skips ``caching_allocator_warmup`` and processes shards one at a time
    (safetensors via mmap'd ``safe_open`` + ``posix_fadvise`` between
    shards; ``.bin`` via ``load_state_dict`` + ``gc.collect`` between
    shards). Per-shard ``LoadStateDictInfo`` results are merged via
    ``_merge_loading_infos`` so the returned object matches a single-call
    invocation.

    The deepspeed-zero3 branch and the in-memory ``state_dict`` branch
    delegate to the original convert/load helpers (no shard streaming
    involved).
    """
    from transformers.core_model_loading import convert_and_load_state_dict_in_model
    from transformers.modeling_utils import (
        accelerate_disk_offload,
        is_deepspeed_zero3_enabled,
    )
    from transformers.utils.loading_report import LoadStateDictInfo
    from safetensors import safe_open

    is_quantized = load_config.is_quantized

    # Materialise expected_keys exactly like the original (5.5.0:4190).
    expected_keys = list(model.state_dict().keys()) if expected_keys is None else expected_keys

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

    if is_deepspeed_zero3_enabled() and not is_quantized:
        from transformers.modeling_utils import _load_state_dict_into_zero3_model, load_state_dict
        if state_dict is None:
            merged = {}
            for ckpt_file in checkpoint_files:
                merged.update(load_state_dict(ckpt_file, map_location="cpu", weights_only=load_config.weights_only))
            state_dict = merged
        error_msgs, missing_keys = _load_state_dict_into_zero3_model(model, state_dict, load_config)
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
        loading_info, disk_offload_index = convert_and_load_state_dict_in_model(
            model=model,
            state_dict=state_dict,
            load_config=load_config,
            tp_plan=model._tp_plan,
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
            shard_state_dict = load_state_dict(ckpt_file)
            li, disk_offload_index = convert_and_load_state_dict_in_model(
                model=model,
                state_dict=shard_state_dict,
                load_config=load_config,
                tp_plan=model._tp_plan,
                disk_offload_index=disk_offload_index,
            )
            per_shard_infos.append(li)
            del shard_state_dict
            gc.collect()
        return _merge_loading_infos(model, per_shard_infos), disk_offload_index

    # safetensors streaming path.
    per_shard_infos = []
    for file in checkpoint_files:
        with safe_open(file, framework="pt", device="cpu") as file_pointer:
            shard_state_dict: dict[str, Any] = {}
            for k in file_pointer.keys():
                shard_state_dict[k] = file_pointer.get_slice(k)
            li, disk_offload_index = convert_and_load_state_dict_in_model(
                model=model,
                state_dict=shard_state_dict,
                load_config=load_config,
                tp_plan=model._tp_plan,
                disk_offload_index=disk_offload_index,
            )
            per_shard_infos.append(li)
            # Drop our refs to the safetensors slices BEFORE the file closes
            # so the mmap region has no live tensors when posix_fadvise runs.
            shard_state_dict.clear()
            del shard_state_dict
        _drop_file_page_cache(file)
        gc.collect()

    loading_info = _merge_loading_infos(model, per_shard_infos)
    return loading_info, disk_offload_index


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
    def _routed_load_pretrained_model(model, state_dict, checkpoint_files, load_config, expected_keys=None):
        # Cheap gates first; do nothing on a discrete GPU.
        if not _is_integrated_gpu():
            return original(model, state_dict, checkpoint_files, load_config, expected_keys)
        # Disk offload uses the original path; streaming doesn't apply.
        if _has_disk_offload(getattr(load_config, "device_map", None)):
            return original(model, state_dict, checkpoint_files, load_config, expected_keys)
        if not _is_pre_quantized_load(load_config):
            return original(model, state_dict, checkpoint_files, load_config, expected_keys)
        try:
            return _streaming_load_pretrained_model(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )
        except Exception as e:
            # The streaming loop may have already mutated the model (some
            # shards loaded, hooks/quantizer state applied, tied-weight
            # bookkeeping started) before raising. Re-running the original
            # loader on top of that partial state usually overwrites the
            # loaded params with the same values, but we cannot guarantee
            # it for every quantizer/offload combination -- the surviving
            # model object may end up in a partially-converted state. Log
            # loudly so this is investigable, then defer to the original.
            logger.warning(
                "Unsloth: integrated_gpu_loader streaming raised %r AFTER "
                "potentially mutating the model; falling back to the "
                "original _load_pretrained_model on the partially-loaded "
                "object. Set UNSLOTH_INTEGRATED_GPU_LOADER=0 to force the "
                "original loader from the start, or report this with the "
                "traceback so the streaming gate can be tightened.", e,
            )
            return original(model, state_dict, checkpoint_files, load_config, expected_keys)

    _routed_load_pretrained_model._is_unsloth_routed = True
    return staticmethod(_routed_load_pretrained_model)


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

def apply_integrated_gpu_loader_patches() -> bool:
    """Idempotently install the routed ``_get_device_map`` and
    ``_load_pretrained_model`` wrappers when the version+hardware gate fires.

    Returns True if patches are now active, False otherwise.
    """
    if not _should_patch():
        return False

    try:
        import transformers
        import transformers.modeling_utils as mu
        import transformers.integrations.accelerate as accel_int
    except Exception as exc:
        logger.debug("integrated_gpu_loader: transformers import failed: %s", exc)
        return False

    if getattr(mu.PreTrainedModel, _PATCH_FLAG_ATTR, False):
        return True  # already installed

    # Patch A
    if not getattr(accel_int._get_device_map, "_is_unsloth_routed", False):
        accel_int._get_device_map = _build_routed_get_device_map(accel_int._get_device_map)
        # transformers/modeling_utils.py imports the symbol locally (line ~4113).
        if hasattr(mu, "_get_device_map"):
            mu._get_device_map = accel_int._get_device_map

    # Patch B
    if not getattr(mu.PreTrainedModel._load_pretrained_model, "_is_unsloth_routed", False):
        mu.PreTrainedModel._load_pretrained_model = _build_routed_load_pretrained_model(
            mu.PreTrainedModel._load_pretrained_model
        )

    setattr(mu.PreTrainedModel, _PATCH_FLAG_ATTR, True)

    logger.info(
        "Unsloth: installed integrated-GPU loader patches (transformers=%s, is_integrated=1)",
        transformers.__version__,
    )
    return True
