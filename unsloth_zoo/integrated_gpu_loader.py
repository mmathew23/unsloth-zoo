"""
Integrated-GPU loader patches for transformers >= 5.5.0.

On unified-memory devices (NVIDIA GB10 / Spark, where
``torch.cuda.get_device_properties(0).is_integrated == 1``) the GPU memory
pool IS the system RAM pool. transformers 5.5.0 restructured the model load
path in two ways that are harmless on discrete GPUs but produce a ~2x peak
memory spike (and an outright load-time error) on integrated GPUs:

1. ``transformers.integrations.accelerate._get_device_map`` (5.5.0:339-375)
   calls ``infer_auto_device_map`` BEFORE ``hf_quantizer.validate_environment``.
   For ``device_map`` strings ("auto"/"sequential"/"balanced") on a unified-
   memory device, ``infer_auto_device_map`` sees host RAM as a separate pool
   and returns a dict mixing GPU indices with ``"cpu"``. The bnb-4bit
   validator at ``transformers/quantizers/quantizer_bnb_4bit.py:74-81`` then
   raises ``ValueError("Some modules are dispatched on the CPU or the disk...")``.
   In 4.57.6 the bnb quantizer's own ``update_device_map(None) -> {"": 0}``
   ran first; the order changed in 5.5.0.

2. ``transformers.modeling_utils.PreTrainedModel._load_pretrained_model``
   (5.5.0:4174-4264) now calls ``caching_allocator_warmup`` to pre-allocate
   the full model size up to ``total_device_memory - 1.2 GiB`` (4827) and then
   opens ALL safetensors shards via ``safe_open(..., device="cpu")`` (4240),
   accumulating slices in one ``merged_state_dict`` (4243) before processing
   (4252). On unified memory, the pre-allocation eats the same RAM the mmap
   needs, AND the all-files-open-at-once pattern keeps every shard's mmap
   pages resident in process RSS until end-of-load (4262). Empirical peak on
   the 122 GiB GB10 with bnb-4bit gpt-oss-120b: 113 GiB (vs 71 GiB on 4.57.6).

This module monkey-patches both functions with routed wrappers gated on
``_should_patch()`` (integrated GPU + transformers >= 5.5.0). On any other
configuration (discrete GPU, or transformers < 5.5.0) the originals run
unchanged. ``apply_integrated_gpu_loader_patches()`` is idempotent.

The streaming ``_load_pretrained_model`` variant:
* skips ``caching_allocator_warmup`` (cudaMalloc speedup is meaningless when
  GPU and CPU share the pool)
* opens, processes and closes one safetensors shard at a time
* calls ``posix_fadvise(POSIX_FADV_DONTNEED)`` after each shard to release
  page cache
* merges per-shard ``LoadStateDictInfo`` results so the returned object is
  identical to a single-call invocation (downstream
  ``_finalize_model_loading`` sees the same data)

Override the auto-detection with ``UNSLOTH_INTEGRATED_GPU_LOADER=1`` (force
on) or ``=0`` (force off) for testing.
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


# Below this version transformers does not have the regressions we patch
# (4.57.x and earlier already streams shards correctly).
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
    """Decide whether to install the integrated-GPU patches.

    Behaviour is binary:
      * returns True  -> patches install; you get the streaming loader.
      * returns False -> patches do NOT install; the original transformers
                         loader runs. On integrated memory this is the
                         slow path (peak ~2x final), but it is correct.

    There is no "warn and patch anyway" branch. If anything looks off
    (hardware not integrated, transformers too old, or any symbol /
    signature we depend on doesn't exist with the expected shape) we
    decline and log WHY. A separate runtime safety net inside
    `_routed_load_pretrained_model` catches exceptions thrown from inside
    the streaming code itself (in case a future transformers release
    passes our install-time checks but breaks the call sequence at run
    time).
    """
    if not _is_integrated_gpu():
        return False
    if not _transformers_at_least_min():
        return False
    ok, reason = _check_symbols()
    if not ok:
        # Surface mismatches loudly so a future upstream change is visible
        # at unsloth_zoo import time rather than just looking like
        # "unsloth got slow again on Spark".
        logger.warning(
            "Unsloth: integrated_gpu_loader will NOT patch this transformers "
            "version because %s. Original (slower on unified-memory) loader "
            "will run unchanged.",
            reason,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _drop_file_page_cache(path: str) -> None:
    """Best-effort: ask the kernel to drop cached pages for ``path``.

    No-op on systems without ``posix_fadvise`` or for non-existent files. We
    swallow OSError because this is a memory hint, not a correctness step.
    """
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
    """Combine per-shard ``LoadStateDictInfo`` objects into one that matches a
    single-call invocation of ``convert_and_load_state_dict_in_model``.

    Each per-shard call seeds ``missing_keys`` from the full model state_dict
    and removes only the keys it loaded. The intersection across shards is
    therefore the truly missing set. ``unexpected_keys`` /
    ``mismatched_keys`` / ``error_msgs`` get unioned; ``conversion_errors``
    is merged first-write-wins.
    """
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

        # Two distinct integrated-GPU branches. We MUST NOT collapse a real
        # multi-GPU placement to a single device just because the user asked
        # for "balanced" or "sequential" — those are intentional distribution
        # strategies. Only collapse when there is exactly one visible GPU
        # (current Spark hardware, plus each DDP rank's per-process view).
        try:
            import torch
            device_count = torch.cuda.device_count()
        except Exception:
            device_count = 1

        if device_count <= 1:
            # Single visible GPU. ``current_device()`` (not hard-coded 0) so
            # each DDP rank picks up its own ``CUDA_VISIBLE_DEVICES`` index.
            try:
                import torch
                idx = torch.cuda.current_device()
            except Exception:
                idx = 0
            coerced = {"": idx}
            try:
                hf_quantizer.validate_environment(device_map=coerced)
            except Exception:
                pass
            return coerced

        # Multi-GPU integrated (multi-Spark or future hardware). Preserve
        # balanced/sequential intent by calling the original infer path, but
        # force ``cpu = 0`` in ``max_memory`` so it places everything on
        # GPUs. Bnb-style validators that reject CPU-tagged maps then pass.
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

    Mirrors transformers/modeling_utils.py:_load_pretrained_model (5.5.0:4174-4264)
    with three changes:
      1. skip caching_allocator_warmup (the up-front pre-allocation is harmful
         on unified memory),
      2. process safetensors shards one-at-a-time (open, build slice dict,
         convert, close, posix_fadvise, gc.collect),
      3. merge per-shard LoadStateDictInfo via ``_merge_loading_infos``.

    Falls back to the original loader for: deepspeed-zero3 paths, .bin
    checkpoints, in-memory state_dicts, and disk-offload device_maps.
    """
    from transformers.core_model_loading import convert_and_load_state_dict_in_model
    from transformers.modeling_utils import (
        accelerate_disk_offload,
        is_deepspeed_zero3_enabled,
    )
    from transformers.utils.quantization_config import QuantizationMethod
    from transformers.utils.loading_report import LoadStateDictInfo
    from safetensors import safe_open

    is_quantized = load_config.is_quantized
    is_hqq_or_quark = is_quantized and load_config.hf_quantizer.quantization_config.quant_method in {
        QuantizationMethod.HQQ,
        QuantizationMethod.QUARK,
    }

    # Materialise expected_keys exactly like the original (5.5.0:4190).
    expected_keys = list(model.state_dict().keys()) if expected_keys is None else expected_keys

    # Disk offload bookkeeping.
    disk_offload_index = None
    if load_config.device_map is not None and "disk" in load_config.device_map.values():
        disk_offload_index = accelerate_disk_offload(
            model,
            load_config.disk_offload_folder,
            checkpoint_files,
            load_config.device_map,
            load_config.sharded_metadata,
            load_config.dtype,
            load_config.weight_mapping,
        )

    # NOTE: caching_allocator_warmup intentionally skipped here.
    _ = is_hqq_or_quark  # kept for parity with the original gate at 4210-4212

    # Deepspeed-zero3 path: defer to original behaviour (no streaming benefit).
    if is_deepspeed_zero3_enabled() and not is_quantized:
        # Replicate the original deepspeed branch (5.5.0:4216-4232) without the warmup.
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
    """True only if the checkpoint is already quantized end-to-end.

    The streaming loader is safe ONLY for pre_quantized=True checkpoints,
    where each checkpoint key maps directly to a model param (no fusion
    converters that would require multiple source tensors landing in the
    same call to convert_and_load_state_dict_in_model). For on-the-fly
    quantization or fp16->fp16 loads with weight fusion (e.g. MoE
    gate/up/down stacking from per-expert checkpoint keys), the original
    loader must run to keep all source tensors live until the converter
    fires.
    """
    try:
        hf_q = getattr(load_config, "hf_quantizer", None)
        if hf_q is None:
            # Non-quantized fp16/bf16 load. May still have safetensors-level
            # weight renaming, but no fusion converters that span shards in
            # any model architecture we know of for the bnb-4bit-on-Spark
            # use-case. Be conservative and decline to stream.
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
        # Disk-offload uses the original path: streaming doesn't apply.
        if (
            getattr(load_config, "device_map", None) is not None
            and "disk" in load_config.device_map.values()
        ):
            return original(model, state_dict, checkpoint_files, load_config, expected_keys)
        # Cross-shard weight converters (the gpt-oss MoE fp16->bnb path, etc.)
        # need every source tensor in one convert_and_load call. Per-shard
        # streaming would split them and produce missing keys. Only stream
        # when the checkpoint is already fully quantized.
        if not _is_pre_quantized_load(load_config):
            return original(model, state_dict, checkpoint_files, load_config, expected_keys)
        # Try the fast streaming path. If anything in our copy of the loader
        # raises (e.g. a future transformers releases reshapes
        # convert_and_load_state_dict_in_model or LoadStateDictInfo in a way
        # our install-time checks didn't catch), fall back to the original
        # loader and log loudly. The unpatched path still WORKS on this
        # hardware -- it just spikes peak memory; correctness is preserved.
        try:
            return _streaming_load_pretrained_model(
                model, state_dict, checkpoint_files, load_config, expected_keys
            )
        except Exception as e:
            logger.warning(
                "Unsloth: integrated_gpu_loader streaming path raised %r; "
                "falling back to original _load_pretrained_model. Peak load "
                "memory may spike.", e,
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
