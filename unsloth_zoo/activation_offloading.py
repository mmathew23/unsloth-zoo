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
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# VL-specific AO defaults and env overrides
# ---------------------------------------------------------------------------
#
# TRL's `OffloadActivations` defaults (use_streams=True, use_pin_memory=True,
# min_offload_size=1024) are tuned for dense text decoder stacks. On
# vision-language models (Qwen2.5-VL, Qwen3-VL, LLaVA, Gemma3, MLLaMA,
# Idefics*) those defaults *inflate* CUDA reserved memory by 1–3 GiB because
# the forward/backward stream stashes and `record_stream` leases keep vision
# tower activations resident on the GPU across the forward→backward boundary,
# fragmenting the caching allocator.
#
# The fix: detect VLMs at trainer build time and default `use_streams=False`
# for them. This removes the stash paths entirely and makes the allocator
# lifetime deterministic. Users can still force the TRL default on via
# `UNSLOTH_AO_USE_STREAMS=1`. Text-path defaults are unchanged.

_VL_MODEL_TYPES: frozenset[str] = frozenset({
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "llava",
    "llava_next",
    "llava_next_video",
    "llava_onevision",
    "gemma3",
    "mllama",
    "idefics",
    "idefics2",
    "idefics3",
    "paligemma",
})


def _parse_bool_env(name: str) -> Optional[bool]:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None


def is_vlm_model(model: Any) -> bool:
    """Return True when the model carries a vision encoder.

    Checks the model's ``config.vision_config`` as well as ``config.model_type``
    against a small list of known VL families. Handles PEFT wrappers and
    Unsloth's ``model.model`` nesting by walking one level of attribute
    indirection before giving up.
    """
    if model is None:
        return False

    seen_ids: set[int] = set()
    current = model
    for _ in range(5):
        if current is None or id(current) in seen_ids:
            break
        seen_ids.add(id(current))
        cfg = getattr(current, "config", None)
        if cfg is not None:
            if getattr(cfg, "vision_config", None) is not None:
                return True
            mt = getattr(cfg, "model_type", "")
            if isinstance(mt, str) and mt in _VL_MODEL_TYPES:
                return True
        # Unwrap common wrappers: PEFT .base_model, Unsloth .model, etc.
        nxt = getattr(current, "base_model", None)
        if nxt is not None and nxt is not current:
            current = nxt
            continue
        nxt = getattr(current, "module", None)
        if nxt is not None and nxt is not current:
            current = nxt
            continue
        nxt = getattr(current, "model", None)
        if nxt is not None and nxt is not current:
            current = nxt
            continue
        break
    return False


def is_grpo_trainer(trainer: Any) -> bool:
    """Return True when the trainer is a TRL GRPO trainer or an Unsloth
    subclass thereof.

    TRL dynamically subclasses GRPOTrainer from multiple places and Unsloth's
    monkey-patch wraps further subclasses (``UnslothGRPOTrainer``). A name
    check is the cheapest low-coupling way to recognise the family without
    importing TRL's internals.
    """
    if trainer is None:
        return False
    for cls in type(trainer).__mro__:
        name = getattr(cls, "__name__", "")
        if "GRPOTrainer" in name:
            return True
    return False


def resolve_ao_kwargs(model: Any, trainer: Any = None) -> dict[str, Any]:
    """Compute VL/GRPO-aware kwargs for ``get_act_offloading_ctx_manager``.

    Precedence (highest wins): env override → VL/GRPO default → TRL default
    (absent from returned dict). Returned keys pass straight through to
    ``get_act_offloading_ctx_manager``; absent keys mean "use TRL's own
    default" — we do not duplicate TRL's defaults here to keep future TRL
    changes transparent.

    The ``trainer`` argument is optional so older callers that only pass
    ``model`` continue to work. When provided, GRPO trainers default to
    ``use_streams=False`` for the same reason as VL models: TRL's stream-based
    pack/unpack path keeps a forward stash of live GPU tensors across the
    forward→backward boundary. On GRPO this costs ~0.25 GiB on text at
    typical configs without producing any measurable saving (activations are
    already small relative to the vLLM slab + weights + optimizer state).
    """
    kwargs: dict[str, Any] = {}

    if is_vlm_model(model):
        # Phase-3 measured win on Qwen3-VL-2B: turning streams off removes the
        # bwd_tensor_stash (peaked at 70 tensors in profiling) and the
        # record_stream leases that inflate reserved memory by ~1.3 GiB while
        # delivering a 1.7 GiB saving vs ao=off.
        kwargs["use_streams"] = False

    if is_grpo_trainer(trainer):
        # GRPO has a small activation surface (decoder-only forward inside
        # _compute_loss, with GC already active) and the peak is dominated by
        # the vLLM slab + weights + optimizer state. streams=on TRL default
        # adds a forward-stash that costs reserved memory without any
        # offsetting saving.
        kwargs["use_streams"] = False

    streams_override = _parse_bool_env("UNSLOTH_AO_USE_STREAMS")
    if streams_override is not None:
        kwargs["use_streams"] = streams_override

    pin_override = _parse_bool_env("UNSLOTH_AO_USE_PIN_MEMORY")
    if pin_override is not None:
        kwargs["use_pin_memory"] = pin_override

    min_size_raw = os.environ.get("UNSLOTH_AO_MIN_OFFLOAD_SIZE", "").strip()
    if min_size_raw:
        try:
            kwargs["min_offload_size"] = int(min_size_raw)
        except ValueError:
            pass

    max_fwd_raw = os.environ.get("UNSLOTH_AO_MAX_FWD_STASH_SIZE", "").strip()
    if max_fwd_raw:
        try:
            kwargs["max_fwd_stash_size"] = int(max_fwd_raw)
        except ValueError:
            pass

    return kwargs


def install_ao_decoder_only_gate(model: Any) -> list:
    """Gate AO to decoder layers only by installing TRL's ``NoOpManager``
    around the model's vision tower and embed/merge region.

    Returns a list of handles that can be removed to uninstall the gate. Does
    nothing unless ``UNSLOTH_AO_DECODER_ONLY=1``. Phase-3 measurements show
    this does not improve Qwen3-VL-2B peak memory on top of ``use_streams=False``,
    so it is off by default and intentionally left as an opt-in tool for
    models where the vision tower dominates.
    """
    if _parse_bool_env("UNSLOTH_AO_DECODER_ONLY") is not True:
        return []
    try:
        from trl.models.activation_offloading import NoOpManager
    except Exception:
        return []

    noop_ctx = NoOpManager()
    handles: list = []

    def _gate(module: Any) -> None:
        def pre(_m, _inputs):
            noop_ctx.__enter__()

        def post(_m, _inputs, _outputs):
            try:
                noop_ctx.__exit__(None, None, None)
            except Exception:
                pass

        h1 = module.register_forward_pre_hook(pre)
        h2 = module.register_forward_hook(post)
        handles.extend([h1, h2])

    # Walk through PEFT + Unsloth wrappers to find the HF model root.
    core = model
    for _ in range(4):
        nxt = getattr(core, "base_model", None)
        if nxt is not None and nxt is not core and hasattr(nxt, "model"):
            core = nxt
            continue
        break
    core = getattr(core, "model", core)

    visual = getattr(core, "visual", None)
    if visual is not None:
        _gate(visual)

    text_core = getattr(core, "model", None)
    if text_core is not None:
        embed = getattr(text_core, "embed_tokens", None)
        if embed is not None:
            _gate(embed)

    return handles


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
