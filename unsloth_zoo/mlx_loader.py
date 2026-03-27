# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Lightweight FastLanguageModel for Apple Silicon / MLX.

Supports both text-only models (via mlx-lm) and VLMs (via mlx-vlm).
Auto-detects VLMs from model config and routes to the correct backend.
"""

import os

# Known VLM model types from HuggingFace configs
_VLM_MODEL_TYPES = {
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
    "mllama", "pixtral", "gemma3", "llava", "llava_next",
    "llava_bunny", "phi3_v", "phi4_siglip", "phi4mm",
    "paligemma", "internvl_chat", "kimi_vl", "mistral3",
    "mistral4", "llama4", "idefics2", "idefics3",
    "molmo", "molmo2", "moondream3", "fastvlm",
    "florence2", "deepseek_vl_v2", "smolvlm",
    "hunyuan_vl", "lfm2_vl", "gemma3n",
}


def _is_vlm(config):
    """Check if a model config indicates a VLM."""
    if "vision_config" in config:
        return True
    model_type = config.get("model_type", "").lower()
    if model_type in _VLM_MODEL_TYPES:
        return True
    # Some VLMs nest the LM config under text_config
    if "text_config" in config and "vision_config" in config:
        return True
    return False


def _resolve_model_path(model_name, token=None):
    """Resolve a model name to a local path, downloading if needed."""
    if os.path.isdir(model_name):
        return model_name
    from huggingface_hub import snapshot_download
    kwargs = {}
    if token:
        kwargs["token"] = token
    return snapshot_download(model_name, **kwargs)


def _load_config(model_path):
    """Load config.json from a model path."""
    import json
    from pathlib import Path
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def _has_mlx_vlm():
    """Check if mlx-vlm is installed."""
    try:
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        return False


def _apply_lora_to_module(module, lora_config, target_names=None):
    """Apply LoRA to all Linear layers in a module tree.

    Unlike mlx-lm's linear_to_lora_layers which expects model.layers (decoder),
    this walks any module tree and wraps matching Linear layers with LoRALinear.

    Args:
        module: Any nn.Module to apply LoRA to.
        lora_config: Dict with rank, alpha, dropout, scale.
        target_names: Optional set of layer names to target. If None, targets
            common projection layers (qkv, proj, gate_proj, up_proj, down_proj).
    """
    from mlx_lm.tuner.lora import LoRALinear
    import mlx.nn as nn

    if target_names is None:
        target_names = {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "qkv", "proj",  # vision tower naming
        }

    r = lora_config["rank"]
    scale = lora_config.get("scale", lora_config.get("alpha", 16) / r)

    def _walk_and_replace(mod, prefix=""):
        for name, child in mod.named_modules():
            leaf_name = name.split(".")[-1] if name else ""
            if leaf_name in target_names and isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                lora_layer = LoRALinear.from_base(child, r=r, scale=scale)
                # Set on the parent
                parts = name.split(".")
                parent = mod
                for p in parts[:-1]:
                    if p.isdigit():
                        parent = parent[int(p)]
                    else:
                        parent = getattr(parent, p)
                if parts[-1].isdigit():
                    parent[int(parts[-1])] = lora_layer
                else:
                    setattr(parent, parts[-1], lora_layer)

    _walk_and_replace(module)


class FastLanguageModel:
    """Unified model loader for Apple Silicon / MLX.

    Supports text models and VLMs. Auto-detects VLMs from config.

    Usage:
        # Text model
        model, tok = FastLanguageModel.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")

        # VLM (auto-detected, requires mlx-vlm)
        model, processor = FastLanguageModel.from_pretrained("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")

        # VLM text-only (strip vision, use as text model)
        model, tok = FastLanguageModel.from_pretrained("mlx-community/Qwen2.5-VL-3B", text_only=True)
    """

    @staticmethod
    def from_pretrained(
        model_name="mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        token=None,
        trust_remote_code=False,
        text_only=None,
        **kwargs,
    ):
        """Load a model for training on Apple Silicon.

        Args:
            model_name: HuggingFace repo name or local path.
            max_seq_length: Maximum sequence length for training.
            load_in_4bit: Accepted for API compat (MLX quantization is per-repo).
            token: HuggingFace token for gated models.
            text_only: Force text-only loading (strips vision weights).
                None = auto-detect from config (default).
                True = always load text-only via mlx-lm.
                False = always load as VLM via mlx-vlm.
        """
        try:
            from mlx_lm import load as mlx_load
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        tokenizer_config = {}
        if token:
            tokenizer_config["token"] = token

        # Auto-detect VLM if text_only not specified
        if text_only is None:
            model_path = _resolve_model_path(model_name, token=token)
            config = _load_config(model_path)
            is_vlm = _is_vlm(config)
        else:
            is_vlm = not text_only

        # VLM path: use mlx-vlm for full model with vision tower
        if is_vlm:
            if not _has_mlx_vlm():
                print(
                    "Unsloth: This appears to be a VLM but mlx-vlm is not installed. "
                    "Install via: pip install mlx-vlm\n"
                    "Loading text-only via mlx-lm instead."
                )
                is_vlm = False

        if is_vlm:
            return FastLanguageModel._load_vlm(
                model_name, max_seq_length, token, **kwargs
            )
        else:
            return FastLanguageModel._load_text(
                model_name, max_seq_length, token, tokenizer_config, **kwargs
            )

    @staticmethod
    def _load_text(model_name, max_seq_length, token, tokenizer_config, **kwargs):
        """Load a text-only model via mlx-lm."""
        from mlx_lm import load as mlx_load

        print(f"Unsloth: Loading {model_name} via mlx-lm...")
        model, tokenizer = mlx_load(
            model_name,
            tokenizer_config=tokenizer_config if tokenizer_config else None,
        )
        model.max_seq_length = max_seq_length
        model._unsloth_is_vlm = False
        return model, tokenizer

    @staticmethod
    def _load_vlm(model_name, max_seq_length, token, **kwargs):
        """Load a VLM via mlx-vlm (vision tower + language model)."""
        from mlx_vlm import load as vlm_load

        print(f"Unsloth: Loading {model_name} via mlx-vlm (VLM)...")
        model, processor = vlm_load(model_name)

        model.max_seq_length = max_seq_length
        model._unsloth_is_vlm = True

        # Report model structure
        components = []
        if hasattr(model, "language_model"):
            components.append("language_model")
        if hasattr(model, "vision_tower"):
            components.append("vision_tower")
        if hasattr(model, "multi_modal_projector"):
            components.append("multi_modal_projector")
        print(f"Unsloth: VLM components: {', '.join(components)}")

        return model, processor

    @staticmethod
    def get_peft_model(
        model,
        r=16,
        target_modules=None,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=2048,
        train_vision=True,
        train_projector=False,
        **kwargs,
    ):
        """Apply LoRA to the model.

        For text models: applies LoRA to all target modules.
        For VLMs: applies LoRA to language model AND vision tower (default).
            Projector stays frozen unless train_projector=True.

        Args:
            train_vision: Apply LoRA to vision tower (VLM only). Default True.
            train_projector: Unfreeze multi-modal projector (VLM only). Default False.
        """
        try:
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except ImportError:
            raise ImportError(
                "Unsloth: mlx-lm is required for LoRA on Apple Silicon. "
                "Install via: pip install unsloth-zoo[mlx]"
            )

        import mlx.utils

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]

        lora_config = {
            "rank": r,
            "alpha": lora_alpha,
            "dropout": 0.0,
            "scale": lora_alpha / r,
        }

        is_vlm = getattr(model, "_unsloth_is_vlm", False)

        if is_vlm:
            # VLM: apply LoRA to language model
            lm = model.language_model
            num_lm_layers = 0
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                num_lm_layers = len(lm.model.layers)

            linear_to_lora_layers(lm, num_layers=num_lm_layers,
                                   config=lora_config, use_dora=False)

            # Freeze everything first
            model.freeze()

            # Unfreeze language model LoRA
            lm.unfreeze(keys=["lora_a", "lora_b"], strict=False)

            # Vision tower: LoRA if train_vision=True
            # Can't use mlx-lm's linear_to_lora_layers — it expects model.layers
            # (decoder pattern). Vision towers use blocks/encoder with different
            # structure. We apply LoRA directly by walking the module tree.
            if train_vision and hasattr(model, "vision_tower"):
                _apply_lora_to_module(model.vision_tower, lora_config)
                model.vision_tower.unfreeze(keys=["lora_a", "lora_b"], strict=False)

            # Projector: unfreeze if requested
            if train_projector:
                for name in ("multi_modal_projector", "mm_projector",
                             "connector", "aligner"):
                    if hasattr(model, name):
                        getattr(model, name).unfreeze()
                        break
        else:
            # Text model: standard LoRA
            num_layers = 0
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                num_layers = len(model.model.layers)

            linear_to_lora_layers(model, num_layers=num_layers,
                                   config=lora_config, use_dora=False)
            model.freeze()
            model.unfreeze(keys=["lora_a", "lora_b"], strict=False)

        # Report trainable params
        trainable = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
        total = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
        pct = 100.0 * trainable / total if total > 0 else 0
        print(
            f"Unsloth: LoRA applied — {trainable:,} trainable params "
            f"({pct:.2f}% of {total:,} total)"
        )
        if is_vlm:
            parts = []
            parts.append("language_model LoRA")
            if train_vision:
                parts.append("vision_tower LoRA")
            if train_projector:
                parts.append("projector unfrozen")
            print(f"Unsloth: VLM training scope: {', '.join(parts)}")

        return model


# Alias for API compat
FastModel = FastLanguageModel
