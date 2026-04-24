# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import torch
import torch.nn as nn
from .common import TEMPORARY_PATCHES, UNSLOTH_ENABLE_LOGGING
from .utils import patch_function, process_return, raise_error, logger
from .moe_utils import (
    patch_param_wrapper_for_moe,
    get_forward_moe_backend,
)


def patch_gemma4_grpo_hidden_states():
    """Patch Gemma4 GRPO logprob calls to return hidden states on demand.

    This path is independent from any MoE layout changes. Keeping it separate
    avoids losing the GRPO memory optimization when MoE internals change across
    Transformers versions.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
            Gemma4CausalLMOutputWithPast,
        )
    except Exception:
        return

    if getattr(Gemma4ForConditionalGeneration, "_unsloth_grpo_hidden_states_patched", False):
        return

    _original_causal_lm_forward = Gemma4ForConditionalGeneration.forward

    def _patched_causal_lm_forward(
        self,
        input_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        position_ids=None,
        image_position_ids=None,
        video_position_ids=None,
        past_key_values=None,
        mm_token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        # Inject mm_token_type_ids=0 for text-only SFT.
        if mm_token_type_ids is None and self.training:
            _ids = input_ids if input_ids is not None else inputs_embeds
            if _ids is not None:
                mm_token_type_ids = torch.zeros(
                    _ids.shape[:2], dtype=torch.long, device=_ids.device
                )

        # Drop stale mm_token_type_ids during KV-cache generation.
        _seq_ref = input_ids if input_ids is not None else inputs_embeds
        if mm_token_type_ids is not None and _seq_ref is not None:
            if mm_token_type_ids.shape[1] != _seq_ref.shape[1]:
                mm_token_type_ids = None

        return_hidden_states = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
        if not return_hidden_states:
            return _original_causal_lm_forward(
                self,
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                input_features=input_features,
                attention_mask=attention_mask,
                input_features_mask=input_features_mask,
                position_ids=position_ids,
                image_position_ids=image_position_ids,
                video_position_ids=video_position_ids,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            image_position_ids=image_position_ids,
            video_position_ids=video_position_ids,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        sliced_hidden_states = hidden_states[:, slice_indices, :]

        return process_return(
            Gemma4CausalLMOutputWithPast,
            {
                "loss": None,
                "logits": sliced_hidden_states,
                "past_key_values": outputs.past_key_values,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
                "image_hidden_states": getattr(outputs, "image_hidden_states", None),
                "audio_hidden_states": getattr(outputs, "audio_hidden_states", None),
            },
        )

    _patched_causal_lm_forward.__qualname__ = _original_causal_lm_forward.__qualname__
    _patched_causal_lm_forward.__name__ = _original_causal_lm_forward.__name__
    _patched_causal_lm_forward.__doc__ = _original_causal_lm_forward.__doc__
    _patched_causal_lm_forward.__wrapped__ = _original_causal_lm_forward
    Gemma4ForConditionalGeneration.forward = _patched_causal_lm_forward
    Gemma4ForConditionalGeneration._unsloth_grpo_hidden_states_patched = True
    if UNSLOTH_ENABLE_LOGGING:
        logger.info(
            "Unsloth: Patched Gemma4ForConditionalGeneration.forward for GRPO hidden states."
        )


def _patch_gemma4_moe_legacy():
    """Patch the older Gemma4TextMoEBlock layout used by some releases."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextMoEBlock,
            Gemma4TextDecoderLayer,
        )
    except Exception:
        return False

    if hasattr(Gemma4TextMoEBlock, "_unsloth_already_patched"):
        return True

    def _patched_decoder_init(self, config, layer_idx):
        _original_decoder_init(self, config, layer_idx)
        if getattr(self, "enable_moe_block", False) and "moe" in self._modules:
            moe_block = self._modules.pop("moe")
            self._modules["experts"] = moe_block
            object.__setattr__(self, "moe", moe_block)

            per_expert_scale_data = moe_block.per_expert_scale.data
            del moe_block._parameters["per_expert_scale"]
            self.router.per_expert_scale = nn.Parameter(per_expert_scale_data)
            # Non-persistent buffer keeps _init_weights happy without appearing in state_dict
            moe_block.register_buffer("per_expert_scale", torch.ones(config.num_experts), persistent=False)
            object.__setattr__(moe_block, "_router_ref", self.router)

    _original_decoder_init = Gemma4TextDecoderLayer.__init__
    Gemma4TextDecoderLayer.__init__ = _patched_decoder_init

    _moe_backend = get_forward_moe_backend()

    def _gemma4_moe_forward(self, hidden_states, top_k_index, top_k_weights):
        # Fold per_expert_scale into routing weights before grouped_mm
        router_ref = getattr(self, "_router_ref", None)
        if router_ref is not None:
            pes = router_ref.per_expert_scale
            top_k_weights = top_k_weights * pes[top_k_index].to(top_k_weights.dtype)
        return _moe_backend(self, hidden_states, top_k_index, top_k_weights)

    patch_function(Gemma4TextMoEBlock, "forward", _gemma4_moe_forward, force=True)
    Gemma4TextMoEBlock._unsloth_already_patched = True
    return True


def _patch_gemma4_moe_current():
    """Patch the current Gemma4 router+experts layout used by latest Transformers."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextExperts,
        )
    except Exception:
        return False

    if hasattr(Gemma4TextExperts, "_unsloth_already_patched"):
        return True

    _moe_backend = get_forward_moe_backend()

    def _gemma4_experts_forward(self, hidden_states, top_k_index, top_k_weights):
        # Current Transformers Gemma4 already folds per_expert_scale into
        # top_k_weights inside Gemma4TextRouter.forward.
        return _moe_backend(self, hidden_states, top_k_index, top_k_weights)

    patch_function(Gemma4TextExperts, "forward", _gemma4_experts_forward, force=True)
    Gemma4TextExperts._unsloth_already_patched = True
    return True


def patch_gemma4_moe():
    """Patch Gemma4 MoE to support Split LoRA using grouped GEMM.

    Supports both the legacy Gemma4TextMoEBlock layout and the current
    Gemma4TextExperts/Gemma4TextRouter layout used by recent Transformers.
    """
    patch_param_wrapper_for_moe()
    patched = _patch_gemma4_moe_current()
    if not patched:
        patched = _patch_gemma4_moe_legacy()

    if patched and UNSLOTH_ENABLE_LOGGING:
        logger.info("Unsloth: Patched Gemma4 MoE for Split LoRA support.")


TEMPORARY_PATCHES.append(patch_gemma4_grpo_hidden_states)
TEMPORARY_PATCHES.append(patch_gemma4_moe)
