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

"""Temporary patches for Qwen3-VL models.

Bundles per-layer compiled functions into 2 chunks (prepare_attn + prepare_mlp)
instead of 8-9 scattered @torch.compile decorators. This reduces compile memory
overhead from ~9.7 GiB to ~0.7 GiB while improving speed by ~10%.

The scattered decorators create redundant compiled autograd boundaries, each saving
input tensors for backward. Bundling into 2 functions per layer eliminates the
redundant boundaries and gives inductor larger graphs for better fusion.
"""

from typing import Optional
import torch
import torch.nn as nn
from .common import TEMPORARY_PATCHES
from .utils import raise_error


def patch_Qwen3VLTextDecoderLayer():
    """Replace Qwen3VLTextDecoderLayer.forward with 2-chunk compiled version."""
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb,
            ALL_ATTENTION_FUNCTIONS,
            eager_attention_forward,
        )
    except Exception as e:
        return raise_error("Qwen3VLTextDecoderLayer", e)

    from functools import partial as _partial
    _compile = _partial(torch.compile, dynamic=True)

    # ── Compiled chunk 1: input_norm + QKV projection + QKNorm + RoPE ──
    def prepare_attn(hidden_states, input_layernorm, q_proj, k_proj, v_proj,
                     q_norm, k_norm, head_dim, cos, sin):
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        query_states = q_norm(q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = k_norm(k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if value_states.dtype != query_states.dtype:
            value_states = value_states.to(query_states.dtype)
        return residual, query_states, key_states, value_states, input_shape

    prepare_attn = _compile(prepare_attn, fullgraph=False)

    # ── Compiled chunk 2: post_attention_norm + MLP ──
    def prepare_mlp(hidden_states, post_attention_layernorm, mlp):
        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)
        hidden_states = mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    prepare_mlp = _compile(prepare_mlp, fullgraph=False)

    # ── Factory for the patched layer forward ──
    def _make_forward(compiled_prepare_attn, compiled_prepare_mlp,
                      attn_functions, default_attn):
        def forward(self, hidden_states, position_embeddings=None,
                    attention_mask=None, position_ids=None,
                    past_key_values=None, use_cache=False,
                    cache_position=None, **kwargs):
            cos, sin = position_embeddings

            # Compiled chunk 1: norm + QKV + QKNorm + RoPE
            residual, query_states, key_states, value_states, input_shape = \
                compiled_prepare_attn(
                    hidden_states, self.input_layernorm,
                    self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj,
                    self.self_attn.q_norm, self.self_attn.k_norm,
                    self.self_attn.head_dim, cos, sin)

            # Uncompiled: KV cache + attention + output proj
            if past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.self_attn.layer_idx, cache_kwargs)

            if self.self_attn.config._attn_implementation != "eager":
                attention_interface = attn_functions[self.self_attn.config._attn_implementation]
            else:
                attention_interface = default_attn

            attn_output, _ = attention_interface(
                self.self_attn, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.self_attn.training else self.self_attn.attention_dropout,
                scaling=self.self_attn.scaling, **kwargs)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.self_attn.o_proj(attn_output)
            hidden_states = residual + attn_output

            # Compiled chunk 2: post_norm + MLP
            hidden_states = compiled_prepare_mlp(
                hidden_states, self.post_attention_layernorm, self.mlp)

            return hidden_states
        return forward

    modeling.Qwen3VLTextDecoderLayer.forward = _make_forward(
        prepare_attn, prepare_mlp, ALL_ATTENTION_FUNCTIONS, eager_attention_forward)
pass
TEMPORARY_PATCHES.append(patch_Qwen3VLTextDecoderLayer)
