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

import torch
import torch.nn as nn
import inspect
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from .common import TEMPORARY_PATCHES, torch_compile_options
from .utils import (
    patch_function,
    process_output_options,
    process_return,
    KWARGS_TYPE,
    raise_error,
    ImageInput,
    PreTokenizedInput,
    TextInput,
    Cache,
    StaticCache,
    HybridCache,
    Unpack,
)

def patch_FalconH1Mixer_torch_forward():
    try:
        import transformers.models.falcon_h1.modeling_falcon_h1
        from transformers.models.falcon_h1.modeling_falcon_h1 import (
            FalconHybridMambaAttentionDynamicCache,
            apply_mask_to_padding_states,
            pad_tensor_by_size,
            reshape_into_chunks,
            segment_sum,
        )
    except Exception as e:
        return raise_error("FalconH1Mixer.torch_forward", e)

    def _projection(self, x):
        proj = self.in_proj(x) * self.mup_vector
        return proj.split(
            [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )  # gate, hidden_BC, dt

    def _conv_train(self, hidden_BC, seq_len: int):
        t = self.act(
            self.conv1d(hidden_BC.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        )
        return t

    def _conv_gen(self, hidden_BC, prev_state):
        # prev_state: [bs, channels, k]  (already on correct device)
        rolled = prev_state.roll(shifts=-1, dims=-1)
        rolled[:, :, -1] = hidden_BC[:, 0, :]
        out = torch.sum(rolled * self.conv1d.weight.squeeze(1), dim=-1)
        if self.use_conv_bias:
            out = out + self.conv1d.bias
        return self.act(out.unsqueeze(1)), rolled

    def _ssm_train(self, H, B, C, dt, batch_size: int):
        dt = nn.functional.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

        seq_len = H.shape[1]
        hidden_states = H.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
        C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)

        # pad_size   = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        # D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)
        pad_size = (-seq_len) % self.chunk_size          # SymInt 0 ≤ pad < chunk
        pad = hidden_states.new_zeros(
            batch_size, pad_size, hidden_states.shape[-1]
        )
        hidden_states = torch.cat((hidden_states, pad), dim=1)
        D_residual = self.D[..., None] * hidden_states

        hidden_states = hidden_states * dt[..., None]
        A = (-torch.exp(self.A_log.float())).to(hidden_states.dtype) * dt
        hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size)
                                for t in (hidden_states, A, B, C)]

        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        L  = torch.exp(segment_sum(A))
        G  = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
        M  = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states  = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

        previous_states = torch.zeros_like(states[:, :1])   # no cache in training helper
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
        decay_chunk = decay_chunk.transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]

        state_decay_out = torch.exp(A_cumsum)
        C_times_states  = C[..., None, :] * states[:, :, None, ...]
        Y_off = C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]

        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual

        y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)

        return y, ssm_state


    def _ssm_gen(self, H, B, C, dt, prev_state, batch_size):
        A = -torch.exp(self.A_log.float())
        cache_device = prev_state.device
        dt = dt[:, 0, :][:, None, ...]
        dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
        dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

        dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

        A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
        dA = torch.exp(dt[..., None] * A).to(device=cache_device)

        B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
        B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
        B = B.reshape(batch_size, -1, B.shape[-1])
        dB = dt[..., None] * B[..., None, :]

        hidden_states = H.reshape(batch_size, -1, self.head_dim)
        dBx = dB * hidden_states[..., None]

        C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
        C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
        C = C.reshape(batch_size, -1, C.shape[-1])

        ssm_states = (prev_state.to(device=C.device, dtype=C.dtype) * dA + dBx).to(C.dtype)
        ssm_flat   = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
        C_flat     = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)

        y = torch.bmm(ssm_flat, C_flat).view(batch_size, self.num_heads, self.head_dim)
        D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
        y = (y + hidden_states * D).to(y.dtype)
        y = y.reshape(batch_size, -1)[:, None, ...]    # [bs,1,intermediate]
        new_state = prev_state * dA + dBx 
        return y, new_state

    def _post(self, y, gate):
        if self.mamba_rms_norm:
            z = self.norm(y, gate)
        else:
            z = y * torch.nn.functional.silu(gate)
        return self.out_proj(z)

    _projection = torch.compile(_projection, fullgraph = True, dynamic = True, options = torch_compile_options)
    _conv_train = torch.compile(_conv_train, fullgraph = True, dynamic = True, options = torch_compile_options)
    _conv_gen = torch.compile(_conv_gen, fullgraph = True, dynamic = True, options = torch_compile_options)
    _ssm_train = torch.compile(_ssm_train, fullgraph = True, dynamic = True, options = torch_compile_options)
    _ssm_gen = torch.compile(_ssm_gen, fullgraph = True, dynamic = True, options = torch_compile_options)
    _post = torch.compile(_post, fullgraph = True, dynamic = True, options = torch_compile_options)

    def torch_forward(
        self,
        input_states,
        cache_params: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bs, seq_len, _ = input_states.shape
        streaming = (
            cache_params is not None and cache_params.has_previous_state
            and seq_len == 1 and cache_position is not None and cache_position[0] > 0
        )

        input_states = apply_mask_to_padding_states(input_states, attention_mask).mul_(self.ssm_in_multiplier)
        gate, hidden_BC, dt = _projection(self, input_states)

        if streaming:
            hidden_BC, new_conv = _conv_gen(self, hidden_BC, cache_params.conv_states[self.layer_idx])
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx].copy_(new_conv)
        else:
            hidden_BC = _conv_train(self, hidden_BC, seq_len)
            new_conv = nn.functional.pad(
                hidden_BC.transpose(1, 2), (self.conv_kernel_size - seq_len, 0)
            )
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx].copy_(new_conv)

        hidden, B, C = torch.split(
            hidden_BC,
            [self.intermediate_size,
             self.n_groups * self.ssm_state_size,
             self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        if streaming:
            y, new_ssm = _ssm_gen(self, hidden, B, C, dt,
                                       cache_params.ssm_states[self.layer_idx])
            if cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(new_ssm)
        else:
            y, new_ssm = _ssm_train(self, hidden, B, C, dt, bs)
            if cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(new_ssm)

        return _post(self, y, gate)

    return patch_function(
        transformers.models.falcon_h1.modeling_falcon_h1.FalconH1Mixer, "torch_forward", torch_forward,
    )
pass

TEMPORARY_PATCHES.append(patch_FalconH1Mixer_torch_forward)
