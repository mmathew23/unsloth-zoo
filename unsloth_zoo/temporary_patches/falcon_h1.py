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

    # def _projection(self, x):
    #     proj = self.in_proj(x) * self.mup_vector
    #     return proj.split(
    #         [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
    #     )  # gate, hidden_BC, dt

    # def _conv_train(self, hidden_BC, seq_len: int):
    #     t = self.act(
    #         self.conv1d(hidden_BC.transpose(1, 2))[..., :seq_len].transpose(1, 2)
    #     )
    #     return t

    # def _conv_gen(self, hidden_BC, prev_state):
    #     # prev_state: [bs, channels, k]  (already on correct device)
    #     rolled = prev_state.roll(shifts=-1, dims=-1)
    #     rolled[:, :, -1] = hidden_BC[:, 0, :]
    #     out = torch.sum(rolled * self.conv1d.weight.squeeze(1), dim=-1)
    #     if self.use_conv_bias:
    #         out = out + self.conv1d.bias
    #     return self.act(out.unsqueeze(1)), rolled

    # def _ssm_train(self, H, B, C, dt, batch_size: int):
    #     dt = nn.functional.softplus(dt + self.dt_bias)
    #     dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

    #     seq_len = H.shape[1]
    #     hidden_states = H.reshape(batch_size, seq_len, -1, self.head_dim).float()
    #     B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
    #     C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
    #     B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2)
    #     C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2)

    #     # pad_size   = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
    #     # D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)
    #     pad_size = (-seq_len) % self.chunk_size          # SymInt 0 ≤ pad < chunk
    #     pad = hidden_states.new_zeros(
    #         batch_size, pad_size, hidden_states.shape[-2], hidden_states.shape[-1]
    #     )
    #     hidden_states_padded = torch.cat((hidden_states, pad), dim=1)
    #     D_residual = self.D[..., None] * hidden_states_padded

    #     hidden_states = hidden_states * dt[..., None]
    #     A = (-torch.exp(self.A_log.float())).to(hidden_states.dtype) * dt
    #     hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size)
    #                             for t in (hidden_states, A, B, C)]

    #     A = A.permute(0, 3, 1, 2)
    #     A_cumsum = torch.cumsum(A, dim=-1)

    #     L  = torch.exp(segment_sum(A))
    #     G  = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)
    #     M  = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
    #     Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

    #     decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    #     B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    #     states  = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

    #     previous_states = torch.zeros_like(states[:, :1])   # no cache in training helper
    #     states = torch.cat([previous_states, states], dim=1)
    #     decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
    #     decay_chunk = decay_chunk.transpose(1, 3)
    #     new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    #     states, ssm_state = new_states[:, :-1], new_states[:, -1]

    #     state_decay_out = torch.exp(A_cumsum)
    #     C_times_states  = C[..., None, :] * states[:, :, None, ...]
    #     Y_off = C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]

    #     y = Y_diag + Y_off
    #     y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
    #     y = y + D_residual

    #     y = y[:, :seq_len, :, :]
    #     y = y.reshape(batch_size, seq_len, -1)

    #     return y, ssm_state


    # def _ssm_gen(self, H, B, C, dt, prev_state, batch_size):
    #     A = -torch.exp(self.A_log.float())
    #     cache_device = prev_state.device
    #     dt = dt[:, 0, :][:, None, ...]
    #     dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
    #     dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

    #     dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
    #     dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

    #     A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
    #     dA = torch.exp(dt[..., None] * A).to(device=cache_device)

    #     B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
    #     B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
    #     B = B.reshape(batch_size, -1, B.shape[-1])
    #     dB = dt[..., None] * B[..., None, :]

    #     hidden_states = H.reshape(batch_size, -1, self.head_dim)
    #     dBx = dB * hidden_states[..., None]

    #     C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
    #     C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
    #     C = C.reshape(batch_size, -1, C.shape[-1])

    #     ssm_states = (prev_state.to(device=C.device, dtype=C.dtype) * dA + dBx).to(C.dtype)
    #     ssm_flat   = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
    #     C_flat     = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)

    #     y = torch.bmm(ssm_flat, C_flat).view(batch_size, self.num_heads, self.head_dim)
    #     D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
    #     y = (y + hidden_states * D).to(y.dtype)
    #     y = y.reshape(batch_size, -1)[:, None, ...]    # [bs,1,intermediate]
    #     new_state = prev_state * dA + dBx 
    #     return y, new_state

    # def _post(self, y, gate):
    #     if self.mamba_rms_norm:
    #         z = self.norm(y, gate)
    #     else:
    #         z = y * torch.nn.functional.silu(gate)
    #     return self.out_proj(z)

    # _projection = torch.compile(_projection, fullgraph = True, dynamic = True, options = torch_compile_options)
    # _conv_train = torch.compile(_conv_train, fullgraph = True, dynamic = True, options = torch_compile_options)
    # _conv_gen = torch.compile(_conv_gen, fullgraph = True, dynamic = True, options = torch_compile_options)
    # _ssm_train = torch.compile(_ssm_train, fullgraph = True, dynamic = True, options = torch_compile_options)
    # _ssm_gen = torch.compile(_ssm_gen, fullgraph = True, dynamic = True, options = torch_compile_options)
    # _post = torch.compile(_post, fullgraph = True, dynamic = True, options = torch_compile_options)

    # def torch_forward(
    #     self,
    #     input_states,
    #     cache_params: Optional[FalconHybridMambaAttentionDynamicCache] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ):
    #     bs, seq_len, _ = input_states.shape
    #     streaming = (
    #         cache_params is not None and cache_params.has_previous_state
    #         and seq_len == 1 and cache_position is not None and cache_position[0] > 0
    #     )

    #     input_states = apply_mask_to_padding_states(input_states, attention_mask).mul_(self.ssm_in_multiplier)
    #     gate, hidden_BC, dt = _projection(self, input_states)

    #     if streaming:
    #         hidden_BC, new_conv = _conv_gen(self, hidden_BC, cache_params.conv_states[self.layer_idx])
    #         if cache_params is not None:
    #             cache_params.conv_states[self.layer_idx].copy_(new_conv)
    #     else:
    #         hidden_BC = _conv_train(self, hidden_BC, seq_len)
    #         new_conv = nn.functional.pad(
    #             hidden_BC.transpose(1, 2), (self.conv_kernel_size - seq_len, 0)
    #         )
    #         if cache_params is not None:
    #             cache_params.conv_states[self.layer_idx].copy_(new_conv)

    #     hidden, B, C = torch.split(
    #         hidden_BC,
    #         [self.intermediate_size,
    #          self.n_groups * self.ssm_state_size,
    #          self.n_groups * self.ssm_state_size],
    #         dim=-1,
    #     )

    #     if streaming:
    #         y, new_ssm = _ssm_gen(self, hidden, B, C, dt,
    #                                    cache_params.ssm_states[self.layer_idx])
    #         if cache_params is not None:
    #             cache_params.ssm_states[self.layer_idx].copy_(new_ssm)
    #     else:
    #         y, new_ssm = _ssm_train(self, hidden, B, C, dt, bs)
    #         if cache_params is not None:
    #             cache_params.ssm_states[self.layer_idx].copy_(new_ssm)

    #     return _post(self, y, gate)
    def _get_data_hidden_states_dt(self, input_states):
        input_states = input_states * self.ssm_in_multiplier
        projected_states = self.in_proj(input_states)
        projected_states = projected_states * self.mup_vector  # ADD Mup Multipliers
        gate, hidden_states_B_C, dt = projected_states.split([
                self.intermediate_size, self.conv_dim, self.num_heads
            ], dim=-1)
        return gate, hidden_states_B_C, dt

    def _conv1d(self, hidden_states_B_C, seq_len):
        hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        return hidden_states_B_C

    def _kern_dt_and_A_and_hs(self, dt, A_log, hs, time_lim):
        """
        dt:  (B, C, Lc, 1)   after broadcasting into chunk shape
        A_log: (H,)          constant
        Return:
          dt_scaled  (B, C, Lc, 1)
          A_scaled   (B, C, Lc, H)
        """
        dt = torch.nn.functional.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, time_lim[0], time_lim[1])
        hs = hs * dt[..., None]
        A  = -torch.exp(A_log).to(torch.float32) * dt               # broadcast over (B,C,L).dt
        return dt, A, hs

    def _kern_intra_chunk(self, hs, B, C, A, dt):
        """
        Inputs all shaped (B, C, Lc, H, ?):
          hs: (B,C,Lc,H,D)
          B : (B,C,Lc,H,S)   S = ssm_state_size
          C : (B,C,Lc,H,S)
          A : (B,C,Lc,H)     after permute -> (B,H,C,Lc)
          dt: (B,C,Lc,1)
        Returns:
          Y_diag  (B,C,Lc,H,D)
          states  (B,C,H,S)
          A_cum   (B,H,C,Lc)
        """
        # rearrange A  ------------------------------------------------------
        A = A.permute(0, 3, 1, 2)                               # (B,H,C,L)
        A_cumsum = torch.cumsum(A, dim=-1)                      # for later

        # ----------- diagonal blocks (mask) -------------------------------
        L = torch.exp(segment_sum(A))

        G = (C[:, :, :, None, :, :] * B[:, :, None, :, :, :]).sum(dim=-1)      # (B,C,Lc,H)
        M = (G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]).sum(dim=-1)
        Y_diag = (M[..., None] * hs[:, :, None]).sum(dim=3)     # (B,C,Lc,H,D)

        # ------------- decay states inside chunk --------------------------
        decay_states = torch.exp(A_cumsum[:, :, :, -1] - A_cumsum)           # (B,H,C,L)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * hs[..., None]).sum(dim=2)       # (B,C,H,S)

        return Y_diag, states, A_cumsum          # keep A_cumsum for inter‑chunk

    def _kern_inter_chunk(
        self,
        states,
        A_cumsum,
        C_chunks,
    ):
        """
        states   : (B,C,H,S)
        A_cumsum : (B,H,C,Lc)
        C_chunks : (B,C,Lc,H,S)

        Returns:
           Y_off : (B,C,Lc,H,D)
           ssm_state_out : (B,H,S)
        """
        # decay from previous chunk boundary -------------------------------
        padded_A_cumsum = torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1,0))
        decay_chunk = torch.exp(segment_sum(padded_A_cumsum)).transpose(1, 3)
        print(decay_chunk.shape, states.shape, padded_A_cumsum.shape, A_cumsum.shape)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]

        state_decay_out = torch.exp(A_cumsum)                          # (B,H,C,Lc)
        C_times_states = (C_chunks[..., None, :] *
                          states[:, :, None, ...])                     # (B,C,Lc,H,S)
        Y_off = (C_times_states.sum(-1) *
                 state_decay_out.permute(0, 2, 3, 1)[..., None])       # (B,C,Lc,H,D)
        return Y_off, ssm_state

    _get_data_hidden_states_dt = torch.compile(_get_data_hidden_states_dt, fullgraph = True, dynamic = True, options = torch_compile_options)
    _conv1d = torch.compile(_conv1d, fullgraph = True, dynamic = True, options = torch_compile_options)
    _kern_dt_and_A_and_hs = torch.compile(_kern_dt_and_A_and_hs, fullgraph = True, dynamic = True, options = torch_compile_options)
    _kern_intra_chunk = torch.compile(_kern_intra_chunk, fullgraph = True, dynamic = True, options = torch_compile_options)
    _kern_inter_chunk = torch.compile(_kern_inter_chunk, fullgraph = False, dynamic = True, options = torch_compile_options)

    def torch_forward(
        self,
        input_states,
        cache_params: Optional[FalconHybridMambaAttentionDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, attention_mask)
        # Add Multipliers
        # input_states = input_states * self.ssm_in_multiplier
        # projected_states = self.in_proj(input_states)
        # projected_states = projected_states * self.mup_vector  # ADD Mup Multipliers
        # gate, hidden_states_B_C, dt = projected_states.split([
        #         self.intermediate_size, self.conv_dim, self.num_heads
        #     ], dim=-1)
        gate, hidden_states_B_C, dt = _get_data_hidden_states_dt(self, input_states)

        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
            and cache_position is not None
            and cache_position[0] > 0
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            cache_params.conv_states[self.layer_idx] = cache_params.conv_states[self.layer_idx].roll(shifts=-1, dims=-1)
            cache_params.conv_states[self.layer_idx][:, :, -1] = hidden_states_B_C[:, 0, :].to(cache_params.conv_states[self.layer_idx].device)

            # We need to guarantee that anything regarding the cache is on the same device
            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)

            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Init cache
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = nn.functional.pad(
                    hidden_states_B_C_transposed, (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_states)

            # hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))
            hidden_states_B_C = _conv1d(self, hidden_states_B_C, seq_len)

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1
        )

        # 3. SSM transformation
        if use_precomputed_states:
            A = -torch.exp(self.A_log.float())                            # [num_heads]
            # We need to guarantee that anything regarding the cache is on the same device
            cache_device = cache_params.ssm_states[self.layer_idx].device

            # Note: there is no need to pad parameter matrices here, as there is just one new token
            # for batched generation
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            # [num_heads] -> [num_heads, head_dim]
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            # [bsz, num_heads, head_dim, state_size]
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

            # Discretize B
            # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
            # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            # [bsz, num_heads, head_dim, state_size]
            dB = dt[..., None] * B[..., None, :]

            # Discretize x into dB
            # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            # State calculation
            cache_params.ssm_states[self.layer_idx].copy_(
                cache_params.ssm_states[self.layer_idx] * dA + dBx
            )

            # Subsequent output
            # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            # [bsz, num_heads, head_dim]

            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
            # Reshape ssm_states to merge the first two dimensions
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)  # Shape: [b*h, d, n]
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)  # Shape: [b*h, n, 1]
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            # D skip connection
            # [num_heads] -> [num_heads, head_dim]
            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)

            # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # A = -torch.exp(self.A_log.float())                            # [num_heads]
            # # begin ssd naive implementation without einsums
            # dt = nn.functional.softplus(dt + self.dt_bias)
            # dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            # hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            # B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            # C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            # B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            # C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            # pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            # D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            # # Discretize x and A
            # hidden_states = hidden_states * dt[..., None]
            # A = A.to(hidden_states.dtype) * dt

            # # Rearrange into blocks/chunks
            # hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]

            # # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
            # A = A.permute(0, 3, 1, 2)
            # A_cumsum = torch.cumsum(A, dim=-1)

            # # 1. Compute the output for each intra-chunk (diagonal blocks)
            # # This is the analog of a causal mask
            # L = torch.exp(segment_sum(A))

            # # Contraction of C and B to get G (attention-weights like)
            # G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
            # G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

            # # Compute M, equivalent to applying attention mask to weights
            # M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            # M = M_intermediate.sum(dim=-1)

            # # Compute Y_diag (apply to values)
            # Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # # 2. Compute the state for each intra-chunk
            # # (right term of low-rank factorization of off-diagonal blocks; B terms)
            # decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            # B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            # states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            # # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
            # # (middle term of factorization of off-diag blocks; A terms)
            # if use_precomputed_states:
            #     previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            # else:
            #     previous_states = torch.zeros_like(states[:, :1])
            # states = torch.cat([previous_states, states], dim=1)
            # decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
            # decay_chunk = decay_chunk.transpose(1, 3)
            # new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            # states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # # 4. Compute state -> output conversion per chunk
            # # (left term of low-rank factorization of off-diagonal blocks; C terms)
            # state_decay_out = torch.exp(A_cumsum)
            # C_times_states = (C[..., None, :] * states[:, :, None, ...])
            # state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            # Y_off = (C_times_states.sum(-1) * state_decay_out_permuted[..., None])

            # # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
            # y = Y_diag + Y_off
            # # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
            # y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)

            # y = y + D_residual
            # # Cutting off padded chunks
            # if pad_size > 0:
            #     y = y[:, :seq_len, :, :]
            # y = y.reshape(batch_size, seq_len, -1)
            H, D        = self.num_heads, self.head_dim
            S           = self.ssm_state_size
            pad_size    = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            hs = hidden_states.view(batch_size, seq_len, H, D).float()                # (B,L,H,D)
            B  = B.view(batch_size, seq_len, self.n_groups, S).float()                # (B,L,G,S)
            C  = C.view(batch_size, seq_len, self.n_groups, S).float()
            A_log = self.A_log.float()
            heads_per_group = H // self.n_groups
            B = B.repeat_interleave(heads_per_group, dim=2, output_size=H)    # (B,L,H,S)
            C = C.repeat_interleave(heads_per_group, dim=2, output_size=H)

            D_residual = self.D[..., None] * pad_tensor_by_size(hs, pad_size)

            dt_scaled, A, hs = _kern_dt_and_A_and_hs(
                self, dt, A_log, hs, self.time_step_limit
            )
            print('ashape', A.shape)

            # ---- 2. chunkify -------------------------------------------------
            hs, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size)
                                for t in (hs, A, B, C)]

            print(hs.shape, A.shape, B.shape, C.shape)
            # ---- 4. compiled (B) intra‑chunk SSM -----------------------------
            Y_diag, states_chunks, A_cumsum = _kern_intra_chunk(
                self, hs, B, C, A, dt_scaled
            )                                    # Y_diag: (B,C,Lc,H,D)
            print(Y_diag.shape, states_chunks.shape, A_cumsum.shape)

            # ---- 5. cache / previous states (still eager) --------------------
            if use_precomputed_states:
                prev_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=hidden_states.device)
            else:
                prev_states = torch.zeros_like(states_chunks[:, :1])
            print('prev_states.shape', prev_states.shape, 'states_chunks.shape', states_chunks.shape)
            states_chunks = torch.cat([prev_states, states_chunks], dim=1)  # prepend
            print(states_chunks.shape)

            # ---- 6. compiled (C) inter‑chunk + output ------------------------
            Y_off, ssm_state = _kern_inter_chunk(
                self, states_chunks, A_cumsum, C
            )                                    # (B,C,Lc,H,D) / (B,H,S)

            # ---- 7. combine + unchunk + final pad slice ----------------------
            y = Y_diag + Y_off                                   # (B,C,Lc,H,D)
            y = y.view(batch_size, -1, H, D)                     # (B,Lpad,H,D)
            y = y + D_residual
            if pad_size:
                y = y[:, :seq_len]                               # remove pad
            y = y.view(batch_size, seq_len, -1)
            # Init cache
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        if self.mamba_rms_norm:
            scan_output = self.norm(y, gate)
        else:
            scan_output = y * torch.nn.functional.silu(gate)


        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.to(dtype))  # [batch, seq_len, hidden_size]
        return contextualized_states


    # only patch if bf16 is not supported
    major_version, minor_version = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = (major_version >= 8)
    if not SUPPORTS_BFLOAT16:
        return patch_function(
            transformers.models.falcon_h1.modeling_falcon_h1.FalconH1Mixer, "torch_forward", torch_forward,
        )
    else:
        return True  # return True if bf16 is not supported since we don't need to patch
pass

TEMPORARY_PATCHES.append(patch_FalconH1Mixer_torch_forward)
