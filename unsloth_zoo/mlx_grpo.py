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
GRPO (Group Relative Policy Optimization) for MLX.

Implements the GRPO training loop with support for GRPO, DAPO, DR-GRPO,
and BNPO loss types. Matches the unsloth PyTorch GRPOTrainer interface.

Usage:
    from unsloth_zoo.mlx_grpo import MLXGRPOTrainer, MLXGRPOConfig

    trainer = MLXGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=MLXGRPOConfig(num_generations=8, max_completion_length=256),
    )
    trainer.train()
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .mlx_trainer import MLXTrainingConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MLXGRPOConfig(MLXTrainingConfig):
    """Configuration for GRPO training on MLX.

    Extends MLXTrainingConfig with GRPO-specific parameters.
    Parameter names match TRL's GRPOConfig for compatibility.
    """
    # Generation
    num_generations: int = 8
    max_prompt_length: int = 512
    max_completion_length: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0

    # Loss
    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None
    loss_type: str = "dapo"
    scale_rewards: str = "group"
    num_iterations: int = 1
    importance_sampling_level: str = "token"
    mask_truncated_completions: bool = False
    reward_weights: Optional[List[float]] = None

    # Training
    steps_per_generation: Optional[int] = None


# ---------------------------------------------------------------------------
# Core loss function
# ---------------------------------------------------------------------------

def grpo_loss(
    per_token_logps,
    old_logps,
    ref_logps,
    advantages,
    completion_mask,
    beta=0.0,
    epsilon_low=0.2,
    epsilon_high=0.2,
    loss_type="dapo",
    importance_sampling_level="token",
    max_completion_length=256,
    num_items_in_batch=None,
    delta=None,
):
    """Compute GRPO loss.

    Args:
        per_token_logps: [B, T] current policy log probs for completion tokens
        old_logps: [B, T] old policy log probs (from generation time)
        ref_logps: [B, T] reference model log probs (or None if beta=0)
        advantages: [B] per-sequence advantages
        completion_mask: [B, T] bool mask for valid completion tokens
        beta: KL penalty coefficient
        epsilon_low: Lower clipping bound
        epsilon_high: Upper clipping bound
        loss_type: "grpo", "dapo", "dr_grpo", "bnpo"
        importance_sampling_level: "token" or "sequence"
        max_completion_length: for dr_grpo normalization
        num_items_in_batch: for dapo normalization
        delta: optional upper bound for unclipped ratio (two-sided GRPO)

    Returns:
        (loss, metrics_dict) where metrics_dict contains kl, completion_length, etc.
    """
    if advantages.ndim == 1:
        advantages = advantages[:, None]  # [B, 1] for broadcasting

    # Importance sampling ratio
    if old_logps is not None:
        log_ratio = per_token_logps - mx.stop_gradient(old_logps)
    else:
        log_ratio = per_token_logps - mx.stop_gradient(per_token_logps)

    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        seq_log_ratio = (log_ratio * completion_mask).sum(-1) / mx.maximum(completion_mask.sum(-1), 1.0)
        log_importance_weights = seq_log_ratio[:, None]  # [B, 1]
    else:
        raise ValueError(f"Unknown importance_sampling_level: {importance_sampling_level}")

    coef_1 = mx.exp(log_importance_weights)

    # KL divergence (reverse KL, low variance estimator from GRPO paper)
    if beta != 0.0 and ref_logps is not None:
        kl_i = mx.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1.0
    else:
        kl_i = mx.zeros_like(per_token_logps)

    # PPO-style clipped loss
    coef_2 = mx.clip(coef_1, 1 - epsilon_low, 1 + epsilon_high)

    if delta is not None:
        loss_1 = mx.clip(coef_1, a_max=delta) * advantages
    else:
        loss_1 = coef_1 * advantages

    loss_2 = coef_2 * advantages
    per_token_loss = -mx.minimum(loss_1, loss_2)

    # Add KL penalty
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_i

    # Aggregate by loss type
    mask_f = completion_mask.astype(mx.float32)
    n_mask_per_seq = mask_f.sum(-1)
    batch_size = per_token_logps.shape[0]

    if loss_type == "grpo":
        loss = ((per_token_loss * mask_f).sum(-1) / mx.maximum(n_mask_per_seq, 1.0)).mean()
    elif loss_type == "dapo":
        normalizer = num_items_in_batch if num_items_in_batch is not None else batch_size
        loss = (per_token_loss * mask_f).sum() / normalizer
    elif loss_type == "dr_grpo":
        loss = (per_token_loss * mask_f).sum() / (batch_size * max_completion_length)
    elif loss_type == "bnpo":
        loss = (per_token_loss * mask_f).sum() / mx.maximum(mask_f.sum(), 1.0)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Metrics
    completion_length = n_mask_per_seq.mean()
    if kl_i.shape[-1] == 1:  # sequence-level
        mean_kl = kl_i.mean()
    else:
        mean_kl = ((kl_i * mask_f).sum(-1) / mx.maximum(n_mask_per_seq, 1.0)).mean()

    # Clip ratios for logging
    clip_low = (coef_1 < (1 - epsilon_low)).astype(mx.float32)
    clip_high = (coef_1 > (1 + epsilon_high)).astype(mx.float32)
    clip_ratio_low = (clip_low * mask_f).sum() / mx.maximum(mask_f.sum(), 1.0)
    clip_ratio_high = (clip_high * mask_f).sum() / mx.maximum(mask_f.sum(), 1.0)

    metrics = {
        "kl": mean_kl,
        "completion_length": completion_length,
        "clip_ratio_low": clip_ratio_low,
        "clip_ratio_high": clip_ratio_high,
    }

    ntoks = mask_f.sum()
    return loss, ntoks, metrics


# ---------------------------------------------------------------------------
# Log probability computation
# ---------------------------------------------------------------------------

def compute_per_token_logps(model, input_ids, completion_start, completion_mask):
    """Compute per-token log probabilities for completion tokens.

    Args:
        model: The language model.
        input_ids: [B, T_total] full input (prompt + completion).
        completion_start: int, index where completion begins in the sequence.
        completion_mask: [B, T_completion] mask for valid completion tokens.

    Returns:
        [B, T_completion] per-token log probs.
    """
    logits = model(input_ids)
    # Slice logits for completion positions (shifted by 1 for next-token prediction)
    # logits[:, t, :] predicts token at position t+1
    # For completion starting at position S, we need logits[:, S-1:S-1+T_comp, :]
    comp_logits = logits[:, completion_start - 1:completion_start - 1 + completion_mask.shape[1], :]

    # Log softmax
    log_probs = comp_logits - mx.logsumexp(comp_logits, axis=-1, keepdims=True)

    # Gather log probs for the actual completion tokens
    completion_ids = input_ids[:, completion_start:completion_start + completion_mask.shape[1]]
    per_token_logps = mx.take_along_axis(
        log_probs, completion_ids[:, :, None], axis=-1
    ).squeeze(-1)

    return per_token_logps


def chunked_compute_per_token_logps(model, input_ids, completion_start,
                                      completion_mask, n_chunks=4):
    """Memory-efficient per-token log prob computation using chunking.

    Splits the hidden→logit→log_softmax computation into chunks to avoid
    materializing the full [B, T, V] logit tensor at once.
    """
    # Get hidden states from model backbone
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Standard text model
        hidden = model.model(input_ids[:, :completion_start + completion_mask.shape[1]])
    elif hasattr(model, "language_model"):
        # VLM
        hidden = model.language_model.model(input_ids[:, :completion_start + completion_mask.shape[1]])
    else:
        # Fallback: use full forward
        return compute_per_token_logps(model, input_ids, completion_start, completion_mask)

    # Get LM head weight
    if hasattr(model, "lm_head"):
        lm_weight = model.lm_head.weight
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        lm_weight = model.model.embed_tokens.weight
    else:
        return compute_per_token_logps(model, input_ids, completion_start, completion_mask)

    # Slice hidden states for completion positions
    comp_hidden = hidden[:, completion_start - 1:completion_start - 1 + completion_mask.shape[1], :]
    completion_ids = input_ids[:, completion_start:completion_start + completion_mask.shape[1]]

    B, T, D = comp_hidden.shape
    chunk_size = max(1, (T + n_chunks - 1) // n_chunks)
    all_logps = []

    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        chunk_hidden = comp_hidden[:, i:end, :]
        chunk_logits = chunk_hidden @ lm_weight.T
        chunk_log_probs = chunk_logits - mx.logsumexp(chunk_logits, axis=-1, keepdims=True)
        chunk_ids = completion_ids[:, i:end, None]
        selected = mx.take_along_axis(chunk_log_probs, chunk_ids, axis=-1).squeeze(-1)
        all_logps.append(selected)

    return mx.concatenate(all_logps, axis=1)


# ---------------------------------------------------------------------------
# LoRA disable for reference model
# ---------------------------------------------------------------------------

@contextmanager
def disable_lora(model):
    """Temporarily disable LoRA by setting scale to 0.

    Used to compute reference model log probabilities from the base model.
    """
    try:
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError:
        yield
        return

    saved_scales = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            saved_scales[id(module)] = module.scale
            module.scale = 0.0
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear) and id(module) in saved_scales:
                module.scale = saved_scales[id(module)]


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(rewards, num_generations, scale_rewards="group"):
    """Compute group-relative advantages.

    Args:
        rewards: [B] per-sequence rewards.
        num_generations: G — number of generations per prompt.
        scale_rewards: "group", "batch", or "none".

    Returns:
        [B] advantages (centered and optionally scaled).
    """
    B = rewards.shape[0]
    num_prompts = B // num_generations

    # Group-relative mean: mean reward within each prompt's generation group
    grouped = rewards.reshape(num_prompts, num_generations)
    group_mean = grouped.mean(axis=1)  # [num_prompts]
    # Expand back to [B]
    group_mean_expanded = mx.repeat(group_mean, num_generations)

    advantages = rewards - group_mean_expanded

    if scale_rewards == "group":
        group_std = grouped.std(axis=1)  # [num_prompts]
        group_std_expanded = mx.repeat(group_std, num_generations)
        advantages = advantages / (group_std_expanded + 1e-4)
    elif scale_rewards == "batch":
        batch_std = rewards.std()
        advantages = advantages / (batch_std + 1e-4)
    elif scale_rewards == "none":
        pass
    else:
        raise ValueError(f"Unknown scale_rewards: {scale_rewards}")

    return advantages


# ---------------------------------------------------------------------------
# MLXGRPOTrainer
# ---------------------------------------------------------------------------

class MLXGRPOTrainer:
    """GRPO trainer for MLX.

    Implements the Generate → Score → Train loop for GRPO/DAPO/DR-GRPO/BNPO.

    Args:
        model: MLX model with LoRA applied.
        reward_funcs: Callable or list of callables.
            Each: (prompts: list[str], completions: list[str], **kwargs) → list[float]
        train_dataset: HuggingFace Dataset with "prompt" column.
        tokenizer: Tokenizer for encoding/decoding.
        args: MLXGRPOConfig instance.
    """

    def __init__(
        self,
        model,
        reward_funcs,
        train_dataset,
        tokenizer=None,
        processor=None,
        args=None,
    ):
        self.model = model
        self.tokenizer = tokenizer or processor
        self.train_dataset = train_dataset
        self.args = args or MLXGRPOConfig()

        # Normalize reward_funcs to list
        if callable(reward_funcs) and not isinstance(reward_funcs, list):
            self.reward_funcs = [reward_funcs]
        else:
            self.reward_funcs = list(reward_funcs)

        # Reward weights
        if self.args.reward_weights is not None:
            self.reward_weights = mx.array(self.args.reward_weights)
        else:
            self.reward_weights = mx.ones(len(self.reward_funcs))

        self.epsilon_high = self.args.epsilon_high or self.args.epsilon

        # Metrics history
        self._train_loss_history = []
        self._reward_history = []

    def _get_prompts(self, batch_indices):
        """Extract prompt texts from dataset."""
        prompts = []
        for idx in batch_indices:
            item = self.train_dataset[idx]
            if "prompt" in item:
                prompt = item["prompt"]
            elif "text" in item:
                prompt = item["text"]
            else:
                raise ValueError("Dataset must have 'prompt' or 'text' column")
            if isinstance(prompt, list):
                # Chat format: apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            prompts.append(prompt)
        return prompts

    def _generate(self, prompts):
        """Generate completions for prompts using mlx-lm.

        Returns:
            completions: list of str
            completion_ids: list of list[int]
        """
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temp=self.args.temperature,
            top_p=self.args.top_p,
        )

        completions = []
        completion_ids_list = []

        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt)
            if len(prompt_ids) > self.args.max_prompt_length:
                prompt_ids = prompt_ids[-self.args.max_prompt_length:]

            prompt_arr = mx.array(prompt_ids)
            tokens = []

            for token, logprobs in generate_step(
                prompt_arr, self.model,
                max_tokens=self.args.max_completion_length,
                sampler=sampler,
            ):
                token_id = token.item() if hasattr(token, 'item') else int(token)
                tokens.append(token_id)
                if token_id == self.tokenizer.eos_token_id:
                    break

            completion = self.tokenizer.decode(tokens)
            completions.append(completion)
            completion_ids_list.append(tokens)

        return completions, completion_ids_list

    def _compute_rewards(self, prompts, completions):
        """Compute rewards using reward functions.

        Returns:
            [B] reward tensor.
        """
        all_rewards = []
        for func in self.reward_funcs:
            rewards = func(prompts=prompts, completions=completions)
            all_rewards.append(rewards)

        # Combine: [B, num_funcs] → weighted sum → [B]
        rewards_matrix = mx.array(all_rewards).T  # [B, num_funcs]
        combined = (rewards_matrix * self.reward_weights[None, :]).sum(axis=-1)
        return combined

    def _prepare_training_batch(self, prompts, completions, completion_ids_list):
        """Prepare padded tensors for the training step.

        Returns dict with:
            input_ids: [B, T_total] padded prompt + completion
            completion_mask: [B, T_completion] bool mask
            completion_start: int, where completion begins
        """
        # Encode prompts
        prompt_ids_list = [
            self.tokenizer.encode(p)[-self.args.max_prompt_length:]
            for p in prompts
        ]

        # Truncate completions
        completion_ids_list = [
            ids[:self.args.max_completion_length]
            for ids in completion_ids_list
        ]

        # Determine max lengths
        max_prompt_len = max(len(p) for p in prompt_ids_list)
        max_comp_len = max(len(c) for c in completion_ids_list)

        B = len(prompts)
        T_total = max_prompt_len + max_comp_len

        # Pad and create arrays
        input_ids = np.zeros((B, T_total), dtype=np.int32)
        completion_mask = np.zeros((B, max_comp_len), dtype=bool)

        for i in range(B):
            p_ids = prompt_ids_list[i]
            c_ids = completion_ids_list[i]

            # Left-pad prompt to max_prompt_len
            pad_len = max_prompt_len - len(p_ids)
            input_ids[i, pad_len:pad_len + len(p_ids)] = p_ids
            input_ids[i, max_prompt_len:max_prompt_len + len(c_ids)] = c_ids

            # Completion mask
            completion_mask[i, :len(c_ids)] = True

            # Mask truncated completions if requested
            if self.args.mask_truncated_completions:
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and (len(c_ids) == 0 or c_ids[-1] != eos_id):
                    completion_mask[i, :] = False

        return {
            "input_ids": mx.array(input_ids),
            "completion_mask": mx.array(completion_mask),
            "completion_start": max_prompt_len,
        }

    def train(self):
        """Run GRPO training loop.

        Returns:
            dict with training metrics.
        """
        args = self.args
        model = self.model

        # Memory management (same as SFT trainer)
        if mx.metal.is_available():
            recommended = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(recommended)
            mx.set_memory_limit(recommended)
            mx.eval(model.parameters())
            active_after_load = mx.get_active_memory()
            mx.set_cache_limit(active_after_load * 2)

        # Optimizer
        total_steps = args.max_steps
        schedule = optim.cosine_decay(args.learning_rate, total_steps)
        optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

        state = [model.state, optimizer.state, mx.random.state]

        # Dataset indices
        num_prompts = len(self.train_dataset)
        rng = np.random.default_rng(args.seed)

        print(f"Unsloth GRPO: Training for {total_steps} steps")
        print(f"  num_generations={args.num_generations}, "
              f"loss_type={args.loss_type}, beta={args.beta}")
        print(f"  max_completion_length={args.max_completion_length}, "
              f"temperature={args.temperature}")

        start_time = time.perf_counter()
        mx.clear_cache()
        mx.reset_peak_memory()

        for step in range(1, total_steps + 1):
            tic = time.perf_counter()

            # Sample prompts
            batch_size = args.per_device_train_batch_size
            num_prompt_samples = batch_size // args.num_generations
            prompt_indices = rng.choice(num_prompts, size=num_prompt_samples, replace=True)
            # Repeat each prompt num_generations times
            all_indices = np.repeat(prompt_indices, args.num_generations)

            prompts = self._get_prompts(all_indices)

            # 1. GENERATE
            completions, completion_ids_list = self._generate(prompts)
            mx.clear_cache()

            # 2. SCORE
            rewards = self._compute_rewards(prompts, completions)
            advantages = compute_advantages(
                rewards, args.num_generations, args.scale_rewards
            )
            mx.eval(rewards, advantages)

            # 3. PREPARE BATCH
            batch = self._prepare_training_batch(prompts, completions, completion_ids_list)
            input_ids = batch["input_ids"]
            completion_mask = batch["completion_mask"]
            completion_start = batch["completion_start"]

            # 4. COMPUTE OLD LOGPROBS
            old_logps = compute_per_token_logps(
                model, input_ids, completion_start, completion_mask
            )
            old_logps = mx.stop_gradient(old_logps)

            # 5. COMPUTE REF LOGPROBS (if beta > 0)
            if args.beta != 0.0:
                with disable_lora(model):
                    ref_logps = compute_per_token_logps(
                        model, input_ids, completion_start, completion_mask
                    )
                ref_logps = mx.stop_gradient(ref_logps)
            else:
                ref_logps = None

            mx.eval(old_logps)
            if ref_logps is not None:
                mx.eval(ref_logps)

            # 6. TRAIN (num_iterations on same data)
            loss_and_grad_fn = nn.value_and_grad(model, lambda model, *a: grpo_loss(
                compute_per_token_logps(model, input_ids, completion_start, completion_mask),
                old_logps, ref_logps, advantages, completion_mask,
                beta=args.beta,
                epsilon_low=args.epsilon,
                epsilon_high=self.epsilon_high,
                loss_type=args.loss_type,
                importance_sampling_level=args.importance_sampling_level,
                max_completion_length=args.max_completion_length,
                num_items_in_batch=len(prompts),
            ))

            for mu in range(args.num_iterations):
                (loss, ntoks, metrics), grad = loss_and_grad_fn(model)
                optimizer.update(model, grad)
                mx.eval(state, loss, ntoks)

            train_time = time.perf_counter() - tic
            mean_reward = rewards.mean().item()
            self._train_loss_history.append(loss.item())
            self._reward_history.append(mean_reward)

            if step % args.logging_steps == 0 or step == total_steps:
                peak = mx.get_peak_memory() / 1e9
                print(
                    f"  Step {step}/{total_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Reward: {mean_reward:.4f} | "
                    f"KL: {metrics['kl'].item():.4f} | "
                    f"Clip: {metrics['clip_ratio_low'].item():.2%}/{metrics['clip_ratio_high'].item():.2%} | "
                    f"CompLen: {metrics['completion_length'].item():.0f} | "
                    f"Peak: {peak:.1f}G | "
                    f"Time: {train_time:.1f}s"
                )

            # Clear cache after first step (warmup pollution)
            if step == 1:
                mx.clear_cache()

        total_time = time.perf_counter() - start_time
        print(f"\nUnsloth GRPO: Complete! {total_steps} steps in {total_time:.1f}s")

        return {
            "train_loss": self._train_loss_history[-1] if self._train_loss_history else 0,
            "train_runtime": total_time,
            "rewards": self._reward_history,
        }
