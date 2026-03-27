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
MLX utilities for Apple Silicon training.

Provides loss functions (CCE via mlx-cce, baseline CE), data batching,
weight extraction helpers, and model save/load for LoRA adapters.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import json
import os
from pathlib import Path

_RUNTIME_CCE_IMPORT_ERROR = None
_RUNTIME_CCE_MAKE_LOSS = None
_RUNTIME_CCE_CACHE = {}


def _load_runtime_cce():
    global _RUNTIME_CCE_IMPORT_ERROR
    global _RUNTIME_CCE_MAKE_LOSS
    if _RUNTIME_CCE_MAKE_LOSS is not None:
        return _RUNTIME_CCE_MAKE_LOSS
    if _RUNTIME_CCE_IMPORT_ERROR is not None:
        return None
    try:
        from mlx_cce_runtime import make_chunked_cross_entropy_loss
    except ImportError as exc:
        _RUNTIME_CCE_IMPORT_ERROR = exc
        return None
    _RUNTIME_CCE_MAKE_LOSS = make_chunked_cross_entropy_loss
    return _RUNTIME_CCE_MAKE_LOSS


def _get_runtime_cce(
    *,
    ignore_index: int,
    logit_softcap: float,
    quantized: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
):
    make_loss = _load_runtime_cce()
    if make_loss is None:
        return None
    key = (ignore_index, logit_softcap, quantized, group_size, bits, mode)
    runtime_cce = _RUNTIME_CCE_CACHE.get(key)
    if runtime_cce is None:
        runtime_cce, _ = make_loss(
            ignore_index=ignore_index,
            logit_softcap=logit_softcap,
            quantized=quantized,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        _RUNTIME_CCE_CACHE[key] = runtime_cce
    return runtime_cce


def apply_gradient_checkpointing(model):
    """Apply gradient checkpointing to all transformer layers.

    Patches the layer class's __call__ to use mx.checkpoint, which
    recomputes activations during backward instead of storing them.
    Trades ~30% more compute for significant memory savings.

    Follows the same pattern as mlx_lm.tuner.trainer.grad_checkpoint.
    """
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        return
    layer_cls = type(layers[0])
    if getattr(layer_cls, "_orig_call", None) is not None:
        return  # already applied
    layer_cls._orig_call = layer_cls.__call__
    fn = layer_cls.__call__

    def checkpointed_fn(self, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            self.update(params)
            return fn(self, *args, **kwargs)

        return mx.checkpoint(inner_fn)(self.trainable_parameters(), *args, **kwargs)

    layer_cls.__call__ = checkpointed_fn


def remove_gradient_checkpointing(model):
    """Remove gradient checkpointing, restoring original layer __call__."""
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        return
    layer_cls = type(layers[0])
    orig = getattr(layer_cls, "_orig_call", None)
    if orig is not None:
        layer_cls.__call__ = orig
        del layer_cls._orig_call


def _get_lm_head_layer(model):
    """Get the raw LM head layer (QuantizedLinear or Linear/Embedding).

    Checks for a separate lm_head first (untied models like Qwen), then
    falls back to embed_tokens (tied models like Gemma/Llama).

    Returns the layer object (not its weight), so callers can access
    .weight, .scales, .biases, .group_size, .bits for quantized layers.
    """
    if hasattr(model, "lm_head") and model.lm_head is not None:
        return model.lm_head
    return model.model.embed_tokens


def _is_quantized_layer(layer):
    """Check if a layer has quantized weights (has .scales attribute)."""
    return hasattr(layer, "scales")


def has_cce_kernel():
    """Check if either direct runtime CCE or mx.fast.cce_loss is available."""
    return _load_runtime_cce() is not None or hasattr(mx.fast, "cce_loss")


def _get_logit_softcap(model):
    """Get logit softcapping value if model uses it (e.g. Gemma-2), else 0.0."""
    softcap = getattr(model, "final_logit_softcapping", None)
    if softcap is None and hasattr(model, "args"):
        softcap = getattr(model.args, "final_logit_softcapping", None)
    return float(softcap) if softcap is not None and softcap > 0 else 0.0


def _is_lm_head_trainable(model):
    """Check if the LM head weight is trainable (not frozen by LoRA).

    For LoRA training, the LM head weight is frozen — computing its gradient
    in CCE is a wasted V x chunk_size x H matmul per chunk. Returns False
    when the weight should be wrapped with mx.stop_gradient.
    """
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
    for key in trainable:
        if "lora" not in key:
            if "lm_head" in key or "embed_tokens.weight" in key:
                return True
    return len(trainable) == 0  # no LoRA = full fine-tuning = trainable


def make_cce_loss_fn(model):
    """Create a CCE loss function using runtime CCE or mx.fast.cce_loss.

    CCE computes cross-entropy directly from hidden states and the LM head weight,
    avoiding full logit materialization. This saves significant memory for large
    vocabularies. For V=256K+ models, CCE is both faster and more memory-efficient.

    If the LM head is quantized, passes raw uint32 weight + scales + biases
    directly to cce_loss, using fused quantized matmul kernels.

    Returns:
        A function (model, batch, lengths) -> (loss, ntoks).
    """
    if not has_cce_kernel():
        raise RuntimeError(
            "mx.fast.cce_loss not available. Install mlx-cce: pip install mlx-cce"
        )

    softcap = _get_logit_softcap(model)
    if softcap > 0:
        print(f"Unsloth: CCE using logit_softcap={softcap} for this model.")

    lm_layer = _get_lm_head_layer(model)
    use_quantized = _is_quantized_layer(lm_layer)

    if use_quantized:
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)
        print(
            f"Unsloth: CCE using quantized matmul (group_size={group_size}, bits={bits})"
        )
        runtime_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
            quantized=True,
            group_size=group_size,
            bits=bits,
        )
        _has_lm_head_q = (
            hasattr(model, "lm_head")
            and model.lm_head is not None
            and hasattr(model.lm_head, "scales")
        )
        _has_biases = hasattr(lm_layer, "biases")

        def loss_fn(model, batch, lengths):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(inputs)
            layer = model.lm_head if _has_lm_head_q else model.model.embed_tokens
            w = layer.weight
            sc = layer.scales
            bi = layer.biases if _has_biases else None
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            if runtime_cce is not None:
                hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
                targets_flat = masked_targets.reshape((-1,))
                loss = runtime_cce(hidden_flat, w, sc, bi, targets_flat).reshape(
                    masked_targets.shape
                )
            else:
                loss = mx.fast.cce_loss(
                    hidden,
                    w,
                    masked_targets,
                    scales=sc,
                    biases=bi,
                    group_size=group_size,
                    bits=bits,
                    ignore_index=-100,
                    logit_softcap=softcap,
                )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks
    else:
        # For full-precision models, access weight through the model parameter
        # tree so nn.value_and_grad can trace gradients through it.
        # Closure-capturing the weight would make autograd treat it as a
        # constant, producing zero gradient for the LM head — causing NaN
        # divergence during full fine-tuning.
        runtime_cce = _get_runtime_cce(
            ignore_index=-100,
            logit_softcap=softcap,
        )
        _has_lm_head = (
            hasattr(model, "lm_head")
            and model.lm_head is not None
            and hasattr(model.lm_head, "weight")
        )
        _skip_weight_grad = not _is_lm_head_trainable(model)
        if _skip_weight_grad:
            print("Unsloth: CCE skipping weight gradient (LM head is frozen).")

        def loss_fn(model, batch, lengths):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(inputs)
            w = (
                model.lm_head.weight
                if _has_lm_head
                else model.model.embed_tokens.weight
            )
            if _skip_weight_grad:
                w = mx.stop_gradient(w)
            steps = mx.arange(1, targets.shape[1] + 1)
            mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
            masked_targets = mx.where(mask, targets, -100)
            ntoks = mask.sum()
            if runtime_cce is not None:
                hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
                targets_flat = masked_targets.reshape((-1,))
                loss = runtime_cce(hidden_flat, w, targets_flat).reshape(
                    masked_targets.shape
                )
            else:
                loss = mx.fast.cce_loss(
                    hidden,
                    w,
                    masked_targets,
                    ignore_index=-100,
                    logit_softcap=softcap,
                )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks

    loss_fn._unsloth_cce_backend = (
        "direct-runtime" if runtime_cce is not None else "mx.fast"
    )
    return loss_fn


def make_baseline_loss_fn():
    """Create a standard cross-entropy loss function.

    Uses the full logit computation through the LM head, then applies
    nn.losses.cross_entropy. This is the fallback when CCE is not available.

    Returns:
        A function (model, batch, lengths) -> (loss, ntoks).
    """
    upcast_logits = os.environ.get("UNSLOTH_MLX_UPCAST_LOGITS", "0") == "1"

    def loss_fn(model, batch, lengths):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        if upcast_logits:
            logits = logits.astype(mx.float32)
        # Mask padding tokens using lengths from iterate_batches
        steps = mx.arange(1, targets.shape[1] + 1)
        mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        loss = ce.astype(mx.float32).sum() / ntoks
        return loss, ntoks

    return loss_fn


def make_packed_cce_loss_fn(model):
    """Create a CCE loss function for packed sequence training.

    Like make_cce_loss_fn but accepts (model, batch, loss_mask, attn_mask, position_ids)
    and passes packed attention mask and position IDs through the model.
    """
    if not has_cce_kernel():
        raise RuntimeError("mx.fast.cce_loss not available for packed CCE.")

    softcap = _get_logit_softcap(model)
    lm_layer = _get_lm_head_layer(model)
    use_quantized = _is_quantized_layer(lm_layer)

    if use_quantized:
        group_size = getattr(lm_layer, "group_size", 64)
        bits = getattr(lm_layer, "bits", 4)
        runtime_cce = _get_runtime_cce(
            ignore_index=-100, logit_softcap=softcap,
            quantized=True, group_size=group_size, bits=bits,
        )
        _has_lm_head_q = (
            hasattr(model, "lm_head") and model.lm_head is not None
            and hasattr(model.lm_head, "scales")
        )
        _has_biases = hasattr(lm_layer, "biases")

        def loss_fn(model, batch, loss_mask, attn_mask, position_ids):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(
                inputs, packed_attn_mask=attn_mask, packed_position_ids=position_ids
            )
            layer = model.lm_head if _has_lm_head_q else model.model.embed_tokens
            w, sc = layer.weight, layer.scales
            bi = layer.biases if _has_biases else None
            masked_targets = mx.where(loss_mask, targets, -100)
            ntoks = loss_mask.sum()
            if runtime_cce is not None:
                hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
                targets_flat = masked_targets.reshape((-1,))
                loss = runtime_cce(hidden_flat, w, sc, bi, targets_flat).reshape(
                    masked_targets.shape
                )
            else:
                loss = mx.fast.cce_loss(
                    hidden, w, masked_targets, scales=sc, biases=bi,
                    group_size=group_size, bits=bits,
                    ignore_index=-100, logit_softcap=softcap,
                )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks
    else:
        runtime_cce = _get_runtime_cce(ignore_index=-100, logit_softcap=softcap)
        _has_lm_head = (
            hasattr(model, "lm_head") and model.lm_head is not None
            and hasattr(model.lm_head, "weight")
        )
        _skip_weight_grad = not _is_lm_head_trainable(model)

        def loss_fn(model, batch, loss_mask, attn_mask, position_ids):
            inputs, targets = batch[:, :-1], batch[:, 1:]
            hidden = model.model(
                inputs, packed_attn_mask=attn_mask, packed_position_ids=position_ids
            )
            w = (model.lm_head.weight if _has_lm_head
                 else model.model.embed_tokens.weight)
            if _skip_weight_grad:
                w = mx.stop_gradient(w)
            masked_targets = mx.where(loss_mask, targets, -100)
            ntoks = loss_mask.sum()
            if runtime_cce is not None:
                hidden_flat = hidden.reshape((-1, hidden.shape[-1]))
                targets_flat = masked_targets.reshape((-1,))
                loss = runtime_cce(hidden_flat, w, targets_flat).reshape(
                    masked_targets.shape
                )
            else:
                loss = mx.fast.cce_loss(
                    hidden, w, masked_targets,
                    ignore_index=-100, logit_softcap=softcap,
                )
            loss = loss.astype(mx.float32).sum() / ntoks
            return loss, ntoks

    return loss_fn


def make_packed_baseline_loss_fn():
    """Create a standard cross-entropy loss for packed sequence training.

    Like make_baseline_loss_fn but accepts (model, batch, loss_mask, attn_mask, position_ids).
    """
    upcast_logits = os.environ.get("UNSLOTH_MLX_UPCAST_LOGITS", "0") == "1"

    def loss_fn(model, batch, loss_mask, attn_mask, position_ids):
        inputs, targets = batch[:, :-1], batch[:, 1:]
        # For baseline loss, we need the full model forward (including LM head).
        # The model's __call__ typically does: hidden = model.model(inputs); logits = lm_head(hidden)
        # We need to intercept to pass packed args to model.model.
        hidden = model.model(
            inputs, packed_attn_mask=attn_mask, packed_position_ids=position_ids
        )
        # Apply LM head manually
        if hasattr(model, "lm_head") and model.lm_head is not None:
            logits = model.lm_head(hidden)
        else:
            logits = model.model.embed_tokens.as_linear(hidden)
        if upcast_logits:
            logits = logits.astype(mx.float32)
        masked_targets = mx.where(loss_mask, targets, -100)
        ce = nn.losses.cross_entropy(logits, masked_targets) * loss_mask
        ntoks = loss_mask.sum()
        loss = ce.astype(mx.float32).sum() / ntoks
        return loss, ntoks

    return loss_fn


def _prepare_dataset(
    dataset, tokenizer, dataset_text_field="text", formatting_func=None
):
    """Wrap a HuggingFace dataset into mlx-lm's dataset classes.

    Uses TextDataset + CacheDataset from mlx_lm so that tokenization
    (including EOS appending) matches mlx-lm's own training pipeline exactly.

    If a formatting_func is provided, each item is pre-formatted into a
    ``{"text": ...}`` dict before wrapping.

    Returns:
        A CacheDataset ready for ``iterate_batches``.
    """
    from mlx_lm.tuner.datasets import TextDataset, CacheDataset

    # Pre-format items into [{"text": str}, ...] so TextDataset can consume them.
    formatted = []
    for item in dataset:
        if formatting_func is not None:
            result = formatting_func(item)
            texts = result if isinstance(result, list) else [result]
        elif isinstance(item, dict):
            texts = []
            if dataset_text_field in item:
                texts = [item[dataset_text_field]]
            else:
                for key in ("text", "content", "instruction"):
                    if key in item:
                        texts = [item[key]]
                        break
        elif isinstance(item, str):
            texts = [item]
        else:
            continue

        for text in texts:
            if text:
                formatted.append({"text": text})

    if not formatted:
        raise ValueError(
            f"No text data found. Provide a dataset with a "
            f"'{dataset_text_field}' column."
        )

    return CacheDataset(TextDataset(formatted, tokenizer, text_key="text"))


def create_batches(
    dataset,
    tokenizer,
    batch_size,
    max_seq_length,
    num_batches=None,
    seed=42,
    dataset_text_field="text",
    formatting_func=None,
):
    """Pre-tokenize and batch a HuggingFace dataset for MLX training.

    Uses iterate_batches from mlx_lm for efficient dynamic-padding batching:
    samples are sorted by length, grouped into batches, and padded to the
    max length within each batch (rounded up to the nearest multiple of 32),
    capped at max_seq_length.

    Tokenization is delegated to mlx_lm's TextDataset (appends EOS, etc.)
    so behaviour matches ``mlx_lm.lora`` exactly.

    Returns:
        List of (batch, lengths) tuples, where batch has shape
        (batch_size, padded_length) and lengths has shape (batch_size, 2)
        with [offset, length] per sequence (from iterate_batches).
    """
    from mlx_lm.tuner.trainer import iterate_batches

    ds = _prepare_dataset(dataset, tokenizer, dataset_text_field, formatting_func)

    batch_pairs = []
    for batch, lengths_info in iterate_batches(
        ds,
        batch_size,
        max_seq_length,
        loop=(num_batches is not None),
        seed=seed,
    ):
        batch_pairs.append((batch, lengths_info))
        if num_batches is not None and len(batch_pairs) >= num_batches:
            break

    mx.eval([b for b, _ in batch_pairs] + [l for _, l in batch_pairs])
    return batch_pairs


def iterate_training_batches(
    dataset,
    tokenizer,
    batch_size,
    max_seq_length,
    seed=42,
    dataset_text_field="text",
    formatting_func=None,
):
    """Streaming batch generator for MLX training.

    Wraps mlx-lm's iterate_batches(loop=True) as a generator, avoiding
    materializing all batches in memory at once. Useful for large datasets.

    Yields:
        (batch, lengths) tuples — same format as create_batches.
    """
    from mlx_lm.tuner.trainer import iterate_batches

    ds = _prepare_dataset(dataset, tokenizer, dataset_text_field, formatting_func)

    for batch, lengths_info in iterate_batches(
        ds,
        batch_size,
        max_seq_length,
        loop=True,
        seed=seed,
    ):
        yield batch, lengths_info


def save_lora_adapters(model, path, adapter_config=None):
    """Save LoRA adapter weights to disk.

    Args:
        model: MLX model with LoRA layers.
        path: Directory to save adapters.
        adapter_config: Optional dict with LoRA config metadata.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Collect only trainable (LoRA) parameters — flatten nested dict for safetensors
    trainable = dict(mlx.utils.tree_flatten(model.trainable_parameters()))

    if trainable:
        mx.save_safetensors(str(path / "adapters.safetensors"), trainable)

    if adapter_config:
        with open(path / "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)


def save_merged_model(model, tokenizer, path):
    """Fuse LoRA weights and save the full merged model.

    Args:
        model: MLX model with LoRA layers.
        tokenizer: Tokenizer to save alongside.
        path: Directory to save merged model.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Fuse LoRA weights
    model.eval()
    de_lora_model = nn.utils.fuse_lora(model)

    # Save all weights — flatten nested dict for safetensors
    weights = dict(mlx.utils.tree_flatten(de_lora_model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)

    # Save tokenizer
    tokenizer.save_pretrained(str(path))


# ---------------------------------------------------------------------------
# Sequence packing for padding-free training
# ---------------------------------------------------------------------------

import numpy as np


def pack_sequences(token_sequences, max_len, pad_token_id=0):
    """Pack variable-length sequences into fixed-length rows using first-fit-decreasing.

    Args:
        token_sequences: List of (token_ids, offset) tuples from the dataset.
            offset is the prompt length — tokens before offset are prompt (masked from loss),
            tokens from offset onward are response (trained on).
        max_len: Maximum packed row length (e.g. 2048).
        pad_token_id: Token ID used for padding (default 0).

    Returns:
        List of dicts, each containing:
          - "input_ids": np.array [max_len] of packed token IDs
          - "seq_lengths": list of individual sequence lengths in this row
          - "seq_offsets": list of per-sequence prompt offsets (for response-only training)
          - "num_sequences": number of sequences packed in this row
    """
    # Extract token lists with offsets, sort by length descending (first-fit-decreasing)
    seqs = []
    for item in token_sequences:
        tokens, offset = item
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            offset = min(offset, max_len)
        if len(tokens) > 0:
            seqs.append((tokens, offset))
    seqs.sort(key=lambda x: len(x[0]), reverse=True)

    # First-fit-decreasing bin packing
    bins = []
    for tokens, offset in seqs:
        seq_len = len(tokens)
        placed = False
        for b in bins:
            if b["remaining"] >= seq_len:
                b["tokens"].extend(tokens)
                b["seq_lengths"].append(seq_len)
                b["seq_offsets"].append(offset)
                b["remaining"] -= seq_len
                placed = True
                break
        if not placed:
            bins.append({
                "tokens": list(tokens),
                "seq_lengths": [seq_len],
                "seq_offsets": [offset],
                "remaining": max_len - seq_len,
            })

    # Convert to fixed-length arrays
    packed_rows = []
    for b in bins:
        input_ids = np.zeros(max_len, dtype=np.int32)
        input_ids[:len(b["tokens"])] = b["tokens"]
        packed_rows.append({
            "input_ids": input_ids,
            "seq_lengths": b["seq_lengths"],
            "seq_offsets": b["seq_offsets"],
            "num_sequences": len(b["seq_lengths"]),
        })

    return packed_rows


def build_block_diagonal_mask(seq_lengths, max_len):
    """Build a block-diagonal causal attention mask for packed sequences.

    Args:
        seq_lengths: List of sequence lengths in this packed row (e.g. [500, 700, 848]).
        max_len: Total row length.

    Returns:
        np.array of shape [max_len, max_len], dtype bool.
        True where attention is allowed (causal within each block, zero across blocks).
    """
    mask = np.zeros((max_len, max_len), dtype=bool)
    offset = 0
    for length in seq_lengths:
        # Causal mask within this block: lower triangular
        block = np.tril(np.ones((length, length), dtype=bool))
        mask[offset:offset + length, offset:offset + length] = block
        offset += length
    return mask


def compute_position_ids(seq_lengths, max_len):
    """Compute per-token position IDs that reset at each sequence boundary.

    Args:
        seq_lengths: List of sequence lengths in this packed row.
        max_len: Total row length.

    Returns:
        np.array of shape [max_len], dtype int32.
        E.g. for seq_lengths=[3, 4]: [0, 1, 2, 0, 1, 2, 3, 0, 0, ...]
    """
    positions = np.zeros(max_len, dtype=np.int32)
    offset = 0
    for length in seq_lengths:
        positions[offset:offset + length] = np.arange(length, dtype=np.int32)
        offset += length
    return positions


def compute_loss_mask(seq_lengths, max_len, seq_offsets=None):
    """Compute loss mask for packed sequences (applied to targets, which are shifted by 1).

    Handles both full-sequence training (offset=0) and response-only training
    (offset>0, where tokens before offset are prompt and masked from loss).

    For inter-sequence boundaries within a packed row, the last token of each
    sub-sequence (except the final one) is masked because it predicts into
    the next sequence's first token.

    Args:
        seq_lengths: List of sequence lengths in this packed row.
        max_len: Total row length.
        seq_offsets: Optional list of per-sequence prompt offsets. If None,
            defaults to 0 for all sequences (train on everything).
            offset=N means skip the first N tokens of that sequence from loss.

    Returns:
        np.array of shape [max_len - 1], dtype bool.
        True where loss should be computed.
    """
    if seq_offsets is None:
        seq_offsets = [0] * len(seq_lengths)

    # Build mask in target space (shifted by 1 from input space).
    # targets[t] = batch[t+1], so target position t corresponds to
    # predicting token at position t+1 given input at position t.
    #
    # For a sub-sequence at packed position [start, start+length):
    #   - Prompt tokens: positions start..start+offset-1 → mask these
    #   - Response tokens: positions start+offset..start+length-1 → train on these
    #   - In target space: target[t] is valid when BOTH input[t] and target[t]=batch[t+1]
    #     are within the response portion of the same sub-sequence.
    mask_targets = np.zeros(max_len - 1, dtype=bool)
    pos = 0
    for i, (length, prompt_offset) in enumerate(zip(seq_lengths, seq_offsets)):
        is_last = (i == len(seq_lengths) - 1)

        # Response starts at position pos + prompt_offset within the packed row.
        # In target space, the first valid target is at position pos + prompt_offset
        # (input[pos+prompt_offset] predicts target[pos+prompt_offset] = batch[pos+prompt_offset+1],
        # both within the response).
        resp_start = pos + prompt_offset

        # Last valid target position within this sub-sequence:
        # - For intermediate sub-sequences: pos + length - 2
        #   (because input[pos+length-1] predicts batch[pos+length] which is in the next sequence)
        # - For the last sub-sequence: pos + length - 1
        #   (predicts into padding, matching unpacked behavior)
        if is_last:
            resp_end = pos + length  # exclusive, in target space
        else:
            resp_end = pos + length - 1  # exclusive, skip boundary

        # Clamp to valid target range
        t_start = max(resp_start, 0)
        t_end = min(resp_end, max_len - 1)

        if t_end > t_start:
            mask_targets[t_start:t_end] = True

        pos += length

    return mask_targets


def create_packed_batches(
    dataset,
    tokenizer,
    batch_size,
    max_seq_length,
    num_batches=None,
    seed=42,
    dataset_text_field="text",
    formatting_func=None,
):
    """Create packed batches with pre-computed attention masks and position IDs.

    Like create_batches() but packs multiple sequences per row to eliminate
    padding waste. Returns a different tuple format with mask/position metadata.

    Returns:
        List of (batch, loss_mask, attn_mask, position_ids) tuples where:
          - batch: mx.array [B, max_seq_length]
          - loss_mask: mx.array [B, max_seq_length - 1] bool
          - attn_mask: mx.array [max_seq_length - 1, max_seq_length - 1] bool
            (shared across batch — all rows use same mask structure if possible,
             otherwise [B, 1, max_seq_length-1, max_seq_length-1])
          - position_ids: mx.array [B, max_seq_length - 1]
    """
    ds = _prepare_dataset(dataset, tokenizer, dataset_text_field, formatting_func)

    # Collect all tokenized sequences
    all_seqs = []
    for i in range(len(ds)):
        item = ds[i]
        all_seqs.append(item)

    # Pack sequences into rows
    packed_rows = pack_sequences(all_seqs, max_seq_length, pad_token_id=0)

    # Shuffle packed rows
    rng = np.random.default_rng(seed)
    rng.shuffle(packed_rows)

    # Group into batches
    batch_pairs = []
    for b_start in range(0, len(packed_rows), batch_size):
        b_end = min(b_start + batch_size, len(packed_rows))
        rows = packed_rows[b_start:b_end]

        if len(rows) < batch_size:
            # Skip incomplete last batch
            continue

        # Stack input_ids
        batch_arr = np.stack([r["input_ids"] for r in rows])  # [B, T]

        # Compute per-row position_ids and loss_mask
        T = max_seq_length
        pos_ids = np.stack([
            compute_position_ids(r["seq_lengths"], T) for r in rows
        ])  # [B, T]
        loss_masks = np.stack([
            compute_loss_mask(r["seq_lengths"], T, r.get("seq_offsets"))
            for r in rows
        ])  # [B, T-1]

        # Compute attention masks — one per row since packing varies
        attn_masks = np.stack([
            build_block_diagonal_mask(r["seq_lengths"], T) for r in rows
        ])  # [B, T, T]

        # Slice to input length (T-1, since inputs = batch[:, :-1])
        T_in = T - 1
        pos_ids_in = pos_ids[:, :T_in]
        attn_masks_in = attn_masks[:, :T_in, :T_in]  # [B, T_in, T_in]
        # Add head dimension for broadcast: [B, 1, T_in, T_in]
        attn_masks_in = attn_masks_in[:, np.newaxis, :, :]

        batch_pairs.append((
            mx.array(batch_arr),
            mx.array(loss_masks),
            mx.array(attn_masks_in),
            mx.array(pos_ids_in),
        ))

        if num_batches is not None and len(batch_pairs) >= num_batches:
            break

    # Pre-evaluate all arrays
    all_arrays = []
    for b, lm, am, pi in batch_pairs:
        all_arrays.extend([b, lm, am, pi])
    mx.eval(all_arrays)

    packing_ratio = len(all_seqs) / max(1, sum(len(packed_rows) for _ in [1]))
    total_tokens = sum(sum(r["seq_lengths"]) for r in packed_rows)
    total_capacity = len(packed_rows) * max_seq_length
    utilization = total_tokens / max(1, total_capacity)
    print(f"Unsloth: Packed {len(all_seqs)} sequences into {len(packed_rows)} rows "
          f"({utilization:.0%} utilization, {len(batch_pairs)} batches)")

    return batch_pairs


# ---------------------------------------------------------------------------
# Packing monkey-patches for model forward pass
# ---------------------------------------------------------------------------

def _apply_rope_per_token(rope_module, x, position_ids):
    """Apply RoPE with per-token position IDs using the reshape trick.

    Args:
        rope_module: The model's RoPE module (nn.RoPE, Llama3RoPE, etc.)
        x: [B, N_heads, T, D] tensor
        position_ids: [B, T] per-token position IDs

    Returns:
        [B, N_heads, T, D] tensor with per-token RoPE applied.
    """
    B, N, T, D = x.shape
    # Reshape to [B*T, N, 1, D] — each token becomes its own "batch"
    x_flat = x.transpose(0, 2, 1, 3).reshape(B * T, N, 1, D)
    offsets = position_ids.reshape(B * T)
    # Apply RoPE with per-"batch" offsets
    x_roped = rope_module(x_flat, offset=offsets)
    # Reshape back to [B, N, T, D]
    return x_roped.reshape(B, T, N, D).transpose(0, 2, 1, 3)


def apply_packing_patches(model):
    """Monkey-patch the model to support packed sequence training.

    Patches:
      1. Attention.__call__ — uses per-token RoPE when _packed_position_ids is set
      2. The inner model's __call__ — passes packed mask/positions through layers

    Follows the same pattern as apply_gradient_checkpointing().
    """
    inner_model = model.model if hasattr(model, "model") else model
    layers = getattr(inner_model, "layers", None)
    if not layers or len(layers) == 0:
        print("Unsloth: Warning — no layers found for packing patches")
        return

    # Patch 1: Attention class — per-token RoPE
    attn_module = layers[0].self_attn
    attn_cls = type(attn_module)

    if getattr(attn_cls, "_orig_call_packing", None) is not None:
        return  # already patched

    attn_cls._orig_call_packing = attn_cls.__call__
    orig_attn_fn = attn_cls.__call__

    def patched_attn_call(self, x, mask=None, cache=None):
        position_ids = getattr(self, "_packed_position_ids", None)
        if position_ids is not None and cache is None:
            B, L, D = x.shape
            queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
            keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

            queries = _apply_rope_per_token(self.rope, queries, position_ids)
            keys = _apply_rope_per_token(self.rope, keys, position_ids)

            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)
        else:
            return orig_attn_fn(self, x, mask, cache)

    attn_cls.__call__ = patched_attn_call

    # Patch 2: Inner model class — thread packed mask and position_ids
    inner_cls = type(inner_model)

    if getattr(inner_cls, "_orig_call_packing", None) is not None:
        return

    inner_cls._orig_call_packing = inner_cls.__call__
    orig_model_fn = inner_cls.__call__

    def patched_model_call(self, inputs, cache=None, input_embeddings=None,
                           packed_attn_mask=None, packed_position_ids=None):
        if packed_attn_mask is not None and cache is None:
            # Set position_ids on each attention module
            for layer in self.layers:
                layer.self_attn._packed_position_ids = packed_position_ids

            h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
            if cache is None:
                cache_list = [None] * len(self.layers)
            else:
                cache_list = cache

            for layer, c in zip(self.layers, cache_list):
                h = layer(h, packed_attn_mask, cache=c)

            # Clean up
            for layer in self.layers:
                layer.self_attn._packed_position_ids = None

            return self.norm(h)
        else:
            return orig_model_fn(self, inputs, cache, input_embeddings)

    inner_cls.__call__ = patched_model_call
    print("Unsloth: Packing patches applied (per-token RoPE + block-diagonal attention)")


def remove_packing_patches(model):
    """Remove packing monkey-patches, restoring original model behavior."""
    inner_model = model.model if hasattr(model, "model") else model
    layers = getattr(inner_model, "layers", None)

    if layers and len(layers) > 0:
        attn_cls = type(layers[0].self_attn)
        orig = getattr(attn_cls, "_orig_call_packing", None)
        if orig is not None:
            attn_cls.__call__ = orig
            del attn_cls._orig_call_packing

        # Clean up any leftover attributes
        for layer in layers:
            if hasattr(layer.self_attn, "_packed_position_ids"):
                del layer.self_attn._packed_position_ids

    inner_cls = type(inner_model)
    orig = getattr(inner_cls, "_orig_call_packing", None)
    if orig is not None:
        inner_cls.__call__ = orig
        del inner_cls._orig_call_packing


