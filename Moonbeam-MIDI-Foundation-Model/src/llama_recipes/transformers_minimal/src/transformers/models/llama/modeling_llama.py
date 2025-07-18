# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model."""

import math
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_llama import LlamaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from music21 import chord
from music21 import harmony
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
def chord_to_midi(chord_symbol):
    # Create a ChordSymbol object
    chord_obj = harmony.ChordSymbol(chord_symbol)
    # Get the pitches of the chord
    pitches = chord_obj.pitches
    # Return the pitch names
    out = [p.midi  for p in pitches]
    return out

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (batch, seq_len)
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) #(dim/2, ) -> (1, dim/2, 1) ->(batch, dim/2, 1)
        position_ids_expanded = position_ids[:, None, :].float() #(batch, len) -> (batch, 1, len)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2) #(batch, dim/2, len) -> (batch, len, dim/2)
            emb = torch.cat((freqs, freqs), dim=-1)  #(batch, len, dim/2)-> (batch, len, dim)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor #scale different types differently (e.g., onset should be scaled smaller)
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaOutputAttention(nn.Module):
    """output attention with normal rope embedding"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

class CustomConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta_onset = config.rope_theta_onset
        self.rope_theta_dur = config.rope_theta_dur
        self.rope_theta_octave = config.rope_theta_octave
        self.rope_theta_pitch = config.rope_theta_pitch
        self.rope_theta_velocity = config.rope_theta_velocity
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb_onset = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_onset,
            )
            self.rotary_emb_dur = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_dur,
            )
            self.rotary_emb_octave = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_octave,
            )
            self.rotary_emb_pitch = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_pitch,
            )
            self.rotary_emb_velocity = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta_velocity,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaAttentionBaseline(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        additional_token_map: Optional[Dict] = None,
        additional_tokens_pos_map: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) #(bsz, head, len, dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        #different heads carry different information: first group of heads: onset; second group of heads: duration; third group of head: pitch so on and so forth
        #TODO: modify positions_id such that sos is assigned with pos 0 and eos is assigned with pos 2 ** config.onset_vocab
        
        where_sos = (position_ids[..., 0] == self.config.sos_token).unsqueeze(-1) #only check the onset dimension  
        where_eos = (position_ids[..., 0] == self.config.eos_token).unsqueeze(-1) 
        
        #Since SOS and EOS are negative number, temporarily change it to 0 to avoid indexing error
        position_ids_sos = torch.where(where_sos, torch.tensor([0 for _ in range(6)]).to(hidden_states.device), position_ids)
        position_ids_eos = torch.where(where_eos, torch.tensor([2**15 for _ in range(6)]).to(hidden_states.device), position_ids_sos)
        position_ids = position_ids_eos

        #Extend this to new tokens positions
        if additional_token_map is not None and additional_tokens_pos_map is not None:
            for token_id in additional_token_map:
                where_new_token = (position_ids[..., 0] == token_id).unsqueeze(-1)
                if str(token_id) in additional_tokens_pos_map:
                    position_ids = torch.where(where_new_token, torch.tensor(additional_tokens_pos_map[str(token_id)]).to(hidden_states.device), position_ids)
                else:
                    position_ids = torch.where(where_new_token, torch.tensor([0 for _ in range(6)]).to(hidden_states.device), position_ids) #TODO: COMMU, add pos for min max vel, pitch
        cos_onset, sin_onset = self.rotary_emb_onset(value_states, position_ids[:, :, 0]) #position_ids: batch, len, 6; last dim: (onset, duration, octave, pitch_class, instrument, velocity)
        cos_dur, sin_dur = self.rotary_emb_dur(value_states, position_ids[:, :, 1]) 
        cos_octave, sin_octave = self.rotary_emb_octave(value_states, position_ids[:, :, 2]) 
        cos_pitch, sin_pitch = self.rotary_emb_pitch(value_states, position_ids[:, :, 3]) 
        cos_velocity, sin_velocity = self.rotary_emb_velocity(value_states, position_ids[:, :, 5]) 

        query_states_split = query_states.view(bsz, 6, -1, q_len, self.head_dim) #(bsz, 6, head_q/6, len, dim) 
        key_states_split = key_states.view(bsz, 6, -1, q_len, self.head_dim)#(bsz, 6, head_kv/6, len, dim)
        
        query_states_onset, key_states_onset = apply_rotary_pos_emb(query_states_split[:,0,:,:,:], key_states_split[:,0,:,:,:], cos_onset, sin_onset)  #apply to head group 0
        query_states_dur, key_states_dur = apply_rotary_pos_emb(query_states_split[:,1,:,:,:], key_states_split[:,1,:,:,:], cos_dur, sin_dur) #apply to head group 1
        query_states_octave, key_states_octave = apply_rotary_pos_emb(query_states_split[:,2,:,:,:], key_states_split[:,2,:,:,:], cos_octave, sin_octave)
        query_states_pitch, key_states_pitch = apply_rotary_pos_emb(query_states_split[:,3,:,:,:], key_states_split[:,3,:,:,:], cos_pitch, sin_pitch) #apply to head group 3
        query_states_instr, key_states_instr = apply_rotary_pos_emb(query_states_split[:,4,:,:,:], key_states_split[:,4,:,:,:], cos_onset, sin_onset) #apply to head group 4
        query_states_velocity, key_states_velocity = apply_rotary_pos_emb(query_states_split[:,5,:,:,:], key_states_split[:,5,:,:,:], cos_velocity, sin_velocity) #apply to head group 5

        query_states = torch.cat((query_states_onset, query_states_dur, query_states_octave, query_states_pitch, query_states_instr, query_states_velocity), dim = 1) #concat all the heads
        key_states = torch.cat((key_states_onset, key_states_dur, key_states_octave, key_states_pitch, key_states_instr, key_states_velocity), dim = 1) #(bsz, head_kv, len, dim)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position} #it's not used in cache_utils.py!
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        is_causal = True if causal_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

class LlamaSdpaAttentionBaseline(LlamaAttentionBaseline):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2) #(bsz, head, len, dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position} #it's not used in cache_utils.py!
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
    "sdpa_baseline": LlamaSdpaAttentionBaseline,
    "output":LlamaOutputAttention 
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: Union[LlamaConfig, CustomConfig], layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        additional_token_map: Optional[Dict] = None,
        additional_tokens_pos_map: Optional[Dict] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            additional_token_map = additional_token_map, #TODO: COMMU, add pos for min max vel, pitch
            additional_tokens_pos_map = additional_tokens_pos_map
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""



class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, padding_idx=None):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx)
    
    def forward(self, inp):
        # device_type = inp.device.type
        # device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # self.to(device_type)
        return self.embedding(inp)


class Fundamental_Music_Embedding(nn.Module):
    def __init__(self, dim, base, padding_idx = None, device=None):
        super().__init__() 
        self.d_model = dim
        self.base = base
        translation_bias = torch.rand((1, self.d_model), dtype = torch.float32).to(device)
        translation_bias = nn.Parameter(translation_bias, requires_grad=True)
        self.register_parameter("translation_bias", translation_bias)

        i = torch.arange(self.d_model)
        angle_rates = 1 / torch.pow(self.base, (2 * (i//2)) / self.d_model)
        self.angles  = angle_rates[None, ... ]
        self.linear_fme = nn.Linear(self.d_model, self.d_model)

    def __call__(self, inp):
        assert inp.dim()==2
        device_type = inp.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        data_type = self.linear_fme.weight.dtype
        # self.to(device_type)

        inp = inp[..., None] #pos (batch, num_pitch, 1)

        angle_rads = inp*self.angles.to(device_type, dtype=data_type) #(batch, num_pitch)*(1,dim)

        # apply sin to even indices in the array; 2i
        angle_rads[:, :, 0::2] = torch.sin(angle_rads.clone()[:, : , 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, :, 1::2] = torch.cos(angle_rads.clone()[:, :, 1::2])

        pos_encoding = angle_rads.to(data_type)
        
        pos_encoding += self.translation_bias.to(data_type)
        out = self.linear_fme(pos_encoding)
        
        return out

EMBEDDING_METHODS = {
    "WE": WordEmbedding,
    "FME": Fundamental_Music_Embedding,
}
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)

class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.sos_token = config.sos_token
        self.eos_token = config.eos_token

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) #Llama's implementation of word embedding

        #First distribute embedding dimensions based on total hidden_size and number of heads

        emb_size = config.hidden_size //6 
        self.onset_embedding = EMBEDDING_METHODS[config.onset_embedding['method']](dim = emb_size, **{k: v for k, v in config.onset_embedding.items() if k != 'method'})
        self.dur_embedding = EMBEDDING_METHODS[config.dur_embedding['method']](dim = emb_size, **{k: v for k, v in config.dur_embedding.items() if k != 'method'})
        self.octave_embedding = EMBEDDING_METHODS[config.octave_embedding['method']](dim = emb_size, **{k: v for k, v in config.octave_embedding.items() if k != 'method'})
        self.pitch_embedding = EMBEDDING_METHODS[config.pitch_embedding['method']](dim = emb_size, **{k: v for k, v in config.pitch_embedding.items() if k != 'method'})
        self.instrument_embedding = EMBEDDING_METHODS[config.instrument_embedding['method']](dim = emb_size, **{k: v for k, v in config.instrument_embedding.items() if k != 'method'})
        self.velocity_embedding = EMBEDDING_METHODS[config.velocity_embedding['method']](dim = emb_size, **{k: v for k, v in config.velocity_embedding.items() if k != 'method'})
        
        self.supplementary_embedding = nn.Embedding(2, config.hidden_size) #one for sos and one for eos
        self.supplementary_MLP = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size //2 ),
            nn.ReLU(),              # Activation function
            nn.Linear(config.hidden_size//2 , config.hidden_size)
        )   

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    def embed_tokens(self, input_ids, additional_token_map = None):
        #(onset, duration, octave, pitch_class, instrument, velocity); batch, len, 6 --> batch, len, dim*6 
        #1. scan through the onset and detect positions of SOS and EOS;         
        #2. where sos --> embed using sos; where eos --> embed using eos, other places embed using FME
        #3. skew them together 
        #additional_token_map (Optional[dict]): A mapping of new token IDs (key) to embedding indices in supplementary_embedding. Example: {token_id: embedding_index}
        #SOS, EOS tokens are embedded seperately
        sos = self.supplementary_embedding(torch.tensor(0).to(input_ids.device))[None, None, ...].expand(input_ids.size(0), -1, -1) #dim*6 --> 1, 1, dim*6 --> batch, 1, dim*6
        eos = self.supplementary_embedding(torch.tensor(1).to(input_ids.device))[None, None, ...].expand(input_ids.size(0), -1, -1)

        #Detect SOS and EOS:
        where_sos = (input_ids[:, :, 0] == self.sos_token).unsqueeze(-1)
        where_eos = (input_ids[:, :, 0] == self.eos_token).unsqueeze(-1)

        # Handle new tokens if provided
        if additional_token_map is not None:
            where_new_tokens_dict = {}
            for token_id in additional_token_map.keys():
                # Create a mask for the current new token
                where_new_tokens_dict[token_id] = (input_ids[:, :, 0] == token_id).unsqueeze(-1)  # (batch, seq_len, 1)

            new_token_embeddings = {}
            for token_id, embed_idx in additional_token_map.items():
                # Embed using supplementary_embedding
                new_token_embeddings[token_id] = self.supplementary_embedding_metadata(
                    torch.tensor(embed_idx).to(input_ids.device)
                )[None, None, ...].expand(input_ids.size(0), -1, -1)

        #Since SOS and EOS are negative number, temporarily change it to 0 to avoid indexing error
        input_ids_tmp = torch.where((where_sos|where_eos), torch.tensor([0 for _ in range(6)]).to(input_ids), input_ids)

        # Also replace new tokens with the same value if additional_token_map is provided
        if additional_token_map is not None:
            for token_id in additional_token_map:
                # Detect where the new token is located
                where_new_token = where_new_tokens_dict[token_id]
                # Replace the new token in input_ids_tmp
                input_ids_tmp = torch.where(where_new_token, torch.tensor([0 for _ in range(6)]).to(input_ids), input_ids_tmp)
        onsets = self.onset_embedding(input_ids_tmp[..., 0])
        durs = self.dur_embedding(input_ids_tmp[..., 1])
        octaves = self.octave_embedding(input_ids_tmp[..., 2]) 
        pitch_classes = self.pitch_embedding(input_ids_tmp[..., 3])
        instruments = self.instrument_embedding(input_ids_tmp[..., 4]) 
        velocities = self.velocity_embedding(input_ids_tmp[..., 5])
        out_fme = torch.concat([onsets, durs, octaves, pitch_classes, instruments, velocities], dim=-1) #batch, len, dim*6
        
        #skew them together
        out_fme_sos = torch.where(where_sos, sos, out_fme) #batch, len, 1; batch, 1, dim; batch, len, dim
        
        out_fme_sos_eos = torch.where(where_eos, eos, out_fme_sos)

        # Handle new tokens using precomputed embeddings and masks
        out_final = out_fme_sos_eos  # Start with SOS/EOS-processed embeddings
        if additional_token_map is not None:
            for token_id in additional_token_map:
                # Retrieve the embedding and mask for this new token
                new_token_embedding = new_token_embeddings[token_id]  # Precomputed embedding
                where_new_token = where_new_tokens_dict[token_id]  # Precomputed mask

                # Replace embeddings for the current new token
                out_final = torch.where(where_new_token, new_token_embedding, out_final)
        #Additional non-linearity to the embeddings

        out_final = self.supplementary_MLP(out_final)

        return out_final
    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        additional_tokens_list: Optional[List[int]] = None,
        additional_tokens_pos_map: Optional[Dict[str, List[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            if additional_tokens_list is not None: #additional_tokens_pos
                additional_token_map = {token_id: i for i, token_id in enumerate(additional_tokens_list)}
            else:
                additional_token_map = None
            inputs_embeds = self.embed_tokens(input_ids, additional_token_map = additional_token_map) #batch, len, 6 --> batch, len, dim*6
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None: #not actually in use during inference
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None: #not actually in use 
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training: #TODO: how to turn this on during training
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    input_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids, #position id is equivalent to input id which carries info about onset, dur, etc.
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    additional_token_map = additional_token_map, #COMMU, add pos for min max vel, pitch
                    additional_tokens_pos_map = additional_tokens_pos_map
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        if attention_mask is None: #During inference, use KV cache
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

            dtype, device = input_tensor.dtype, input_tensor.device
            min_dtype = torch.finfo(dtype).min
            sequence_length = input_tensor.shape[1]

            target_length = past_seen_tokens + sequence_length

            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)

            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1) #during inference, *= not actually in use

            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1) #batch, 1, seq_len, tar_len

            return causal_mask 
        
        if len(attention_mask.shape) == 2: #During training or evaluation, attention_mask contains concatenated data
            attention_mask = attention_mask[:, None, :] #batch, 1, len
            attention_mask_rep = attention_mask.expand(-1, attention_mask.shape[2], -1) #batch, len, len
            block_mask = (attention_mask_rep == attention_mask_rep.transpose(1, 2)) 
            #Create a causal mask for each block, ensuring tokens only attend to previous tokens in their block
            seq_len = attention_mask.shape[2]
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=attention_mask.device))
            attention_mask = block_mask & causal_mask  # Shape: (batch_size, seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(1) #unsqueeze in head dimension: batch, len, len
            return attention_mask 

    def add_supplementary_embedding(self, num_tokens, embedding_name, hidden_size):
        new_embedding = nn.Embedding(num_tokens, hidden_size)
        new_embedding.require_grad = True
        setattr(self, embedding_name, new_embedding)

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)

        output_decoder_config= {k: v for k, v in config.decoder.items()}
        decoder_config = LlamaConfig.from_dict(output_decoder_config)
        self.attn_implementation = config._attn_implementation #this determines whether or not to apply music RoPE
        self.decoder_attn_implementation = decoder_config._attn_implementation
        self.decoder = DECODING_METHODS[decoder_config._attn_implementation](decoder_config)

        #TODO: add projection layer to shrink the size! nn.Embedding(config.decode_vocab_size, config.decoder.hidden_size)
        self.decoder_embedding = nn.Embedding(config.decode_vocab_size, config.decoder["hidden_size"])
        self.summary_projection = nn.Linear(config.hidden_size, config.decoder["hidden_size"], bias=False) 
        #TODO: add projection layer to shrink the size! nn.Embedding(config.decoder.hidden_size, config.decode_vocab_size)
        self.lm_head = nn.Linear(config.decoder["hidden_size"], config.decode_vocab_size, bias=False) 


        # # Initialize weights array with ones
        # weights = torch.ones(self.config.decode_vocab_size)

        # # Set weight for onset=0 to 0.5
        # weights[0] = 0.5

        # # Set weight for the rest of the classes to 1.0
        # weights[1:] = 1.0
        # weights = weights.float()
        # print(f"check weight? {weights.dtype}") #torch.float32
        # self.loss_func = CrossEntropyLoss(weight=weights)
        
        self.loss_func = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # need to be provided during training/evaluation
        input_ids_encoded: Optional[torch.Tensor] = None, # need to be provided during inference
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        past_key_values_decoder: Optional[Union[Cache, List[torch.FloatTensor]]] = None, #TODO: add another cache for decoder
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
		decoded_language_tokens: Optional[torch.LongTensor] = None,
        decoded_hidden_state: Optional[torch.LongTensor] = None, #batch, len, dim
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if input_ids is not None and decoded_hidden_state is None: #training/evaluation
            if self.attn_implementation == "sdpa":
                position_ids = input_ids
            elif self.attn_implementation == "sdpa_baseline":
                position_ids = position_ids
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids, #if sdpa: position_ids carry information about onset, dur, pitch, instr, vel; elif sdpa_baseline: position_ids are None
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            hidden_states = outputs[0]
            logits = hidden_states
            logits_shrinked = self.summary_projection(hidden_states) #batch, len, dim --> batch, len, decoder_hidden_size

        elif input_ids is None and decoded_hidden_state is not None: #inference
            hidden_states = decoded_hidden_state
        else:
            print("warning You cannot provide input_ids and input_ids_encoded at the same time!", )
            # assert False, "You cannot provide input_ids and input_ids_encoded at the same time!"
        logits = None
        loss = None
        generation_logits = None
        generation_hidden_state = None
        if labels is not None: #if label exists: during training/evaluation --> teacher forcing, return loss;
            shift_logits_x = logits_shrinked[..., :-1, :].contiguous() #batch, len_x-1, dim
            shift_labels_x = labels[..., 1:, :].contiguous().to(logits_shrinked.device)

            if self.decoder_attn_implementation == "output": #DANGEROURS: here shift labels does not contain SOS_decoding token 
                # 1. get the "SOS" token for each decoding step
                music_summary = shift_logits_x.view(-1, shift_logits_x.shape[-1]).unsqueeze(1) #batch*(len_x-1), 1, dim 
                
                #2. shift the labels and concat with intermediate "SOS" tokens: music summary
                shift_labels_x = shift_labels_x.view(-1, shift_labels_x.shape[-1]) #batch*(len_x-1), onset_vocab_size + dur_size + .. + vel_size / batch*(len_x-1), len_y
                

                shift_labels_x_y_encoded = self.decoder_embedding(shift_labels_x[:, :-1]) #batch*(len_x-1), len_y-1, dim
                decoder_input = torch.concat([music_summary, shift_labels_x_y_encoded], dim = 1) #batch*(len_x-1), len_y, dim


                generation_logits = self.decoder(
                    input_ids=None,
                    attention_mask=None, 
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=decoder_input,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    cache_position=None)
            elif self.decoder_attn_implementation == "MLP": #DANGEROURS: here shift labels does not contain SOS_decoding token 

                music_summary = shift_logits_x.view(-1, shift_logits_x.shape[-1]) #batch*(len_x-1), dim 
                generation_logits = self.decoder(music_summary) #batch*(len_x-1), len_y*dim 
                generation_logits = generation_logits.view(music_summary.shape[0], shift_labels_x.shape[-1], -1) #batch*(len_x-1), len_y, decode_vocab_size
                generation_logits = [generation_logits]
            
            elif self.decoder_attn_implementation == "GRU": #DANGEROURS: here shift labels contain SOS_decoding token / does not need EOS?
                shift_logits_x_flattened = shift_logits_x.view(-1, shift_logits_x.shape[-1]) #batch*(len_x-1), dim 
                shift_labels_x_flattened = shift_labels_x.view(-1, shift_labels_x.shape[-1]) #batch*(len_x-1), onset_vocab_size + dur_size + .. + vel_size / batch*(len_x-1), len_y
                
                shift_labels_x_y = shift_labels_x_flattened[:, 1:].contiguous() #batch*(len_x-1), len_y-1(1:)
                
                shift_labels_x_y_encoded = self.decoder_embedding(shift_labels_x_flattened[:, :-1]) #batch*(len_x-1), len_y-1(:-1), dim
                generation_logits, generation_hidden_state = self.decoder(shift_labels_x_y_encoded, shift_logits_x_flattened.unsqueeze(0).expand(self.decoder.num_hidden_layers, -1, -1)) #batch*(len_x-1), len_y-1, decode_vocab_size

                generation_logits = [generation_logits]

            elif self.decoder_attn_implementation == "LSTM": #DANGEROURS: here shift labels contain SOS_decoding token 
                print("not yet implemented")

            generation_logits= self.lm_head(generation_logits[0]).float().view(-1, self.config.decode_vocab_size)
            shift_labels_x_y = shift_labels_x_y.view(-1)
            loss = self.loss_func(generation_logits, shift_labels_x_y)
        elif decoded_language_tokens is not None and decoded_hidden_state is not None: #else during inference (decoding)--> inference autoregressively, return generated tokens
            if self.decoder_attn_implementation == "GRU":              
                decoded_language_tokens_encoded = self.decoder_embedding(decoded_language_tokens)##batch*len_x, len_y--> batch*lenx, len_y, dim
                generation_logits_flattened, generation_hidden_state_flattened = self.decoder(decoded_language_tokens_encoded, decoded_hidden_state) #output: batch*len_x, len_y, dim ,  hidden state: num_layers, batch*len_x, dim
                
                generation_logits = generation_logits_flattened.view(decoded_language_tokens_encoded.shape[0],decoded_language_tokens_encoded.shape[1], -1) #batch*len_x, len_y, decode_vocab_size
                generation_hidden_state = generation_hidden_state_flattened.view(self.decoder.num_hidden_layers, decoded_language_tokens_encoded.shape[0], -1) #num_layers, batch*len_x, dim
                generation_logits= self.lm_head(generation_logits)
                generation_logits = generation_logits.float()
                logits_shrinked = None


        #final todo: return logits intermediate, loss, and decoded tokens 
        

        if not return_dict:
            output = (logits,generation_logits) + outputs[1:]
            return (loss, logits) + output if loss is not None else output 
        if input_ids is not None and input_ids_encoded is None: #during training and evaluation
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits_shrinked,
                generation_logits=generation_logits,
                generation_hidden_state=generation_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions, 
            )
        else: #during inference
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits_shrinked,
                generation_logits=generation_logits,
                generation_hidden_state=generation_hidden_state,
                past_key_values=None,
                hidden_states=None,
                attentions=None, 
            )        

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class LlamaForCausalLM_Conditional_Generation(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.sos_token = config.sos_token
        self.eos_token = config.eos_token
        self.soc_token = -4 
        self.eoc_token = -5
        self.model.add_supplementary_embedding(num_tokens = len(config.metadata_tokens), embedding_name = "supplementary_embedding_metadata", hidden_size = config.hidden_size) #commu_specific, add soc, eoc and metadata tokens
        self.if_add_metadata_in_decoder = config.if_add_metadata_in_decoder
        self.if_add_chord_in_decoder = config.if_add_chord_in_decoder
        self.chord_idx2symbol = {v: k for k, v in config.chord_dict.items()}
        self.bar_classes = config.bar_classes
        self.beat_classes = config.beat_classes
        self.chord_placeholder_embedding = nn.Embedding(1, config.hidden_size//6)
        output_decoder_config= {k: v for k, v in config.decoder.items()}
        decoder_config = LlamaConfig.from_dict(output_decoder_config)
        self.attn_implementation = config._attn_implementation #this determines whether or not to apply music RoPE
        self.decoder_attn_implementation = decoder_config._attn_implementation
        self.decoder = DECODING_METHODS[decoder_config._attn_implementation](decoder_config)

        #TODO: add projection layer to shrink the size! nn.Embedding(config.decode_vocab_size, config.decoder.hidden_size)
        self.decoder_embedding = nn.Embedding(config.decode_vocab_size, config.decoder["hidden_size"])
        self.summary_projection = nn.Linear(config.hidden_size, config.decoder["hidden_size"], bias=False) 
        #TODO: add projection layer to shrink the size! nn.Embedding(config.decoder.hidden_size, config.decode_vocab_size)
        self.lm_head = nn.Linear(config.decoder["hidden_size"], config.decode_vocab_size, bias=False) 
        if self.if_add_metadata_in_decoder:
            self.gru_condition_layer = nn.Linear(11*config.hidden_size, decoder_config.hidden_size, bias=False) 
        if self.if_add_chord_in_decoder:
            self.chord_condition_layer = nn.Sequential(
                nn.Linear(config.hidden_size // 6 + self.bar_classes + self.beat_classes, decoder_config.hidden_size//2, bias=False),
                nn.ReLU(), 
                nn.Linear(decoder_config.hidden_size//2, decoder_config.hidden_size, bias=False))
                
        # # Initialize weights array with ones
        # weights = torch.ones(self.config.decode_vocab_size)

        # # Set weight for onset=0 to 0.5
        # weights[0] = 0.5

        # # Set weight for the rest of the classes to 1.0
        # weights[1:] = 1.0
        # weights = weights.float()
        # print(f"check weight? {weights.dtype}") #torch.float32
        # self.loss_func = CrossEntropyLoss(weight=weights)
        
        self.loss_func = CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # need to be provided during training/evaluation
        input_ids_encoded: Optional[torch.Tensor] = None, # need to be provided during inference
        metadata_condition: Optional[torch.Tensor] = None, # need to be provided during inference
        bar_beat_chord_condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        past_key_values_decoder: Optional[Union[Cache, List[torch.FloatTensor]]] = None, #TODO: add another cache for decoder
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
		decoded_language_tokens: Optional[torch.LongTensor] = None,
        decoded_hidden_state: Optional[torch.LongTensor] = None, #batch, len, dim
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if input_ids is not None and decoded_hidden_state is None: #training/evaluation
            import random
            #with 10% probablity remove <metadata> tokens
            # print("chords not dropped!", input_ids.shape, attention_mask.shape, labels.shape)

            #with 10% probablity remove chord tokens:  for each data in one batch, 1. find indices between <soc> and <eoc>, record the length: L , between <sos> and <eos>, 2. stitch the <metadata> tokens, tokens between <sos> and <eos>, and add the length L to the end of the sequence, 3. concat all data in this batch
            """            if random.random() < 0.2 and self.training:
                print("chords dropped!")
                input_ids_no_chord = []
                input_attention_mask_no_chord = []
                labels_no_chord = []

                def stitch_input(input_ids,seq_indices_sos, seq_indices_eos, seq_indices_soc, seq_indices_eoc):

                    #1. find indices between <soc> and <eoc>, record the length: L , between <sos> and <eos>, 2. stitch the <metadata> tokens, tokens between <sos> and <eos>, and add the length L to the end of the sequence, 3. concat all data in this batch
                    input_id_metadata = input_ids[:seq_indices_soc]

                    chord_length = len(input_ids[seq_indices_soc:seq_indices_eoc+1])

                    input_id_music= input_ids[seq_indices_sos:seq_indices_eos+1]
                    input_id_pad = input_ids[seq_indices_eos+1: ]
                    # print(f"input_id_metadata:{input_id_metadata}, input_id_music:{input_id_music}, input_id_pad:{input_id_pad}")
                    # Extract the first element and repeat it

                    first_element_pad = input_id_pad[0].unsqueeze(0) # Shape (1, dim) or (1, )
                    # print(f"first_element_pad:{first_element_pad}")
                    if input_ids.dim() == 2:
                        # print(f"input_ids.dim() == 2")
                        first_element_pad_extended = first_element_pad.repeat(chord_length, 1) #(len, dim)
                    elif input_ids.dim() == 1:
                        # print(f"input_ids.dim() == 1")
                        first_element_pad_extended = first_element_pad.repeat(chord_length)  #(len,)
                    input_id_pad_extended = torch.cat([first_element_pad_extended, input_id_pad], dim = 0)
                    # print(f"first_element_pad_extended:{first_element_pad_extended}, input_id_pad_extended:{input_id_pad_extended}")


                    input_id_metadata_music_pad = torch.cat([input_id_metadata, input_id_music, input_id_pad_extended], dim = 0)

                    return input_id_metadata_music_pad

                for batch_idx in range(len(input_ids)):
                    # Find the indices of the `sos` and `eos` tokens in the current sequence
                    seq_indices_sos = (input_ids[batch_idx, :, 0] == self.sos_token).nonzero(as_tuple=True)[0].item()
                    seq_indices_eos = (input_ids[batch_idx, :,0] == self.eos_token).nonzero(as_tuple=True)[0].item()
                    seq_indices_soc = (input_ids[batch_idx, :, 0] == self.soc_token).nonzero(as_tuple=True)[0].item()
                    seq_indices_eoc = (input_ids[batch_idx, :, 0] == self.eoc_token).nonzero(as_tuple=True)[0].item()

                    i = stitch_input(input_ids[batch_idx], seq_indices_sos, seq_indices_eos, seq_indices_soc, seq_indices_eoc)
                    m = stitch_input(attention_mask[batch_idx], seq_indices_sos, seq_indices_eos, seq_indices_soc, seq_indices_eoc)
                    l = stitch_input(labels[batch_idx], seq_indices_sos, seq_indices_eos, seq_indices_soc, seq_indices_eoc)

                    input_ids_no_chord.append(i)
                    input_attention_mask_no_chord.append(m)
                    labels_no_chord.append(l)
                # print("chords dropped!", input_ids.shape, attention_mask.shape, labels.shape)
                input_ids = torch.stack(input_ids_no_chord, dim = 0)
                attention_mask = torch.stack(input_attention_mask_no_chord, dim = 0)
                labels = torch.stack(labels_no_chord, dim = 0)
                """
            if self.attn_implementation == "sdpa":
                position_ids = input_ids
            elif self.attn_implementation == "sdpa_baseline":
                position_ids = position_ids
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids, #if sdpa: position_ids carry information about onset, dur, pitch, instr, vel; elif sdpa_baseline: position_ids are None
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                additional_tokens_list=self.config.metadata_tokens,
                additional_tokens_pos_map=self.config.metadata_tokens_pos
            )
            hidden_states = outputs[0]
            logits = hidden_states
            logits_shrinked = self.summary_projection(hidden_states) #batch, len, dim --> batch, len, decoder_hidden_size

        elif input_ids is None and decoded_hidden_state is not None: #inference
            hidden_states = decoded_hidden_state
        else:
            print("warning You cannot provide input_ids and input_ids_encoded at the same time!", )
            # assert False, "You cannot provide input_ids and input_ids_encoded at the same time!"
        logits = None
        loss = None
        generation_logits = None
        generation_hidden_state = None
        additional_token_map = {token_id: i for i, token_id in enumerate(self.config.metadata_tokens)}
        if labels is not None: #if label exists: during training/evaluation --> teacher forcing, return loss;
            shift_logits_x = logits_shrinked[..., :-1, :].contiguous() #batch, len_x-1, dim
            shift_labels_x = labels[..., 1:, :].contiguous().to(logits_shrinked.device)

            #gather logits between each sos and eos token 
            batch_size, seq_len, _ = input_ids.shape
            
            logits_list = []
            labels_list = []
            metadata_condition_list = []
            bar_beat_chord_condition_list = []
            for batch_idx in range(batch_size):
                # Find the indices of the `sos` and `eos` tokens in the current sequence
                seq_indices_sos = (input_ids[batch_idx, :, 0] == self.sos_token).nonzero(as_tuple=True)[0].item()
                seq_indices_eos = (input_ids[batch_idx, :,0] == self.eos_token).nonzero(as_tuple=True)[0].item()
                
                # Gather logits between `sos` and `eos` tokens (exclusive of `sos` and `eos` themselves)
                logits_between_sos_eos = shift_logits_x[batch_idx, seq_indices_sos : seq_indices_eos]
                labels_between_sos_eos = shift_labels_x[batch_idx, seq_indices_sos : seq_indices_eos]
                if self.if_add_metadata_in_decoder:
                    metadata_condition_embedded_single = self.model.supplementary_embedding_metadata(torch.tensor([additional_token_map[token.item()] for token in metadata_condition[batch_idx]]).to(input_ids)).reshape(-1).unsqueeze(0).expand(seq_indices_eos - seq_indices_sos, -1).to(shift_logits_x) #11, dim --> 11*dim --> (1, 11*dim), (len, 11*dim)
                    metadata_condition_shrinked = self.gru_condition_layer(metadata_condition_embedded_single).unsqueeze(1) #(len, 11*dim) --> (len, dim) --> len, 1, dim
                    metadata_condition_list.append(metadata_condition_shrinked)
                if self.if_add_chord_in_decoder: 
                    bar_OH = F.one_hot(bar_beat_chord_condition[batch_idx, : seq_indices_eos - seq_indices_sos, 0].long(), num_classes=self.bar_classes).to(input_ids) #(len, bar_classes)
                    beat_OH = F.one_hot(bar_beat_chord_condition[batch_idx, : seq_indices_eos - seq_indices_sos, 1].long(), num_classes=self.beat_classes).to(input_ids) #(len, beat_classes)
                    #FME/WE:
                    chord_condition = []
                    for chord in [self.chord_idx2symbol[chord_idx.item()] for chord_idx in bar_beat_chord_condition[batch_idx, : seq_indices_eos - seq_indices_sos, 2]]:
                        if chord!="s":
                            embedded_pitches = self.model.pitch_embedding(torch.tensor(chord_to_midi(chord)).unsqueeze(0).to(input_ids)) # (1, num_pitches, dim)
                            chord_condition.append(embedded_pitches.sum(dim = 1)) # (1, dim)
                        else: 
                            chord_condition.append(self.chord_placeholder_embedding(torch.tensor([0]).to(input_ids))) # (1, dim)
                    chord_condition_cat = torch.cat(chord_condition, dim = 0) # (len, dim)
                    bar_beat_chord_condition_cat = torch.cat([bar_OH, beat_OH, chord_condition_cat], dim = -1)
                    bar_beat_chord_condition_cat_linear = self.chord_condition_layer(bar_beat_chord_condition_cat).unsqueeze(1)

                    bar_beat_chord_condition_list.append(bar_beat_chord_condition_cat_linear)

                logits_list.append(logits_between_sos_eos)
                labels_list.append(labels_between_sos_eos)
                
            shift_logits_x = torch.cat(logits_list, dim = 0) #len_concat, dim
            shift_labels_x = torch.cat(labels_list, dim = 0) #len_concat, onset_vocab_size + dur_size + .. + vel_size
            if self.if_add_metadata_in_decoder:
                metadata_condition_embedded = torch.cat(metadata_condition_list, dim = 0) #(len_concat, 1, dim)
            if self.if_add_chord_in_decoder:
                bar_beat_chord_condition_embedded = torch.cat(bar_beat_chord_condition_list, dim = 0) #(len_concat, 1, dim)
            if self.decoder_attn_implementation == "output": #DANGEROURS: here shift labels does not contain SOS_decoding token 
                # 1. get the "SOS" token for each decoding step
                music_summary = shift_logits_x.view(-1, shift_logits_x.shape[-1]).unsqueeze(1) #batch*(len_x-1), 1, dim 
                
                #2. shift the labels and concat with intermediate "SOS" tokens: music summary
                shift_labels_x = shift_labels_x.view(-1, shift_labels_x.shape[-1]) #batch*(len_x-1), onset_vocab_size + dur_size + .. + vel_size / batch*(len_x-1), len_y
                

                shift_labels_x_y_encoded = self.decoder_embedding(shift_labels_x[:, :-1]) #batch*(len_x-1), len_y-1, dim
                decoder_input = torch.concat([music_summary, shift_labels_x_y_encoded], dim = 1) #batch*(len_x-1), len_y, dim


                generation_logits = self.decoder(
                    input_ids=None,
                    attention_mask=None, 
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=decoder_input,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    cache_position=None)
            elif self.decoder_attn_implementation == "MLP": #DANGEROURS: here shift labels does not contain SOS_decoding token 

                music_summary = shift_logits_x.view(-1, shift_logits_x.shape[-1]) #batch*(len_x-1), dim 
                generation_logits = self.decoder(music_summary) #batch*(len_x-1), len_y*dim 
                generation_logits = generation_logits.view(music_summary.shape[0], shift_labels_x.shape[-1], -1) #batch*(len_x-1), len_y, decode_vocab_size
                generation_logits = [generation_logits]
            
            elif self.decoder_attn_implementation == "GRU": #DANGEROURS: here shift labels contain SOS_decoding token / does not need EOS?
                shift_logits_x_flattened = shift_logits_x.view(-1, shift_logits_x.shape[-1]) #batch*(len_x-1), dim 
                shift_labels_x_flattened = shift_labels_x.view(-1, shift_labels_x.shape[-1]) #batch*(len_x-1), onset_vocab_size + dur_size + .. + vel_size / batch*(len_x-1), len_y
                
                shift_labels_x_y = shift_labels_x_flattened[:, 1:].contiguous() #batch*(len_x-1), len_y-1(1:)
                
                shift_labels_x_y_encoded = self.decoder_embedding(shift_labels_x_flattened[:, :-1]) #batch*(len_x-1), len_y-1(:-1), dim

                if self.if_add_metadata_in_decoder:
                    shift_labels_x_y_encoded =  shift_labels_x_y_encoded + metadata_condition_embedded
                if self.if_add_chord_in_decoder:
                    shift_labels_x_y_encoded =  shift_labels_x_y_encoded + bar_beat_chord_condition_embedded
                generation_logits, generation_hidden_state = self.decoder(shift_labels_x_y_encoded, shift_logits_x_flattened.unsqueeze(0).expand(self.decoder.num_hidden_layers, -1, -1)) #batch*(len_x-1), len_y-1, decode_vocab_size

                generation_logits = [generation_logits]

            elif self.decoder_attn_implementation == "LSTM": #DANGEROURS: here shift labels contain SOS_decoding token 
                print("not yet implemented")

            generation_logits= self.lm_head(generation_logits[0]).float().view(-1, self.config.decode_vocab_size)
            shift_labels_x_y = shift_labels_x_y.view(-1)
            loss = self.loss_func(generation_logits, shift_labels_x_y)
        elif decoded_language_tokens is not None and decoded_hidden_state is not None: #else during inference (decoding)--> inference autoregressively, return generated tokens
            if self.decoder_attn_implementation == "GRU":    
                decoded_language_tokens_encoded = self.decoder_embedding(decoded_language_tokens)##batch*len_x, len_y--> batch*lenx, len_y, dim
                decoder_input = decoded_language_tokens_encoded
                if self.if_add_metadata_in_decoder:
                    metadata_condition = self.model.supplementary_embedding_metadata(
                        torch.tensor([
                            additional_token_map[token.item()] 
                            for batch in metadata_condition # Iterate through the first 11 tokens in each batch
                            for token in batch                  # Iterate through each token in the batch
                        ])
                    ).reshape(metadata_condition.shape[0] , -1).unsqueeze(1)  # Reshape and move to the same device as input_ids #batch, 11 --> batch, 11, dim --> batch, 11*dim, --> batch, 1, 11*dim
                    metadata_condition_shrinked = self.gru_condition_layer(metadata_condition) #(batch, 1, 11*dim) --> (batch, 1, dim)
                    decoder_input = decoder_input+metadata_condition_shrinked
                if self.if_add_chord_in_decoder:
                    # bar_beat_chord_condition: batch*len, 3 --> batch*len, 1, dim
                    bar_OH = F.one_hot(bar_beat_chord_condition[:, 0].long(), num_classes=self.bar_classes).to(input_ids) #(batch*len, bar_classes)
                    beat_OH = F.one_hot(bar_beat_chord_condition[:, 1].long(), num_classes=self.beat_classes).to(input_ids) #(batch*len, beat_classes)
                    #FME/WE:
                    chord_condition = []
                    for chord in [self.chord_idx2symbol[chord_idx.item()] for chord_idx in bar_beat_chord_condition[:, 2]]:
                        if chord!="s":
                            embedded_pitches = self.model.pitch_embedding(torch.tensor(chord_to_midi(chord)).unsqueeze(0).to(input_ids)) # (1, num_pitches, dim)
                            chord_condition.append(embedded_pitches.sum(dim = 1)) # (1, dim)
                        else: 
                            chord_condition.append(self.chord_placeholder_embedding(torch.tensor([0]).to(input_ids))) # (1, dim)
                    chord_condition_cat = torch.cat(chord_condition, dim = 0) # (batch*len, dim)
                    bar_beat_chord_condition_cat = torch.cat([bar_OH, beat_OH, chord_condition_cat], dim = -1)
                    bar_beat_chord_condition_cat_linear = self.chord_condition_layer(bar_beat_chord_condition_cat).unsqueeze(1)
                    decoder_input = decoder_input+bar_beat_chord_condition_cat_linear

                generation_logits_flattened, generation_hidden_state_flattened = self.decoder(decoder_input, decoded_hidden_state) #output: batch*len_x, len_y, dim ,  hidden state: num_layers, batch*len_x, dim
                
                generation_logits = generation_logits_flattened.view(decoded_language_tokens_encoded.shape[0],decoded_language_tokens_encoded.shape[1], -1) #batch*len_x, len_y, decode_vocab_size
                generation_hidden_state = generation_hidden_state_flattened.view(self.decoder.num_hidden_layers, decoded_language_tokens_encoded.shape[0], -1) #num_layers, batch*len_x, dim
                generation_logits= self.lm_head(generation_logits)
                generation_logits = generation_logits.float()
                logits_shrinked = None


        #final todo: return logits intermediate, loss, and decoded tokens 
        

        if not return_dict:
            output = (logits,generation_logits) + outputs[1:]
            return (loss, logits) + output if loss is not None else output 
        if input_ids is not None and input_ids_encoded is None: #during training and evaluation
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits_shrinked,
                generation_logits=generation_logits,
                generation_hidden_state=generation_hidden_state,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions, 
            )
        else: #during inference
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits_shrinked,
                generation_logits=generation_logits,
                generation_hidden_state=generation_hidden_state,
                past_key_values=None,
                hidden_states=None,
                attentions=None, 
            )        

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class OutputMLP(nn.Module):
    def __init__(self, config):
        super(OutputMLP, self).__init__()
        hidden_size = config.hidden_size
        num_hidden_layers = config.num_hidden_layers
        self.input_size = hidden_size
        self.output_size = 6*hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        # Create a list to hold all layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, self.output_size))
        
        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class OutputLSTM(nn.Module):
    def __init__(self, config):
        super(OutputLSTM, self).__init__()

        hidden_size = config.hidden_size
        output_size = config.hidden_size
        num_hidden_layers = config.num_hidden_layers
        self.num_hidden_layers = num_hidden_layers
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=num_hidden_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden, cell):
        # x: [batch_size, seq_len, output_dim]
        # hidden: [batch_size, hidden_dim]
        # cell: [batch_size, hidden_dim]
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        logits = self.fc_out(output)  # [batch_size, seq_len, output_dim]
        return logits, hidden, cell

class OutputGRU(nn.Module):
    def __init__(self, config):
        super(OutputGRU, self).__init__()

        hidden_size = config.hidden_size
        output_size = config.hidden_size
        num_hidden_layers = config.num_hidden_layers
        self.num_hidden_layers = num_hidden_layers
        self.gru = nn.GRU(output_size, hidden_size, num_layers=num_hidden_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden):
        # x: [batch_size, seq_len, output_dim]
        # hidden: [num_layers, batch_size, hidden_dim] 
        # cell: [batch_size, hidden_dim]
        output, hidden = self.gru(x, hidden)
        logits = self.fc_out(output)  # [batch_size, seq_len, output_dim]
        return logits, hidden

DECODING_METHODS = {
    "MLP": OutputMLP,
    "LSTM": OutputLSTM,
    "GRU": OutputGRU
}


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
