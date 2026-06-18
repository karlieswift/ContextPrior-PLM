"""
@Env: /anaconda3/python3.11
@Time: 2026/3/10-19:16
@Auth: karlieswift
@File: motif_encoder_sdpa.py
@Desc:
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MotifInferenceConfig


def safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    all_pad = mask.all(dim=-1, keepdim=True)
    return mask & (~all_pad)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dim, got {dim}")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device=device, dtype=dtype)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(-2)
        cos, sin = self._get_cos_sin(seq_len, q.device, q.dtype)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.value_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.value_proj(x)
        x = self.out_proj(x)
        return self.dropout(x)


class AxialSelfAttention1DSDPA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.attention_dropout = attention_dropout
        self.rope = RotaryEmbedding(self.head_dim) if use_rope else None

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if key_padding_mask is not None:
            key_padding_mask = safe_key_padding_mask(key_padding_mask)
            mask_value = -1e4 if q.dtype in (torch.float16, torch.bfloat16) else -1e9
            attn_scores = attn_scores.masked_fill(key_padding_mask[:, None, None, :], mask_value)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        if self.training and self.attention_dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.attention_dropout)
        context = torch.matmul(attn_probs, v)
        return context, attn_probs

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            q, k = self.rope(q, k)

        valid_mask = None
        if key_padding_mask is not None:
            key_padding_mask = safe_key_padding_mask(key_padding_mask)
            valid_mask = (~key_padding_mask).to(x.dtype)

        if output_attentions:
            context, attn_probs = self._manual_attention(q, k, v, key_padding_mask)
        else:
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = (~key_padding_mask)[:, None, None, :]
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
            attn_probs = None

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.out_proj(context)

        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1)

        return out, attn_probs


class MotifTAxisAttentionSDPA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()
        self.attn = AxialSelfAttention1DSDPA(
            dim=dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            use_rope=use_rope,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residue_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, num_motifs, motif_len, dim = hidden_states.shape
        x = hidden_states.permute(0, 2, 1, 3).contiguous().view(bsz * motif_len, num_motifs, dim)
        mask = ~residue_mask.permute(0, 2, 1).contiguous().view(bsz * motif_len, num_motifs)
        out, attn = self.attn(x, key_padding_mask=mask, output_attentions=output_attentions)
        out = out.reshape(bsz, motif_len, num_motifs, dim).permute(0, 2, 1, 3).contiguous()
        if attn is None:
            return out, None
        attn = attn.mean(dim=1).reshape(bsz, motif_len, num_motifs, num_motifs)
        return out, attn


class MotifFAxisAttentionSDPA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        bias: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()
        self.attn = AxialSelfAttention1DSDPA(
            dim=dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            use_rope=use_rope,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residue_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, num_motifs, motif_len, dim = hidden_states.shape
        x = hidden_states.reshape(bsz * num_motifs, motif_len, dim)
        mask = ~residue_mask.contiguous().view(bsz * num_motifs, motif_len)
        out, attn = self.attn(x, key_padding_mask=mask, output_attentions=output_attentions)
        out = out.reshape(bsz, num_motifs, motif_len, dim)
        if attn is None:
            return out, None
        attn = attn.mean(dim=1).reshape(bsz, num_motifs, motif_len, motif_len)
        return out, attn


class MotifAxialLayerSDPA(nn.Module):
    def __init__(self, cfg: MotifInferenceConfig):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        intermediate_dim = hidden_dim * cfg.motif_encoder_intermediate_mult
        self.t_norm = RMSNorm(hidden_dim, eps=cfg.motif_encoder_norm_eps)
        self.f_norm = RMSNorm(hidden_dim, eps=cfg.motif_encoder_norm_eps)
        self.ffn_norm = RMSNorm(hidden_dim, eps=cfg.motif_encoder_norm_eps)
        self.t_attn = MotifTAxisAttentionSDPA(
            dim=hidden_dim,
            num_heads=cfg.motif_encoder_num_attention_heads,
            attention_dropout=cfg.motif_encoder_attention_dropout,
            bias=cfg.motif_encoder_attn_bias,
            use_rope=cfg.motif_encoder_use_rope,
        )
        self.f_attn = MotifFAxisAttentionSDPA(
            dim=hidden_dim,
            num_heads=cfg.motif_encoder_num_attention_heads,
            attention_dropout=cfg.motif_encoder_attention_dropout,
            bias=cfg.motif_encoder_attn_bias,
            use_rope=cfg.motif_encoder_use_rope,
        )
        self.ffn = SwiGLUFeedForward(
            dim=hidden_dim,
            hidden_dim=intermediate_dim,
            dropout=cfg.motif_encoder_dropout,
            bias=cfg.motif_encoder_ffn_bias,
        )
        self.resid_dropout = nn.Dropout(cfg.motif_encoder_dropout)

    @staticmethod
    def _apply_residue_mask(x: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        return x * residue_mask.unsqueeze(-1).to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residue_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        t_out, t_attn = self.t_attn(self.t_norm(hidden_states), residue_mask, output_attentions=output_attentions)
        hidden_states = hidden_states + self.resid_dropout(t_out)
        hidden_states = self._apply_residue_mask(hidden_states, residue_mask)

        f_out, f_attn = self.f_attn(self.f_norm(hidden_states), residue_mask, output_attentions=output_attentions)
        hidden_states = hidden_states + self.resid_dropout(f_out)
        hidden_states = self._apply_residue_mask(hidden_states, residue_mask)

        ffn_out = self.ffn(self.ffn_norm(hidden_states))
        hidden_states = hidden_states + ffn_out
        hidden_states = self._apply_residue_mask(hidden_states, residue_mask)
        return hidden_states, t_attn, f_attn


@dataclass
class MotifEncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    t_attentions: Optional[Tuple[torch.Tensor, ...]] = None
    f_attentions: Optional[Tuple[torch.Tensor, ...]] = None


class MotifAxialEncoderSDPA(nn.Module):
    def __init__(self, cfg: MotifInferenceConfig):
        super().__init__()
        self.cfg = cfg
        self.layer = nn.ModuleList([MotifAxialLayerSDPA(cfg) for _ in range(cfg.motif_encoder_num_hidden_layers)])
        self.final_norm = RMSNorm(cfg.hidden_dim, eps=cfg.motif_encoder_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residue_mask: Optional[torch.Tensor] = None,
        motif_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        if residue_mask is None:
            residue_mask = torch.ones(hidden_states.shape[:-1], device=hidden_states.device, dtype=torch.bool)
        if motif_mask is None:
            motif_mask = residue_mask.any(dim=-1)
        residue_mask = residue_mask & motif_mask.unsqueeze(-1)

        all_hidden_states = () if output_hidden_states else None
        all_t_attentions = () if output_attentions else None
        all_f_attentions = () if output_attentions else None

        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states, t_attn, f_attn = layer_module(
                hidden_states,
                residue_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                all_t_attentions = all_t_attentions + (t_attn,)
                all_f_attentions = all_f_attentions + (f_attn,)

        hidden_states = self.final_norm(hidden_states)
        hidden_states = hidden_states * residue_mask.unsqueeze(-1).to(hidden_states.dtype)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            out = (hidden_states,)
            if output_hidden_states:
                out = out + (all_hidden_states,)
            if output_attentions:
                out = out + (all_t_attentions, all_f_attentions)
            return out

        return MotifEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            t_attentions=all_t_attentions,
            f_attentions=all_f_attentions,
        )
