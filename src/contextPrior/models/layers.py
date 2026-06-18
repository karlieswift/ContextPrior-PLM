from __future__ import annotations

import torch
import torch.nn as nn

def get_mask_fill_value(x: torch.Tensor) -> float:
    if x.dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e6
def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, dim):
    mask = mask.to(x.dtype)
    summed = (x * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return summed / denom


def safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    all_pad = mask.all(dim=-1, keepdim=True)
    return mask & (~all_pad)


class FeedForwardResidual(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MotifAxisAttention(nn.Module):
    """Self-attention along motif axis T for each residue slot f.

    Input / output: [B, T, F, D]
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1,
                 attn_scale: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardResidual(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn_scale = attn_scale

    def forward(self, x: torch.Tensor, motif_mask: torch.Tensor):
        bsz, num_motifs, motif_len, dim = x.shape
        xt = x.permute(0, 2, 1, 3).contiguous().view(bsz * motif_len, num_motifs, dim)

        key_padding_mask = ~motif_mask.unsqueeze(1).expand(bsz, motif_len, num_motifs)
        key_padding_mask = key_padding_mask.contiguous().view(bsz * motif_len, num_motifs)
        key_padding_mask = safe_key_padding_mask(key_padding_mask)

        attn_out, attn_weights = self.attn(
            xt, xt, xt, key_padding_mask=key_padding_mask, need_weights=True
        )
        xt = self.norm(xt + self.attn_scale * self.dropout(attn_out))
        xt = self.ffn(xt)
        x = xt.view(bsz, motif_len, num_motifs, dim).permute(0, 2, 1, 3).contiguous()

        attn_weights = attn_weights.view(bsz, motif_len, num_motifs, num_motifs).mean(dim=1)
        valid_pair = motif_mask.unsqueeze(1) & motif_mask.unsqueeze(2)
        attn_weights = attn_weights * valid_pair.to(attn_weights.dtype)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return x, attn_weights


class MotifConvStageBlock(nn.Module):
    def __init__(self, dim: int, kernel_size=(3, 3), dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, padding=pad)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 3, 1)
        h = self.norm(h) * residue_mask.unsqueeze(-1).to(h.dtype)
        h = h.permute(0, 3, 1, 2)
        h = self.conv1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x + h


class AttentionPooling2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T, F]
        bsz, dim, num_motifs, motif_len = x.shape
        h = x.permute(0, 2, 3, 1).contiguous()
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~residue_mask, get_mask_fill_value(residue_mask))
        weights = torch.softmax(scores.view(bsz, num_motifs * motif_len), dim=-1)
        weights = weights.view(bsz, num_motifs, motif_len)
        return (h * weights.unsqueeze(-1)).sum(dim=(1, 2))


class MotifFieldToSequence(nn.Module):
    """Collapse each motif field [F, D] into one motif token [D]."""

    def __init__(self, dim: int, pooling: str = "attn"):
        super().__init__()
        if pooling not in {"attn", "mean"}:
            raise ValueError(f"Unsupported motif field pooling: {pooling}")
        self.pooling = pooling
        self.score = nn.Linear(dim, 1) if pooling == "attn" else None

    def forward(self, x: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T, F] -> [B, T, F, D]
        h = x.permute(0, 2, 3, 1).contiguous()
        if self.pooling == "mean":
            return masked_mean_pool(h, residue_mask.unsqueeze(-1), dim=2)

        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~residue_mask, get_mask_fill_value(h))
        weights = torch.softmax(scores, dim=-1)
        weights = weights * residue_mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (h * weights.unsqueeze(-1)).sum(dim=2)


class MotifFieldQueryTokenizer(nn.Module):
    """Summarize each motif field into K learnable query tokens."""

    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_heads: int,
        max_residues: int,
        dropout: float = 0.1,
        use_field_pos: bool = True,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.use_field_pos = use_field_pos
        self.query_tokens = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardResidual(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.field_pos_embed = nn.Embedding(max_residues, dim) if use_field_pos else None

    def forward(
        self,
        x: torch.Tensor,
        residue_mask: torch.Tensor,
        motif_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, D, T, F] -> [B, T, F, D]
        bsz, dim, num_motifs, motif_len = x.shape
        h = x.permute(0, 2, 3, 1).contiguous()
        if self.field_pos_embed is not None:
            pos = self.field_pos_embed(torch.arange(motif_len, device=x.device))
            h = h + pos.view(1, 1, motif_len, dim)

        context = h.view(bsz * num_motifs, motif_len, dim)
        queries = self.query_tokens.unsqueeze(0).expand(bsz * num_motifs, self.num_queries, dim)

        key_padding_mask = ~residue_mask.contiguous().view(bsz * num_motifs, motif_len)
        key_padding_mask = safe_key_padding_mask(key_padding_mask)

        attn_out, _ = self.attn(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = queries + self.dropout(attn_out)
        tokens = self.ffn(tokens)
        tokens = tokens.view(bsz, num_motifs, self.num_queries, dim)
        tokens = tokens * motif_mask.unsqueeze(-1).unsqueeze(-1).to(tokens.dtype)
        return tokens


class AttentionPooling1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T]
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~mask, get_mask_fill_value(x))
        weights = torch.softmax(scores, dim=-1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x * weights.unsqueeze(-1)).sum(dim=1)
