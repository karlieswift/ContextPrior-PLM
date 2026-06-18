"""
@Env: /anaconda3/python3.11
@Time: 2026/3/18-9:22
@Auth: karlieswift
@File: graph.py
@Desc:
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
def finite_or_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # return x

def get_mask_fill_value(x: torch.Tensor) -> float:
    if x.dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e6
def safe_l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = x_fp32.norm(dim=dim, keepdim=True).clamp_min(eps)
    out = x_fp32 / denom
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(x.dtype)

class MotifStateEncoder(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, use_stats: bool = True):
        super().__init__()
        self.use_stats = use_stats
        self.score = nn.Linear(dim, 1)
        in_dim = dim * 3 + (4 if use_stats else 0)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, residue_mask: torch.Tensor, motif_mask: torch.Tensor):
        # x: [B,T,F,D]
        bsz, num_motifs, motif_len, dim = x.shape
        mask = residue_mask.to(x.dtype)

        score = self.score(x).squeeze(-1).masked_fill(~residue_mask, -1e4)
        attn = torch.softmax(score, dim=-1)
        attn = attn * residue_mask.to(attn.dtype)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn_pool = torch.sum(x * attn.unsqueeze(-1), dim=2)

        mean_pool = (x * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        max_pool = x.masked_fill(~residue_mask.unsqueeze(-1), -1e4).max(dim=2).values
        max_pool = torch.where(motif_mask.unsqueeze(-1), max_pool, torch.zeros_like(max_pool))

        pieces = [attn_pool, mean_pool, max_pool]
        aux = {"attn_weights_f": attn}

        if self.use_stats:
            valid_ratio = mask.mean(dim=2, keepdim=True)
            mean_abs = (x.abs() * mask.unsqueeze(-1)).sum(dim=(2, 3), keepdim=True)
            mean_abs = mean_abs / mask.sum(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1.0)
            mean_abs = mean_abs.squeeze(-1)

            var_pool = (((x - mean_pool.unsqueeze(2)) ** 2) * mask.unsqueeze(-1)).sum(dim=(2, 3), keepdim=True)
            var_pool = var_pool / (mask.sum(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1.0) * dim)
            var_pool = var_pool.squeeze(-1)

            pos = torch.linspace(0.0, 1.0, steps=num_motifs, device=x.device, dtype=x.dtype)
            pos = pos.view(1, num_motifs, 1).expand(bsz, num_motifs, 1)
            stats = torch.cat([valid_ratio, mean_abs, var_pool, pos], dim=-1)
            pieces.append(stats)
            aux.update({
                'valid_ratio': valid_ratio,
                'mean_abs': mean_abs,
                'var': var_pool,
                'pos': pos,
                'stats': stats,
            })

        z = torch.cat(pieces, dim=-1)
        z = self.proj(self.norm(z))
        z = z * motif_mask.unsqueeze(-1).to(z.dtype)
        return z, aux


class ConservativeGraphSmoothing(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.edge_gate = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        node: torch.Tensor,
        edge_mask: torch.Tensor,
        cosine: torch.Tensor,
        local_edge: torch.Tensor,
        nv_score: torch.Tensor,
        motif_mask: torch.Tensor,
    ):
        bsz, num_motifs, dim = node.shape

        node = finite_or_zero(node)
        cosine = finite_or_zero(cosine)
        nv_score = finite_or_zero(nv_score)

        edge_feat = torch.stack(
            [cosine, local_edge.to(node.dtype), nv_score],
            dim=-1,
        )
        gate = self.edge_gate(edge_feat) * edge_mask.unsqueeze(-1).to(node.dtype)

        node_i = node.unsqueeze(2).expand(bsz, num_motifs, num_motifs, dim)
        node_j = node.unsqueeze(1).expand(bsz, num_motifs, num_motifs, dim)
        delta_ij = node_j - node_i
        msg_ij = gate * delta_ij * edge_mask.unsqueeze(-1).to(node.dtype)
        deg = edge_mask.to(node.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = msg_ij.sum(dim=2) / deg
        agg = finite_or_zero(agg)
        out = node + self.alpha * agg
        out = self.norm1(out)
        out = self.norm2(out + self.ffn(out))
        out = out * motif_mask.unsqueeze(-1).to(out.dtype)
        out = finite_or_zero(out)
        aux = {
            'gate_mean': gate.mean(dim=-1),
            'delta_norm': delta_ij.norm(dim=-1),
            'agg_norm': agg.norm(dim=-1),
        }
        return out, aux


class GraphStateToFieldInjection(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.field_proj = nn.Linear(dim, dim)
        self.delta_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, field_x: torch.Tensor, node_raw: torch.Tensor, node_smooth: torch.Tensor,
                residue_mask: torch.Tensor, motif_mask: torch.Tensor, inject_scale: float = 0.1):
        bsz, num_motifs, motif_len, dim = field_x.shape
        delta = self.delta_proj(node_smooth - node_raw).unsqueeze(2).expand(bsz, num_motifs, motif_len, dim)
        field_h = self.field_proj(field_x)
        gate = self.gate(torch.cat([field_h, delta], dim=-1))
        out = field_x + inject_scale * gate * delta
        out = out * residue_mask.unsqueeze(-1).to(out.dtype)
        out = out * motif_mask.unsqueeze(-1).unsqueeze(-1).to(out.dtype)
        out = finite_or_zero(out)
        return out


class ContextualMotifGraphInference(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        topk_nv: int = 2,
        local_window: int = 2,
        edge_dropout: float = 0.1,
        inject_scale: float = 0.1,
        dropout: float = 0.1,
        detach_topology: bool = True,
        smoothing_alpha: float = 0.1,
        prior_exclude_local: bool = True,
    ):
        super().__init__()
        self.topk_nv = topk_nv
        self.local_window = 2
        self.edge_dropout = edge_dropout
        self.inject_scale = inject_scale
        self.detach_topology = detach_topology
        self.prior_exclude_local = prior_exclude_local
        self.layers = nn.ModuleList([
            ConservativeGraphSmoothing(dim=dim, dropout=dropout, alpha=smoothing_alpha)
            for _ in range(num_layers)
        ])
        self.field_inject = GraphStateToFieldInjection(dim=dim, dropout=dropout)

    @staticmethod
    def _masked_topk(masked_score: torch.Tensor, k: int, mask_value: float) -> torch.Tensor:
        bsz, num_motifs, _ = masked_score.shape
        k = max(0, min(k, num_motifs))
        if k == 0:
            return torch.zeros_like(masked_score, dtype=torch.bool)

        vals, idx = torch.topk(masked_score, k=k, dim=-1)
        valid = torch.isfinite(vals) & (vals > mask_value + 1.0)

        out = torch.zeros_like(masked_score, dtype=torch.bool)
        out.scatter_(-1, idx, valid)
        return out

    @staticmethod
    def _pairwise_feature_score(
        features: Optional[torch.Tensor],
        motif_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if features is None:
            return None, None

        feat = finite_or_zero(features)
        feat_valid = motif_mask & feat.abs().sum(dim=-1).gt(0)
        feat = feat * feat_valid.unsqueeze(-1).to(feat.dtype)
        feat_norm = safe_l2_normalize(feat, dim=-1, eps=1e-6)
        score = torch.matmul(feat_norm, feat_norm.transpose(1, 2))
        score = finite_or_zero(score)
        valid = feat_valid.unsqueeze(1) & feat_valid.unsqueeze(2)
        return score, valid

    def _build_graph(
        self,
        node: torch.Tensor,
        residue_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        motif_nv_features: Optional[torch.Tensor],
    ):
        bsz, num_motifs, _ = node.shape
        device = node.device
        dtype = node.dtype

        node_topo = node.detach() if self.detach_topology else node
        node_topo = finite_or_zero(node_topo)
        node_norm = safe_l2_normalize(node_topo, dim=-1, eps=1e-6)
        cosine = torch.matmul(node_norm, node_norm.transpose(1, 2))
        cosine = finite_or_zero(cosine)
        valid_pair = motif_mask.unsqueeze(1) & motif_mask.unsqueeze(2)
        eye = torch.eye(num_motifs, device=device, dtype=torch.bool).unsqueeze(0)
        valid_pair = valid_pair & (~eye)

        idx = torch.arange(num_motifs, device=device)
        dist = (idx[None, :, None] - idx[None, None, :]).abs().expand(bsz, num_motifs, num_motifs)
        local_edge = (dist <= self.local_window) & valid_pair
        prior_pair = valid_pair & (~local_edge) if self.prior_exclude_local else valid_pair
        mask_value = get_mask_fill_value(node)

        cosine_clean = torch.where(valid_pair, cosine, torch.zeros_like(cosine))

        nv_score, nv_valid_pair = self._pairwise_feature_score(motif_nv_features, motif_mask)
        if nv_score is None:
            nv_score = torch.zeros(bsz, num_motifs, num_motifs, device=device, dtype=dtype)
            nv_valid_pair = torch.zeros_like(valid_pair)
        nv_candidate_pair = prior_pair & nv_valid_pair
        nv_topk = nv_score.masked_fill(~nv_candidate_pair, mask_value)
        nv_edge = torch.zeros_like(local_edge)
        if self.topk_nv > 0:
            nv_edge = self._masked_topk(nv_topk, self.topk_nv, mask_value)
        nv_clean = torch.where(nv_edge, nv_score, torch.zeros_like(nv_score))

        edge_mask =nv_edge & valid_pair
        if self.training and self.edge_dropout > 0:
            drop = (torch.rand_like(cosine) < self.edge_dropout) & edge_mask
            edge_mask = edge_mask & (~drop)
        return {
            'edge_mask': edge_mask,
            'local_edge': local_edge,
            'nv_edge': nv_edge,
            'cosine': cosine_clean,
            'nv_score': nv_clean,
        }

    def forward(
        self,
        field_x: torch.Tensor,
        node: torch.Tensor,
        residue_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        motif_nv_features: Optional[torch.Tensor] = None,
    ):
        graph = self._build_graph(
            node=node,
            residue_mask=residue_mask,
            motif_mask=motif_mask,
            motif_nv_features=motif_nv_features,
        )
        node_raw = node
        h = node
        layer_aux = {}
        for i, layer in enumerate(self.layers):
            h, aux_i = layer(
                node=h,
                edge_mask=graph['edge_mask'],
                cosine=graph['cosine'],
                local_edge=graph['local_edge'],
                nv_score=graph['nv_score'],
                motif_mask=motif_mask,
            )
            layer_aux[f'layer_{i}_gate_mean'] = aux_i['gate_mean']
            layer_aux[f'layer_{i}_agg_norm'] = aux_i['agg_norm']

        field_x = self.field_inject(
            field_x=field_x,
            node_raw=node_raw,
            node_smooth=h,
            residue_mask=residue_mask,
            motif_mask=motif_mask,
            inject_scale=self.inject_scale,
        )
        smooth_delta = h - node_raw
        aux = {
            **graph,
            **layer_aux,
            'node_raw': node_raw,
            'node_smooth': h,
            'smooth_delta_norm': finite_or_zero(smooth_delta.norm(dim=-1)),
            'graph_degree': graph["edge_mask"].float().sum(dim=-1)
        }

        return field_x, h, aux
