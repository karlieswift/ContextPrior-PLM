from __future__ import annotations

import torch
import torch.nn as nn

from ..config import MotifInferenceConfig
from .graph import ContextualMotifGraphInference, MotifStateEncoder
from .layers import (
    AttentionPooling1D,
    AttentionPooling2D,
    MotifAxisAttention,
    MotifConvStageBlock,
    MotifFieldToSequence,
    MotifFieldQueryTokenizer,
    masked_mean_pool,
)
from .esm_residue_adapter import ESMResidueFeatureExtractor
from .modeling_esm import EsmConfig, EsmEncoder
from .motif_encoder import MotifAxialEncoder
from .motif_encoder_sdpa import MotifAxialEncoderSDPA

class MotifContextInferenceBlock(nn.Module):
    def __init__(self, cfg: MotifInferenceConfig):
        super().__init__()

        self.attn = nn.ModuleList([
            MotifAxisAttention(
                dim=cfg.hidden_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                attn_scale=cfg.attn_scale,
            )
            for _ in range(cfg.num_motif_axis_attention)
        ])
        #
        self.state_encoder = MotifStateEncoder(
            dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            use_stats=cfg.graph_use_stats,
        )
        self.graph = ContextualMotifGraphInference(
            dim=cfg.hidden_dim,
            num_layers=cfg.graph_num_layers_per_block,
            topk_nv=cfg.graph_topk_nv,
            local_window=cfg.graph_local_window,
            edge_dropout=cfg.graph_edge_dropout,
            inject_scale=cfg.graph_inject_scale,
            dropout=cfg.dropout,
            detach_topology=cfg.graph_detach_topology,
            smoothing_alpha=cfg.graph_smoothing_alpha,
            prior_exclude_local=bool(getattr(cfg, "graph_prior_exclude_local", True)),
        )

    def forward(
        self,
        field_x: torch.Tensor,
        residue_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        motif_nv_features: torch.Tensor | None = None,
    ):
        # field_x, motif_attn_score = self.attn(field_x, motif_mask)
        motif_attn_score = None
        for idx, block in enumerate(self.attn):
            field_x, motif_attn_score = block(field_x, motif_mask)

        motif_state, state_aux = self.state_encoder(field_x, residue_mask, motif_mask)



        field_x, motif_state, graph_aux = self.graph(
            field_x=field_x,
            node=motif_state,
            residue_mask=residue_mask,
            motif_mask=motif_mask,
            motif_nv_features=motif_nv_features,
        )
        aux = {
            'motif_state': motif_state,
            'motif_attn_score': motif_attn_score,
            **state_aux,
            **graph_aux,
        }
        return field_x, aux


class ProteinMotifInferenceBackbone(nn.Module):
    """Backbone following embed → (Attn + Graph)×K → Conv stage → pool."""
    def __init__(self, cfg: MotifInferenceConfig):
        super().__init__()
        self.cfg = cfg
        self.pad_token_id = cfg.pad_token_id
        self.input_feature_source = cfg.input_feature_source
        self.use_motif_esm_encoder = cfg.use_motif_esm_encoder
        self.use_motif_axial_encoder = cfg.use_motif_axial_encoder
        self.motif_encoder_impl = cfg.motif_encoder_impl
        if self.use_motif_axial_encoder and self.use_motif_esm_encoder:
            raise ValueError("use_motif_axial_encoder and use_motif_esm_encoder cannot both be True")
        if self.motif_encoder_impl not in {"legacy", "sdpa"}:
            raise ValueError(f"Unsupported motif_encoder_impl: {self.motif_encoder_impl}")
        if self.input_feature_source not in {"native_embedding", "esm_residue"}:
            raise ValueError(f"Unsupported input_feature_source: {self.input_feature_source}")

        self.token_embed = None
        self.input_proj = None
        self.esm_feature_extractor = None
        self.esm_input_proj = None
        self.esm_hidden_size = None
        if self.input_feature_source == "native_embedding":
            self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=cfg.pad_token_id)
            self.input_proj = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
        else:
            self.esm_feature_extractor = ESMResidueFeatureExtractor(
                cfg.esm_model_name_or_path,
                local_files_only=bool(cfg.esm_local_files_only),
                feature_layer=cfg.esm_feature_layer,
                tune_mode=cfg.esm_tune_mode,
                train_last_n_layers=cfg.esm_train_last_n_layers,
            )
            self.esm_hidden_size = self.esm_feature_extractor.hidden_size
            if cfg.esm_project_mode not in {"auto", "none", "linear"}:
                raise ValueError(f"Unsupported esm_project_mode: {cfg.esm_project_mode}")
            if cfg.esm_project_mode == "none":
                if self.esm_hidden_size != cfg.hidden_dim:
                    raise ValueError(
                        "esm_project_mode='none' requires esm hidden size to match hidden_dim, "
                        f"but got {self.esm_hidden_size} vs {cfg.hidden_dim}."
                    )
                self.esm_input_proj = nn.Identity()
            elif cfg.esm_project_mode == "auto" and self.esm_hidden_size == cfg.hidden_dim:
                self.esm_input_proj = nn.Identity()
            else:
                self.esm_input_proj = nn.Linear(self.esm_hidden_size, cfg.hidden_dim)
        self.input_dropout = nn.Dropout(cfg.dropout)

        self.inference_blocks = nn.ModuleList([
            MotifContextInferenceBlock(cfg) for _ in range(cfg.num_inference_blocks)
        ])
        self.conv_stage = nn.ModuleList([
            MotifConvStageBlock(cfg.hidden_dim, cfg.conv_kernel_size, cfg.dropout)
            for _ in range(cfg.num_conv_layers)
        ])

        self.use_attn_pool = cfg.pooling == 'attn'
        self.pool = AttentionPooling2D(cfg.hidden_dim) if self.use_attn_pool else None
        self.use_motif_attn_pool = cfg.motif_sequence_pooling == 'attn'
        self.field_to_motif = None
        self.query_field_to_motif = None
        if self.use_motif_axial_encoder:
            if self.motif_encoder_impl == "sdpa":
                self.motif_axial_encoder = MotifAxialEncoderSDPA(cfg)
            else:
                self.motif_axial_encoder = MotifAxialEncoder(cfg)
        else:
            self.motif_axial_encoder = None
        self.motif_input_norm = None
        self.motif_encoder = None
        self.motif_pool = None
        self.motif_pos_embed = None
        self.slot_pos_embed = None
        self.global_cls_token = None
        self.motif_field_tokenizer = cfg.motif_field_tokenizer
        self.num_motif_tokens_per_motif = 1
        if self.use_motif_esm_encoder:
            if cfg.hidden_dim % cfg.motif_esm_num_attention_heads != 0:
                raise ValueError(
                    f"hidden_dim={cfg.hidden_dim} must be divisible by "
                    f"motif_esm_num_attention_heads={cfg.motif_esm_num_attention_heads}"
                )
            if self.motif_field_tokenizer not in {'query', 'pool'}:
                raise ValueError(f"Unsupported motif_field_tokenizer: {self.motif_field_tokenizer}")
            if self.motif_field_tokenizer == 'query':
                summary_heads = cfg.motif_summary_num_heads or cfg.num_heads
                if cfg.hidden_dim % summary_heads != 0:
                    raise ValueError(
                        f"hidden_dim={cfg.hidden_dim} must be divisible by motif_summary_num_heads={summary_heads}"
                    )
                if cfg.motif_summary_queries <= 0:
                    raise ValueError("motif_summary_queries must be positive")
                self.num_motif_tokens_per_motif = cfg.motif_summary_queries
                self.query_field_to_motif = MotifFieldQueryTokenizer(
                    dim=cfg.hidden_dim,
                    num_queries=cfg.motif_summary_queries,
                    num_heads=summary_heads,
                    max_residues=cfg.motif_len,
                    dropout=cfg.dropout,
                    use_field_pos=cfg.motif_summary_use_field_pos,
                )
            else:
                self.field_to_motif = MotifFieldToSequence(cfg.hidden_dim, pooling=cfg.motif_field_pooling)
            self.motif_input_norm = nn.LayerNorm(cfg.hidden_dim)
            self.motif_pool = AttentionPooling1D(cfg.hidden_dim) if self.use_motif_attn_pool else None
            if cfg.motif_summary_use_motif_pos:
                self.motif_pos_embed = nn.Embedding(cfg.max_motifs, cfg.hidden_dim)
            if cfg.motif_summary_use_slot_pos:
                self.slot_pos_embed = nn.Embedding(self.num_motif_tokens_per_motif, cfg.hidden_dim)
            if cfg.motif_summary_use_global_cls:
                self.global_cls_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
            self.motif_encoder = EsmEncoder(self._build_motif_esm_config())
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def _build_motif_esm_config(self) -> EsmConfig:
        max_positions = self.cfg.motif_esm_max_position_embeddings
        if max_positions is None:
            seq_len = self.cfg.max_motifs * self.num_motif_tokens_per_motif
            if self.global_cls_token is not None:
                seq_len += 1
            max_positions = max(32, seq_len + 2)
        return EsmConfig(
            attention_probs_dropout_prob=self.cfg.motif_esm_attention_dropout,
            classifier_dropout=None,
            emb_layer_norm_before=False,
            esmfold_config=None,
            hidden_act="gelu",
            hidden_dropout_prob=self.cfg.motif_esm_hidden_dropout,
            hidden_size=self.cfg.hidden_dim,
            initializer_range=0.02,
            intermediate_size=self.cfg.hidden_dim * self.cfg.motif_esm_intermediate_mult,
            is_folding_model=False,
            layer_norm_eps=1e-5,
            mask_token_id=self.cfg.mask_token_id,
            max_position_embeddings=max_positions,
            model_type="esm",
            num_attention_heads=self.cfg.motif_esm_num_attention_heads,
            num_hidden_layers=self.cfg.motif_esm_num_hidden_layers,
            pad_token_id=self.pad_token_id,
            position_embedding_type=self.cfg.motif_esm_position_embedding_type,
            token_dropout=False,
            use_cache=False,
            vocab_list=None,
            vocab_size=2,
        )

    def _pool_protein(self, h_2d: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
        if self.use_attn_pool:
            return self.pool(h_2d, residue_mask)
        return masked_mean_pool(h_2d, residue_mask.unsqueeze(1), dim=(2, 3))

    def _pool_motif_sequence(self, h_3d: torch.Tensor, motif_mask: torch.Tensor) -> torch.Tensor:
        if self.use_motif_attn_pool:
            return self.motif_pool(h_3d, motif_mask)
        return masked_mean_pool(h_3d, motif_mask.unsqueeze(-1), dim=1)

    @staticmethod
    def _motif_evidence_reliability(
        motif_mask: torch.Tensor,
        residue_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:

        valid_motifs = motif_mask.sum(dim=1, keepdim=True).to(dtype)
        residue_evidence = residue_mask.to(dtype).sum(dim=(1, 2)).unsqueeze(-1)

        target_motifs = torch.sqrt(residue_evidence.clamp_min(1.0))
        support = (valid_motifs / target_motifs.clamp_min(1.0)).clamp(0.0, 1.0)
        return support.pow(2.0)

    def _tokenize_motif_field(
        self,
        motif_field_2d: torch.Tensor,
        residue_mask: torch.Tensor,
        motif_mask: torch.Tensor,
    ):
        if self.motif_field_tokenizer == 'query':
            token_grid = self.query_field_to_motif(motif_field_2d, residue_mask, motif_mask)
        else:
            token_grid = self.field_to_motif(motif_field_2d, residue_mask).unsqueeze(2)

        bsz, num_motifs, num_slots, dim = token_grid.shape
        token_grid = self.motif_input_norm(token_grid)

        if self.motif_pos_embed is not None:
            motif_pos = self.motif_pos_embed(torch.arange(num_motifs, device=token_grid.device))
            token_grid = token_grid + motif_pos.view(1, num_motifs, 1, dim)
        if self.slot_pos_embed is not None:
            slot_pos = self.slot_pos_embed(torch.arange(num_slots, device=token_grid.device))
            token_grid = token_grid + slot_pos.view(1, 1, num_slots, dim)

        token_grid = token_grid * motif_mask.unsqueeze(-1).unsqueeze(-1).to(token_grid.dtype)
        token_seq = token_grid.view(bsz, num_motifs * num_slots, dim)
        token_mask = motif_mask.unsqueeze(-1).expand(bsz, num_motifs, num_slots).contiguous()
        token_mask = token_mask.view(bsz, num_motifs * num_slots)
        return token_grid, token_seq, token_mask

    @staticmethod
    def _safe_motif_mask(motif_mask: torch.Tensor) -> torch.Tensor:
        safe_mask = motif_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if empty.any():
            safe_mask[empty, 0] = True
        return safe_mask

    @staticmethod
    def _build_encoder_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        mask_value = -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
        return (1.0 - attention_mask[:, None, None, :].to(dtype)) * mask_value

    def _build_input_field(
        self,
        x: torch.Tensor | None,
        residue_mask: torch.Tensor | None,
        seq_input_ids: torch.Tensor | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        motif_residue_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aux: dict[str, torch.Tensor] = {}
        if self.input_feature_source == "native_embedding":
            if x is None:
                raise ValueError("Native embedding input requires motif token ids 'x'.")
            field_x = self.token_embed(x)
            field_x = self.input_proj(field_x)
        else:
            if seq_input_ids is None or seq_attention_mask is None:
                raise ValueError(
                    "ESM residue input requires seq_input_ids and seq_attention_mask."
                )
            if motif_residue_index is None or residue_mask is None:
                raise ValueError(
                    "Fixed-window ESM residue input requires motif_residue_index and residue_mask."
                )
            extractor_out = self.esm_feature_extractor(
                seq_input_ids=seq_input_ids,
                seq_attention_mask=seq_attention_mask,
                motif_residue_index=motif_residue_index,
                residue_mask=residue_mask,
                return_contacts=False,
                return_residue_states=False,
            )
            if isinstance(extractor_out, dict):
                field_x = extractor_out["motif_features"]
            else:
                field_x = extractor_out
            field_x = self.esm_input_proj(field_x)
        return self.input_dropout(field_x), aux

    def forward(
        self,
        x: torch.Tensor | None,
        residue_mask: torch.Tensor | None = None,
        motif_mask: torch.Tensor | None = None,
        *,
        seq_input_ids: torch.Tensor | None = None,
        seq_attention_mask: torch.Tensor | None = None,
        motif_residue_index: torch.Tensor | None = None,
        motif_nv_features: torch.Tensor | None = None,
    ):
        if residue_mask is None:
            if x is None:
                raise ValueError("residue_mask is required when x is None.")
            residue_mask = x.ne(self.pad_token_id)
        if motif_mask is None:
            motif_mask = residue_mask.any(dim=-1)

        field_x, input_aux = self._build_input_field(
            x,
            residue_mask,
            seq_input_ids=seq_input_ids,
            seq_attention_mask=seq_attention_mask,
            motif_residue_index=motif_residue_index,
        )
        if "residue_mask" in input_aux:
            residue_mask = input_aux["residue_mask"]
        if "motif_mask" in input_aux:
            motif_mask = input_aux["motif_mask"]
        if "motif_residue_index" in input_aux:
            motif_residue_index = input_aux["motif_residue_index"]
        if residue_mask is None or motif_mask is None:
            raise RuntimeError("Backbone requires residue_mask and motif_mask after building the input field.")

        input_field_2d = field_x.permute(0, 3, 1, 2).contiguous()
        input_protein_feat = self._pool_protein(input_field_2d, residue_mask)

        block_outputs = []
        final_aux = {}
        for idx, block in enumerate(self.inference_blocks):
            field_x, aux = block(
                field_x,
                residue_mask,
                motif_mask,
                motif_nv_features=motif_nv_features,
            )
            block_outputs.append(aux)
            final_aux = aux

        motif_field_2d = field_x.permute(0, 3, 1, 2).contiguous()  # [B,D,T,F]
        for conv in self.conv_stage:
            motif_field_2d = conv(motif_field_2d, residue_mask)

        stem_feat_2d = motif_field_2d
        motif_feat_4d = motif_field_2d.permute(0, 2, 3, 1).contiguous()
        motif_encoder_outputs = None
        if self.use_motif_axial_encoder:
            motif_encoder_outputs = self.motif_axial_encoder(
                hidden_states=motif_feat_4d,
                residue_mask=residue_mask,
                motif_mask=motif_mask,
                return_dict=True,
            )
            motif_feat_4d = motif_encoder_outputs.last_hidden_state
            motif_field_2d = motif_feat_4d.permute(0, 3, 1, 2).contiguous()

        motif_feat_3d = None
        motif_token_input = None
        motif_token_grid = None
        motif_global_token = None
        motif_sequence_mask = None
        if self.use_motif_esm_encoder:
            motif_token_grid, motif_token_input, motif_sequence_mask = self._tokenize_motif_field(
                motif_field_2d, residue_mask, motif_mask
            )

            encoder_input = motif_token_input
            encoder_mask = motif_sequence_mask
            if self.global_cls_token is not None:
                batch_size = residue_mask.size(0)
                cls = self.global_cls_token.expand(batch_size, -1, -1)
                encoder_input = torch.cat([cls, encoder_input], dim=1)
                cls_mask = torch.ones(batch_size, 1, device=residue_mask.device, dtype=torch.bool)
                encoder_mask = torch.cat([cls_mask, encoder_mask], dim=1)

            safe_motif_mask = self._safe_motif_mask(encoder_mask)
            encoder_outputs = self.motif_encoder(
                hidden_states=encoder_input,
                attention_mask=self._build_encoder_attention_mask(safe_motif_mask, motif_token_input.dtype),
                return_dict=True,
            )
            encoded_sequence = encoder_outputs.last_hidden_state
            if self.global_cls_token is not None:
                motif_global_token = encoded_sequence[:, 0]
                motif_feat_3d = encoded_sequence[:, 1:]
                protein_feat = motif_global_token
            else:
                motif_feat_3d = encoded_sequence
                protein_feat = self._pool_motif_sequence(motif_feat_3d, motif_sequence_mask)

            motif_feat_3d = motif_feat_3d * motif_sequence_mask.unsqueeze(-1).to(motif_feat_3d.dtype)
            motif_token_grid = motif_feat_3d.reshape(
                residue_mask.size(0), motif_mask.size(1), self.num_motif_tokens_per_motif, self.cfg.hidden_dim
            )
        else:
            protein_feat = self._pool_protein(motif_field_2d, residue_mask)
        protein_feat = self.out_norm(protein_feat)

        motif_reliability = self._motif_evidence_reliability(
            motif_mask=motif_mask,
            residue_mask=residue_mask,
            dtype=protein_feat.dtype,
        )
        input_protein_feat = self.out_norm(input_protein_feat)
        protein_feat = input_protein_feat + motif_reliability * (protein_feat - input_protein_feat)

        return {
            'protein_feat': protein_feat,
            'stem_feat_2d': stem_feat_2d,
            'motif_feat_2d': motif_field_2d,
            'motif_feat_4d': motif_feat_4d,
            'motif_feat_3d': motif_feat_3d,
            'motif_token_grid': motif_token_grid,
            'motif_token_input': motif_token_input,
            'motif_global_token': motif_global_token,
            'motif_sequence_mask': motif_sequence_mask,
            'motif_encoder_outputs': motif_encoder_outputs,
            'input_feature_source': self.input_feature_source,
            'esm_hidden_size': self.esm_hidden_size,
            'local_field': field_x,
            'motif_state': final_aux.get('motif_state'),
            'motif_attn_score': final_aux.get('motif_attn_score'),
            'block_outputs': block_outputs,
            **input_aux,
            **final_aux,
        }


# Backward-compatible alias
MotifTabModelBackbone = ProteinMotifInferenceBackbone
