from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MotifInferenceConfig:
    # token /
    vocab_size: int = 28
    pad_token_id: int = 1
    motif_len: int = 40
    max_motifs: int = 24
    motif_stride: int = 20

    # input stem
    input_feature_source: str = "native_embedding"  # ['native_embedding', 'esm_residue']
    esm_model_name_or_path: str = "facebook/esm2_t6_8M_UR50D"
    esm_tokenizer_name_or_path: Optional[str] = None
    esm_local_files_only: bool = False
    esm_max_seq_len: int = 1022
    esm_feature_layer: int = -1  # -1: final encoder output, 0: embedding output
    esm_project_mode: str = "auto"  # ['auto', 'none', 'linear']
    esm_tune_mode: str = "frozen"  # ['frozen', 'last_n', 'full']
    esm_train_last_n_layers: int = 4

    # embedding / hidden sizes
    embed_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.2

    # attention + local/NV prior graph block
    num_heads: int = 4
    num_inference_blocks: int = 1 #
    num_motif_axis_attention: int = 2
    attn_scale: float = 0.1
    graph_use_stats: bool = True
    graph_num_layers_per_block: int = 1
    graph_local_window: int = 2
    graph_topk_nv: int = 2
    graph_edge_dropout: float = 0.1
    graph_inject_scale: float = 0.3
    graph_detach_topology: bool = True
    graph_smoothing_alpha: float = 0.1
    graph_prior_exclude_local: bool = True

    # conv stage
    num_conv_layers: int = 2
    conv_kernel_size: Tuple[int, int] = (3, 3)

    # motif-sequence encoder after conv
    use_motif_esm_encoder: bool = False
    motif_field_tokenizer: str = 'query'  # ['query', 'pool']
    motif_field_pooling: str = 'attn'
    motif_summary_queries: int = 4
    motif_summary_num_heads: Optional[int] = None
    motif_summary_use_field_pos: bool = True
    motif_summary_use_global_cls: bool = True
    motif_summary_use_motif_pos: bool = True
    motif_summary_use_slot_pos: bool = True
    motif_sequence_pooling: str = 'attn'
    motif_esm_num_hidden_layers: int = 2
    motif_esm_num_attention_heads: int = 4
    motif_esm_intermediate_mult: int = 4
    motif_esm_hidden_dropout: float = 0.1
    motif_esm_attention_dropout: float = 0.1
    motif_esm_max_position_embeddings: Optional[int] = None
    motif_esm_position_embedding_type: str = 'rotary'

    # 4D motif encoder after stem
    use_motif_axial_encoder: bool = False
    motif_encoder_num_hidden_layers: int = 2
    motif_encoder_num_attention_heads: int = 4
    motif_encoder_intermediate_mult: int = 4
    motif_encoder_dropout: float = 0.0
    motif_encoder_attention_dropout: float = 0.0
    motif_encoder_norm_eps: float = 1e-6
    motif_encoder_use_rope: bool = True
    motif_encoder_attn_bias: bool = False
    motif_encoder_ffn_bias: bool = False
    motif_encoder_impl: str = "legacy"  # ['legacy', 'sdpa']


    pooling: str = 'attn'
    classifier_dropout: float = 0.2


    mask_token_id: int = 2
    min_seq_len: int = 20
