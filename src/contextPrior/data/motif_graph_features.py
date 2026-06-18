"""
@Env: /anaconda3/python3.11
@Time: 2026/3/18-9:22
@Auth: karlieswift
@File:
@Desc:
"""
from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch


_NV_ALPHABET = "LAGVSERTIDPKQNFYMHWCXBUZO"
_DIM_PROBE_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
_NV_BLOCKS = {
    "nv": False,
    "eev": False,
    "enhanced_nv635": True,
    "aa_entropy": False,
    "kmer_hashed": False,
    "kmer_entropy": False,
    "pair_lag": True,
    "pair_bucket": False,
    "pair_long_range": False,
    "biophys_moments": False,
}
_PHYS_BLOCKS = {
    "nv": False,
    "eev": False,
    "enhanced_nv635": False,
    "aa_entropy": False,
    "kmer_hashed": False,
    "kmer_entropy": False,
    "pair_lag": False,
    "pair_bucket": False,
    "pair_long_range": False,
    "biophys_moments": True,
}

# This is an early attempt, not our experiment.
# Physicochemical feature extraction is kept for downstream attribution figures.
# The final public model graph itself only consumes Natural Vector features.
_PAIR_LAGS = (1, 2, 3, 4, 8, 16)
_PAIR_BUCKETS = ((1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10**9))
_LONG_RANGE_THRESHOLDS = (0.2, 0.4)
_KMER_KS = (3,)
_KMER_DIM_PER_K = 1024
_KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
_SIMPLE_CHARGE = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}


def _feature_module_path() -> Path:
    workspace_root = Path(__file__).resolve().parents[4]
    return workspace_root / "motif_inference_stack_dual_input" / "biomapDatasets" / "protein_feature_blocks1.py"


@lru_cache(maxsize=1)
def _load_external_feature_module() -> Any | None:
    module_path = _feature_module_path()
    if not module_path.exists():
        return None

    spec = spec_from_file_location("motif_graph_external_features", module_path)
    if spec is None or spec.loader is None:
        return None

    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _build_external_extractors() -> tuple[Any | None, Any | None]:
    module = _load_external_feature_module()
    if module is None:
        return None, None

    nv_blocks = module.FeatureBlocks(**_NV_BLOCKS)
    phys_blocks = module.FeatureBlocks(**_PHYS_BLOCKS)
    nv_extractor = module.ProteinFeatureExtractor(module.ExtractorConfig(blocks=nv_blocks))
    phys_extractor = module.ProteinFeatureExtractor(module.ExtractorConfig(blocks=phys_blocks))
    return nv_extractor, phys_extractor


def _fallback_natural_vector(seq: str) -> np.ndarray:
    seq = (seq or "").upper()
    idx = {ch: i for i, ch in enumerate(_NV_ALPHABET)}
    counts = np.zeros((len(_NV_ALPHABET),), dtype=np.float32)
    sum_pos = np.zeros_like(counts)
    sum_pos2 = np.zeros_like(counts)
    valid_len = 0.0

    for pos, ch in enumerate(seq, start=1):
        j = idx.get(ch)
        if j is None:
            continue
        valid_len += 1.0
        counts[j] += 1.0
        sum_pos[j] += float(pos)
        sum_pos2[j] += float(pos * pos)

    if valid_len <= 0:
        return np.zeros((len(_NV_ALPHABET) * 3,), dtype=np.float32)

    mu = np.zeros_like(counts)
    d2 = np.zeros_like(counts)
    nonzero = counts > 0
    mu[nonzero] = sum_pos[nonzero] / counts[nonzero]
    sse = sum_pos2[nonzero] - (sum_pos[nonzero] * sum_pos[nonzero]) / counts[nonzero]
    d2[nonzero] = sse / (counts[nonzero] * valid_len)

    out = np.zeros((len(_NV_ALPHABET) * 3,), dtype=np.float32)
    out[0::3] = counts
    out[1::3] = mu
    out[2::3] = d2
    return out


def _expected_dim_from_blocks(blocks: dict[str, bool]) -> int:
    dim = 0
    if blocks.get("nv", False):
        dim += 75
    if blocks.get("enhanced_nv635", False):
        dim += 635
    if blocks.get("aa_entropy", False):
        dim += 8
    if blocks.get("kmer_hashed", False):
        dim += len(_KMER_KS) * _KMER_DIM_PER_K
    if blocks.get("kmer_entropy", False):
        dim += 4 * len(_KMER_KS)
    if blocks.get("pair_lag", False):
        dim += 64 * len(_PAIR_LAGS)
    if blocks.get("pair_bucket", False):
        dim += 64 * len(_PAIR_BUCKETS)
    if blocks.get("pair_long_range", False):
        dim += 64 * len(_LONG_RANGE_THRESHOLDS)
    if blocks.get("biophys_moments", False):
        dim += 10
    return dim


def _fallback_nv_features(seq: str) -> np.ndarray:
    seq = (seq or "").upper()
    parts: list[np.ndarray] = []

    if _NV_BLOCKS.get("nv", False):
        parts.append(_fallback_natural_vector(seq))
    if _NV_BLOCKS.get("enhanced_nv635", False):
        parts.append(np.zeros((635,), dtype=np.float32))
    if _NV_BLOCKS.get("aa_entropy", False):
        parts.append(np.zeros((8,), dtype=np.float32))
    if _NV_BLOCKS.get("kmer_hashed", False):
        parts.append(np.zeros((len(_KMER_KS) * _KMER_DIM_PER_K,), dtype=np.float32))
    if _NV_BLOCKS.get("kmer_entropy", False):
        parts.append(np.zeros((4 * len(_KMER_KS),), dtype=np.float32))
    if _NV_BLOCKS.get("pair_lag", False):
        parts.append(np.zeros((64 * len(_PAIR_LAGS),), dtype=np.float32))
    if _NV_BLOCKS.get("pair_bucket", False):
        parts.append(np.zeros((64 * len(_PAIR_BUCKETS),), dtype=np.float32))
    if _NV_BLOCKS.get("pair_long_range", False):
        parts.append(np.zeros((64 * len(_LONG_RANGE_THRESHOLDS),), dtype=np.float32))
    if _NV_BLOCKS.get("biophys_moments", False):
        parts.append(_fallback_physchem_features(seq))

    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _central_moments(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((5,), dtype=np.float32)
    total = float(np.sum(values))
    mean = float(np.mean(values))
    centered = values - mean
    var = float(np.mean(centered ** 2))
    m3 = float(np.mean(centered ** 3))
    m4 = float(np.mean(centered ** 4))
    return np.asarray([total, mean, var, m3, m4], dtype=np.float32)


def _fallback_physchem_features(seq: str) -> np.ndarray:
    seq = (seq or "").upper()
    hydro = np.asarray([_KYTE_DOOLITTLE.get(ch, 0.0) for ch in seq], dtype=np.float32)
    charge = np.asarray([_SIMPLE_CHARGE.get(ch, 0.0) for ch in seq], dtype=np.float32)
    return np.concatenate([_central_moments(hydro), _central_moments(charge)], axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def _external_feature_dims() -> tuple[int, int]:
    nv_extractor, phys_extractor = _build_external_extractors()

    nv_dim = _expected_dim_from_blocks(_NV_BLOCKS)
    if nv_extractor is not None:
        nv_dim = int(np.asarray(nv_extractor.extract(_DIM_PROBE_SEQUENCE), dtype=np.float32).shape[0])

    phys_dim = _expected_dim_from_blocks(_PHYS_BLOCKS)
    if phys_extractor is not None:
        phys_dim = int(np.asarray(phys_extractor.extract(_DIM_PROBE_SEQUENCE), dtype=np.float32).shape[0])

    return nv_dim, phys_dim


def extract_motif_nv(seq: str) -> np.ndarray:
    seq = (seq or "").upper()
    nv_extractor, _ = _build_external_extractors()
    if nv_extractor is not None:
        if not seq:
            nv_dim, _ = _external_feature_dims()
            return np.zeros((nv_dim,), dtype=np.float32)
        return np.asarray(nv_extractor.extract(seq), dtype=np.float32)
    return _fallback_nv_features(seq)


def extract_motif_physchem(seq: str) -> np.ndarray:
    seq = (seq or "").upper()
    _, phys_extractor = _build_external_extractors()
    if phys_extractor is not None:
        if not seq:
            _, phys_dim = _external_feature_dims()
            return np.zeros((phys_dim,), dtype=np.float32)
        return np.asarray(phys_extractor.extract(seq), dtype=np.float32)
    if not seq:
        return np.zeros((10,), dtype=np.float32)
    return _fallback_physchem_features(seq)


def build_motif_prior_feature_tensors(
    motif_sequences: Sequence[Sequence[str]],
    *,
    use_nv: bool,
    use_physchem: bool,
) -> dict[str, torch.Tensor]:
    if not use_nv and not use_physchem:
        return {}

    batch_size = len(motif_sequences)
    max_motifs = max((len(row) for row in motif_sequences), default=0)
    out: dict[str, torch.Tensor] = {}

    nv_tensor: torch.Tensor | None = None
    phys_tensor: torch.Tensor | None = None
    if use_nv:
        nv_dim = int(extract_motif_nv("").shape[0])
        nv_tensor = torch.zeros((batch_size, max_motifs, nv_dim), dtype=torch.float32)
    if use_physchem:
        phys_dim = int(extract_motif_physchem("").shape[0])
        phys_tensor = torch.zeros((batch_size, max_motifs, phys_dim), dtype=torch.float32)

    for batch_idx, row in enumerate(motif_sequences):
        for motif_idx, motif_seq in enumerate(row):
            if not motif_seq:
                continue
            if nv_tensor is not None:
                nv_tensor[batch_idx, motif_idx] = torch.from_numpy(extract_motif_nv(motif_seq))
            if phys_tensor is not None:
                phys_tensor[batch_idx, motif_idx] = torch.from_numpy(extract_motif_physchem(motif_seq))

    if nv_tensor is not None:
        out["motif_nv_features"] = nv_tensor
    if phys_tensor is not None:
        out["motif_physchem_features"] = phys_tensor
    return out


def build_motif_nv_feature_tensors(
    motif_sequences: Sequence[Sequence[str]],
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Build Natural Vector motif-window descriptors for final graph topology."""
    return build_motif_prior_feature_tensors(
        motif_sequences,
        use_nv=enabled,
        use_physchem=False,
    )
