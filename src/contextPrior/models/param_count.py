from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


def summarize_parameter_groups(model: nn.Module) -> OrderedDict[str, int]:
    summary: OrderedDict[str, int] = OrderedDict()
    summary["total_params"] = sum(p.numel() for p in model.parameters())
    summary["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary["frozen_params"] = summary["total_params"] - summary["trainable_params"]

    groups = {
        "esm_backbone": 0,
        "projection_or_input": 0,
        "structured_adapter": 0,
        "motif_encoder": 0,
        "head": 0,
        "other": 0,
    }
    for name, param in model.named_parameters():
        if name.startswith("esm.") or name.startswith("model.esm") or ".esm." in name or "esm_feature_extractor.esm" in name:
            groups["esm_backbone"] += param.numel()
        elif name.startswith("head"):
            groups["head"] += param.numel()
        elif "motif_axial_encoder" in name:
            groups["motif_encoder"] += param.numel()
        elif any(key in name for key in ["token_embed", "input_proj", "esm_input_proj"]):
            groups["projection_or_input"] += param.numel()
        elif any(
            key in name
            for key in [
                "inference_blocks",
                "conv_stage",
                "blocks",
                "cross_blocks",
                "window_cross_blocks",
                "motif_blocks",
                "film_blocks",
                "query_tokens",
                "window_query",
                "pool",
                "out_norm",
            ]
        ):
            groups["structured_adapter"] += param.numel()
        else:
            groups["other"] += param.numel()

    for key, value in groups.items():
        summary[key] = value
    return summary
