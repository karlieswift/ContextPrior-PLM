from __future__ import annotations

from typing import Optional
import warnings

import torch
import torch.nn as nn

try:
    from transformers import EsmModel
except ImportError as exc:  # pragma: no cover - import is environment dependent
    EsmModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class ESMResidueFeatureExtractor(nn.Module):
    """Load a pretrained ESM model and return motif-aligned residue features."""

    @staticmethod
    def _has_missing_contact_regression_weights(missing_keys: list[str]) -> bool:
        targets = {
            "contact_head.regression.weight",
            "contact_head.regression.bias",
        }
        for key in missing_keys:
            if key in targets:
                return True
            if any(key.endswith(target) for target in targets):
                return True
        return False

    def __init__(
        self,
        model_name_or_path: str,
        *,
        local_files_only: bool = False,
        feature_layer: int = -1,
        tune_mode: str = "frozen",
        train_last_n_layers: int = 4,
    ):
        super().__init__()
        if EsmModel is None:
            raise ImportError(
                "transformers is required to use input_feature_source='esm_residue'."
            ) from _TRANSFORMERS_IMPORT_ERROR

        load_result = EsmModel.from_pretrained(
            model_name_or_path,
            add_pooling_layer=False,
            local_files_only=bool(local_files_only),
            output_loading_info=True,
        )
        loading_info = None
        if isinstance(load_result, tuple) and len(load_result) == 2:
            self.esm, loading_info = load_result
        else:
            self.esm = load_result

        missing_keys = []
        if isinstance(loading_info, dict):
            missing_keys = list(loading_info.get("missing_keys", []))
        self.contact_head_weights_loaded: Optional[bool]
        if loading_info is None:
            self.contact_head_weights_loaded = None
            warnings.warn(
                "Unable to verify whether ESM contact regression weights were loaded. "
                "Contact prediction quality may be unreliable if the checkpoint is incomplete.",
                stacklevel=2,
            )
        else:
            self.contact_head_weights_loaded = not self._has_missing_contact_regression_weights(missing_keys)
            if not self.contact_head_weights_loaded:
                warnings.warn(
                    "ESM contact regression weights were not loaded. "
                    "predict_contacts outputs will not be reliable.",
                    stacklevel=2,
                )
        self.hidden_size = int(self.esm.config.hidden_size)
        self.feature_layer = int(feature_layer)
        self.tune_mode = str(tune_mode)
        self.train_last_n_layers = int(train_last_n_layers)

        if self.tune_mode not in {"frozen", "last_n", "full"}:
            raise ValueError(f"Unsupported esm_tune_mode: {self.tune_mode}")
        if self.tune_mode == "last_n" and self.train_last_n_layers < 0:
            raise ValueError("esm_train_last_n_layers must be non-negative")

        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        for param in self.esm.parameters():
            param.requires_grad = False

        if self.tune_mode == "full":
            for param in self.esm.parameters():
                param.requires_grad = True
            return

        if self.tune_mode == "last_n":
            encoder_layers = getattr(self.esm.encoder, "layer", None)
            if encoder_layers is None:
                raise AttributeError("Loaded ESM model does not expose encoder.layer for partial finetuning.")

            total_layers = len(encoder_layers)
            start_idx = max(0, total_layers - self.train_last_n_layers)
            for layer in encoder_layers[start_idx:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tune_mode == "frozen":
            self.esm.eval()
        else:
            self.esm.train(mode)
        return self

    def _select_residue_states(self, outputs) -> torch.Tensor:
        if self.feature_layer == -1:
            return outputs.last_hidden_state

        hidden_states: Optional[tuple[torch.Tensor, ...]] = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("ESM hidden states were not returned; set output_hidden_states=True.")
        num_states = len(hidden_states)
        layer_idx = self.feature_layer
        if layer_idx < -num_states or layer_idx >= num_states:
            raise IndexError(
                f"esm_feature_layer={self.feature_layer} is out of range for {num_states} available hidden states."
            )
        return hidden_states[layer_idx]

    def _predict_contacts_from_outputs(
        self,
        outputs,
        seq_input_ids: torch.Tensor,
        seq_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attns = outputs.attentions
        if attns is None:
            raise RuntimeError("ESM attentions were not returned; set output_attentions=True.")
        if not hasattr(self.esm, "contact_head"):
            raise AttributeError("Loaded ESM model does not expose contact_head for contact prediction.")

        attns = torch.stack(attns, dim=1)
        attns = attns * seq_attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(attns.dtype)
        attns = attns * seq_attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4).to(attns.dtype)
        return self.esm.contact_head(seq_input_ids, attns)

    @staticmethod
    def _gather_motif_features(
        residue_states: torch.Tensor,
        motif_residue_index: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_positions = motif_residue_index + 1
        token_positions = token_positions.clamp(min=0, max=residue_states.size(1) - 1)
        gather_index = token_positions.unsqueeze(-1).expand(-1, -1, -1, residue_states.size(-1))
        expanded_states = residue_states.unsqueeze(1).expand(-1, motif_residue_index.size(1), -1, -1)
        motif_features = torch.gather(expanded_states, dim=2, index=gather_index)
        motif_features = motif_features * residue_mask.unsqueeze(-1).to(motif_features.dtype)
        return motif_features

    def forward(
        self,
        seq_input_ids: torch.Tensor,
        seq_attention_mask: torch.Tensor,
        motif_residue_index: torch.Tensor | None,
        residue_mask: torch.Tensor | None,
        *,
        return_contacts: bool = False,
        return_residue_states: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        model_kwargs = {
            "input_ids": seq_input_ids,
            "attention_mask": seq_attention_mask,
            "output_hidden_states": self.feature_layer != -1,
            "output_attentions": bool(return_contacts),
            "return_dict": True,
        }
        if self.tune_mode == "frozen":
            with torch.no_grad():
                outputs = self.esm(**model_kwargs)
        else:
            outputs = self.esm(**model_kwargs)

        residue_states = self._select_residue_states(outputs)
        motif_features = None
        if motif_residue_index is not None and residue_mask is not None:
            motif_features = self._gather_motif_features(residue_states, motif_residue_index, residue_mask)

        if not return_contacts and not return_residue_states and motif_features is not None:
            return motif_features

        out: dict[str, torch.Tensor] = {}
        if motif_features is not None:
            out["motif_features"] = motif_features
        if return_residue_states:
            residue_token_states = residue_states[:, 1:-1]
            residue_token_states = residue_token_states * seq_attention_mask[:, 1:-1].unsqueeze(-1).to(residue_token_states.dtype)
            out["residue_states"] = residue_token_states
            out["residue_sequence_mask"] = seq_attention_mask[:, 1:-1].to(torch.bool)

        if self.contact_head_weights_loaded is False:
            if return_contacts:
                raise RuntimeError(
                    "ESM contact_head.regression weights were not loaded, so residue contact predictions "
                    "are unreliable. Use a complete pretrained checkpoint or set return_contacts=False."
                )

        if return_contacts:
            contact_map = self._predict_contacts_from_outputs(
                outputs,
                seq_input_ids=seq_input_ids,
                seq_attention_mask=seq_attention_mask,
            )
            out["residue_contact_map"] = contact_map

        if motif_features is not None and not return_contacts and not return_residue_states:
            return motif_features
        return out
