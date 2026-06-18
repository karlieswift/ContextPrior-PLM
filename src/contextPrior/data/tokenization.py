from __future__ import annotations

from dataclasses import dataclass


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
CANONICAL_WITH_X = set("ACDEFGHIKLMNPQRSTVWYX")


class ProteinTokenizer:
    def __init__(self):
        aas = list("ACDEFGHIKLMNPQRSTVWYBXZJUO")
        self.unk_token_id = 0
        self.pad_token_id = 1
        self.mask_token_id = 2
        self.stoi = {aa: idx + 4 for idx, aa in enumerate(aas)}
        self.vocab_size = len(self.stoi) + 4

    def encode_sequence(self, seq: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_token_id) for ch in seq.upper()]


@dataclass
class MotifWindowConfig:
    motif_len: int = 40
    motif_stride: int = 20
    max_motifs: int = 24
    add_last_window: bool = True


class MotifTokenizer:
    def __init__(self, tokenizer: ProteinTokenizer, cfg: MotifWindowConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def split_sequence_with_positions(self, seq: str) -> tuple[list[list[int]], list[list[int]]]:
        ids = self.tokenizer.encode_sequence(seq)
        n = len(ids)
        if n == 0:
            return [[]], [[]]

        windows: list[list[int]] = []
        positions: list[list[int]] = []
        start = 0
        while start < n:
            end = min(start + self.cfg.motif_len, n)
            windows.append(ids[start:end])
            positions.append(list(range(start, end)))
            if end >= n:
                break
            start += self.cfg.motif_stride

        if self.cfg.add_last_window and n > self.cfg.motif_len:
            last_start = max(0, n - self.cfg.motif_len)
            last_window = ids[last_start:n]
            last_positions = list(range(last_start, n))
            if len(windows) == 0 or windows[-1] != last_window:
                windows.append(last_window)
                positions.append(last_positions)

        return windows[: self.cfg.max_motifs], positions[: self.cfg.max_motifs]


def normalize_sequence_for_esm(seq: str, max_seq_len: int) -> str:
    seq = seq.strip().upper()
    seq = "".join(ch if ch in AMINO_ACIDS else "X" for ch in seq)
    seq = seq[:max_seq_len]
    return " ".join(seq)


def normalize_sequence_for_plm(seq: str, max_seq_len: int, sequence_format: str = "spaced_aa") -> str:

    sequence_format = str(sequence_format or "spaced_aa")
    seq = seq.strip().upper()
    seq = "".join(ch if ch in CANONICAL_WITH_X else "X" for ch in seq)
    seq = seq[:max_seq_len]
    if sequence_format == "raw_aa":
        return seq
    if sequence_format == "saprot_masked_structure":
        return "".join(f"{aa}#" for aa in seq)
    if sequence_format == "saprot_masked_structure_spaced":
        return " ".join(f"{aa}#" for aa in seq)
    if sequence_format == "spaced_aa":
        return " ".join(seq)
    raise ValueError(f"Unsupported PLM sequence_format: {sequence_format}")


def normalize_sequence_for_motif(seq: str) -> str:
    seq = seq.strip().upper()
    return "".join(ch if ch in AMINO_ACIDS else "X" for ch in seq)
