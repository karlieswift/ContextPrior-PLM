from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_SINGLE_MUT_RE = re.compile(r"(?P<wt>[A-Z])(?P<pos>\d+)(?P<mut>[A-Z])")


@dataclass(frozen=True)
class Mutation:
    wt: str
    pos: int  # 1-indexed
    mut: str

    @property
    def zero_index(self) -> int:
        return self.pos - 1

    def label(self) -> str:
        return f"{self.wt}{self.pos}{self.mut}"


def parse_mutation_string(text: str | None) -> list[Mutation]:
    """Parse strings such as A12V, A12V/G38D, A12V:G38D, or p.A12V."""
    if not text:
        return []
    text = str(text).strip()
    if not text or text.lower() in {"wt", "wildtype", "wild-type", "none", "nan"}:
        return []
    return [Mutation(m.group("wt"), int(m.group("pos")), m.group("mut")) for m in _SINGLE_MUT_RE.finditer(text)]


def apply_mutations(wt_sequence: str, mutations: Iterable[Mutation], *, strict: bool = False) -> str:
    seq = list(str(wt_sequence).strip().upper())
    for mut in mutations:
        idx = mut.zero_index
        if idx < 0 or idx >= len(seq):
            if strict:
                raise ValueError(f"Mutation {mut.label()} out of bounds for sequence length {len(seq)}")
            continue
        if strict and seq[idx] != mut.wt:
            raise ValueError(f"Mutation {mut.label()} expects {mut.wt} at {mut.pos}, found {seq[idx]}")
        seq[idx] = mut.mut
    return "".join(seq)


def infer_mutations_from_pair(wt_sequence: str, mutant_sequence: str) -> list[Mutation]:
    wt = str(wt_sequence).strip().upper()
    mt = str(mutant_sequence).strip().upper()
    muts: list[Mutation] = []
    if len(wt) != len(mt):
        return muts
    for i, (a, b) in enumerate(zip(wt, mt), start=1):
        if a != b:
            muts.append(Mutation(a, i, b))
    return muts


def enumerate_single_mutants(sequence: str, alphabet: str = AMINO_ACIDS) -> list[dict[str, str | int]]:
    """Return rows for every non-wild-type single substitution."""
    seq = str(sequence).strip().upper()
    rows: list[dict[str, str | int]] = []
    for pos0, wt in enumerate(seq):
        if wt not in alphabet:
            continue
        for aa in alphabet:
            if aa == wt:
                continue
            mutant = seq[:pos0] + aa + seq[pos0 + 1:]
            rows.append({
                "sequence": mutant,
                "wt_sequence": seq,
                "mutation": f"{wt}{pos0 + 1}{aa}",
                "position": pos0 + 1,
                "wt_aa": wt,
                "mut_aa": aa,
            })
    return rows


def mutation_count(text: str | None, wt_sequence: str | None = None, mutant_sequence: str | None = None) -> int:
    muts = parse_mutation_string(text)
    if muts:
        return len(muts)
    if wt_sequence and mutant_sequence and len(wt_sequence) == len(mutant_sequence):
        return sum(a != b for a, b in zip(wt_sequence, mutant_sequence))
    return 0
