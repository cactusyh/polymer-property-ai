"""Utilities for robust polymer PSMILES preprocessing.

The functions in this module never mutate raw input strings and never silently
drop invalid molecules. RDKit is used when available; otherwise a conservative
syntax-level fallback records that chemistry-level validation was not possible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Optional


try:  # pragma: no cover - exercised only when RDKit is installed
    from rdkit import Chem
except Exception:  # pragma: no cover - deterministic fallback in this env
    Chem = None  # type: ignore[assignment]


_BRACKETED_STAR = re.compile(r"\[(\*)\]")


@dataclass(frozen=True)
class PolymerSmilesRecord:
    """Canonicalization result for one raw PSMILES string."""

    polymer_id: str
    raw_psmiles: object
    canonical_psmiles: Optional[str]
    source: str
    valid: bool
    error_message: Optional[str]
    n_connection_atoms: int
    canonicalization_method: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_connection_atoms(psmiles: str) -> str:
    """Normalize bracketed wildcard connection atoms without changing raw data."""

    return _BRACKETED_STAR.sub("*", psmiles.strip())


def count_connection_atoms(psmiles: str) -> int:
    """Count polymer connection atoms represented as wildcard stars."""

    return normalize_connection_atoms(psmiles).count("*")


def stable_polymer_id(canonical_or_raw: str) -> str:
    """Return a stable short SHA-256 identifier for canonical polymer identity."""

    digest = hashlib.sha256(canonical_or_raw.encode("utf-8")).hexdigest()[:16]
    return f"poly_{digest}"


def _fallback_validate(psmiles: str) -> tuple[bool, Optional[str]]:
    if not psmiles.strip():
        return False, "empty PSMILES"
    if count_connection_atoms(psmiles) != 2:
        return False, "expected exactly two polymer connection atoms '*'"
    balance_pairs = [("(", ")"), ("[", "]")]
    for left, right in balance_pairs:
        if psmiles.count(left) != psmiles.count(right):
            return False, f"unbalanced {left}{right} delimiters"
    return True, None


def canonicalize_psmiles(raw_psmiles: object, source: str = "unknown") -> PolymerSmilesRecord:
    """Validate and canonicalize a PSMILES string where possible.

    RDKit canonicalization handles chemically equivalent atom ordering and most
    reverse-orientation duplicates. Without RDKit, the fallback only normalizes
    whitespace and ``[*]`` markers, and records that limitation in the method.
    """

    if raw_psmiles is None:
        key = f"INVALID::{source}::None"
        return PolymerSmilesRecord(
            polymer_id=stable_polymer_id(key),
            raw_psmiles=raw_psmiles,
            canonical_psmiles=None,
            source=source,
            valid=False,
            error_message="missing PSMILES",
            n_connection_atoms=0,
            canonicalization_method="none",
        )

    raw_text = str(raw_psmiles)
    normalized = normalize_connection_atoms(raw_text)
    n_stars = count_connection_atoms(normalized)

    if Chem is not None:  # pragma: no cover - depends on optional RDKit
        mol = Chem.MolFromSmiles(normalized)
        if mol is None:
            key = f"INVALID::{source}::{raw_text}"
            return PolymerSmilesRecord(
                polymer_id=stable_polymer_id(key),
                raw_psmiles=raw_psmiles,
                canonical_psmiles=None,
                source=source,
                valid=False,
                error_message="RDKit failed to parse PSMILES",
                n_connection_atoms=n_stars,
                canonicalization_method="rdkit",
            )
        if n_stars != 2:
            key = f"INVALID::{source}::{raw_text}"
            return PolymerSmilesRecord(
                polymer_id=stable_polymer_id(key),
                raw_psmiles=raw_psmiles,
                canonical_psmiles=None,
                source=source,
                valid=False,
                error_message="expected exactly two polymer connection atoms '*'",
                n_connection_atoms=n_stars,
                canonicalization_method="rdkit",
            )
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return PolymerSmilesRecord(
            polymer_id=stable_polymer_id(canonical),
            raw_psmiles=raw_psmiles,
            canonical_psmiles=canonical,
            source=source,
            valid=True,
            error_message=None,
            n_connection_atoms=n_stars,
            canonicalization_method="rdkit",
        )

    valid, error = _fallback_validate(normalized)
    canonical = normalized if valid else None
    key = canonical if valid else f"INVALID::{source}::{raw_text}"
    return PolymerSmilesRecord(
        polymer_id=stable_polymer_id(key),
        raw_psmiles=raw_psmiles,
        canonical_psmiles=canonical,
        source=source,
        valid=valid,
        error_message=error or "RDKit unavailable; chemistry-level parsing not performed",
        n_connection_atoms=n_stars,
        canonicalization_method="fallback_syntax",
    )


def equivalent_repeat_augmentations(psmiles: str, max_repeats: int = 3) -> list[str]:
    """Generate simple repeat-number augmentations for linear ``*AB*`` units.

    This conservative implementation only augments strings with exactly two
    terminal wildcard markers and no dot-disconnected fragments. It leaves
    chemistry validation to ``canonicalize_psmiles``.
    """

    normalized = normalize_connection_atoms(psmiles)
    if max_repeats < 2 or normalized.count("*") != 2:
        return []
    if not (normalized.startswith("*") and normalized.endswith("*")):
        return []
    repeat_unit = normalized[1:-1]
    if not repeat_unit or "." in repeat_unit:
        return []
    return [f"*{repeat_unit * repeat_count}*" for repeat_count in range(2, max_repeats + 1)]
