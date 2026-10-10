#!/usr/bin/env python
"""Build a unified polymer identity registry from raw PSMILES CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.polymer_smiles import canonicalize_psmiles  # noqa: E402


DEFAULT_INPUTS = [
    ("pi1m", ROOT / "data/raw/pi1m/PI1M.csv", "SMILES"),
    ("md_metrics", ROOT / "data/all_chain_metrics.csv", "psmiles"),
    ("tg", ROOT / "data/raw/properties/tg/tg.csv", "SMILES"),
]


def _write_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(output_path, index=False)
        logging.info("wrote %s", output_path)
    except ImportError as exc:
        fallback = output_path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        logging.warning("parquet engine unavailable; wrote CSV fallback to %s", fallback)
        logging.warning("install pyarrow or fastparquet to create %s: %s", output_path, exc)


def _load_source(source: str, path: Path, smiles_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if smiles_column not in df.columns:
        raise ValueError(f"{path} does not contain column {smiles_column!r}")
    records = [canonicalize_psmiles(value, source=source).to_dict() for value in df[smiles_column]]
    out = pd.DataFrame.from_records(records)
    out["source_row"] = range(len(out))
    out["source_path"] = str(path)
    return out


def build_registry(inputs: list[tuple[str, Path, str]]) -> pd.DataFrame:
    registry = pd.concat([_load_source(*item) for item in inputs], ignore_index=True)
    registry["exact_duplicate"] = registry.duplicated(["source", "raw_psmiles"], keep=False)
    registry["canonical_duplicate"] = registry["canonical_psmiles"].notna() & registry.duplicated(
        ["canonical_psmiles"], keep=False
    )
    required = ["polymer_id", "raw_psmiles", "canonical_psmiles", "source", "valid", "error_message"]
    return registry[required + [col for col in registry.columns if col not in required]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/polymer_registry.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(message)s")
    existing_inputs = [item for item in DEFAULT_INPUTS if item[1].exists()]
    registry = build_registry(existing_inputs)
    _write_table(registry, args.output)
    logging.info("rows=%d valid=%d", len(registry), int(registry["valid"].sum()))


if __name__ == "__main__":
    main()

