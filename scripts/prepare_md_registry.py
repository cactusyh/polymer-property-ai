#!/usr/bin/env python
"""Create a unified MD metadata registry from available descriptor tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.polymer_smiles import canonicalize_psmiles  # noqa: E402


TARGET_COLUMNS = [
    "polymer_id",
    "canonical_psmiles",
    "trajectory_path",
    "topology_path",
    "temperature_K",
    "n_atoms",
    "n_frames",
    "trajectory_time_ps",
    "timestep_fs",
    "frame_interval_ps",
    "force_field",
    "ensemble",
    "degree_of_polymerization",
    "quality_flag",
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


def build_md_registry(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    rows: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        record = canonicalize_psmiles(row.get("psmiles"), source="md_metrics")
        has_descriptor = pd.notna(row.get("Rg_mean")) or pd.notna(row.get("Re_mean"))
        has_trajectory_ref = pd.notna(row.get("dump_file"))
        quality_parts = []
        if not record.valid:
            quality_parts.append("invalid_or_missing_psmiles")
        if not has_descriptor:
            quality_parts.append("missing_core_descriptors")
        if not has_trajectory_ref:
            quality_parts.append("trajectory_path_unavailable")
        rows.append(
            {
                "polymer_id": record.polymer_id,
                "canonical_psmiles": record.canonical_psmiles,
                "trajectory_path": row.get("dump_file") if has_trajectory_ref else None,
                "topology_path": row.get("data_file") if pd.notna(row.get("data_file")) else None,
                "temperature_K": row.get("temperature") if pd.notna(row.get("temperature")) else None,
                "n_atoms": row.get("total_atoms") if pd.notna(row.get("total_atoms")) else None,
                "n_frames": row.get("n_frames_used") if pd.notna(row.get("n_frames_used")) else None,
                "trajectory_time_ps": None,
                "timestep_fs": None,
                "frame_interval_ps": None,
                "force_field": row.get("force_field") if pd.notna(row.get("force_field")) else None,
                "ensemble": None,
                "degree_of_polymerization": row.get("degree_of_polymerization")
                if pd.notna(row.get("degree_of_polymerization"))
                else None,
                "quality_flag": ";".join(quality_parts) if quality_parts else "ok_from_descriptor_table",
            }
        )
    return pd.DataFrame(rows, columns=TARGET_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "data/all_chain_metrics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/md_registry.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(message)s")
    registry = build_md_registry(args.metrics)
    _write_table(registry, args.output)
    logging.info("rows=%d with_trajectory_refs=%d", len(registry), int(registry["trajectory_path"].notna().sum()))


if __name__ == "__main__":
    main()

