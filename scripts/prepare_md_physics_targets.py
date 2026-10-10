#!/usr/bin/env python
"""Prepare MD physics pretraining targets from trusted descriptor columns."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.polymer_smiles import canonicalize_psmiles  # noqa: E402
from src.md.quality import positive_log_target  # noqa: E402


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


def _valid_positive(series: pd.Series) -> pd.Series:
    return series.notna() & np.isfinite(series.astype(float)) & (series.astype(float) > 0)


def build_targets(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    rows: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        record = canonicalize_psmiles(row.get("psmiles"), source="md_metrics")
        n_frames = row.get("n_frames_used") if pd.notna(row.get("n_frames_used")) else None
        trajectory_sufficient_for_dynamics = bool(n_frames is not None and n_frames >= 50)
        rg = row.get("Rg_mean") if pd.notna(row.get("Rg_mean")) else np.nan
        re = row.get("Re_mean") if pd.notna(row.get("Re_mean")) else np.nan
        persistence = row.get("persistence_length") if pd.notna(row.get("persistence_length")) else np.nan
        rg_tau = row.get("Rg_autocorrelation_time") if pd.notna(row.get("Rg_autocorrelation_time")) else np.nan
        torsion_rate = row.get("torsion_transition_rate") if pd.notna(row.get("torsion_transition_rate")) else np.nan
        rows.append(
            {
                "polymer_id": record.polymer_id,
                "canonical_psmiles": record.canonical_psmiles,
                "raw_psmiles": row.get("psmiles"),
                "Rg_mean": rg,
                "Rg_std": row.get("Rg_std") if pd.notna(row.get("Rg_std")) else np.nan,
                "Re_mean": re,
                "Re_std": row.get("Re_std") if pd.notna(row.get("Re_std")) else np.nan,
                "Re_over_Rg": float(re / rg) if pd.notna(rg) and pd.notna(re) and rg > 0 else np.nan,
                "relative_shape_anisotropy": row.get("relative_shape_anisotropy_mean")
                if pd.notna(row.get("relative_shape_anisotropy_mean"))
                else np.nan,
                "asphericity": row.get("asphericity_mean") if pd.notna(row.get("asphericity_mean")) else np.nan,
                "gyration_lambda1": row.get("lambda1_mean") if pd.notna(row.get("lambda1_mean")) else np.nan,
                "gyration_lambda2": row.get("lambda2_mean") if pd.notna(row.get("lambda2_mean")) else np.nan,
                "gyration_lambda3": row.get("lambda3_mean") if pd.notna(row.get("lambda3_mean")) else np.nan,
                "lambda1_over_lambda3": float(row.get("lambda1_mean") / row.get("lambda3_mean"))
                if pd.notna(row.get("lambda1_mean")) and pd.notna(row.get("lambda3_mean")) and row.get("lambda3_mean") > 0
                else np.nan,
                "persistence_length": persistence,
                "Kuhn_length": row.get("Kuhn_length") if pd.notna(row.get("Kuhn_length")) else np.nan,
                "Rg_autocorrelation_time": rg_tau,
                "torsion_transition_rate": torsion_rate,
                "log_Rg": positive_log_target(rg),
                "log_Re": positive_log_target(re),
                "log_persistence_length": positive_log_target(persistence),
                "log_Rg_autocorrelation_time": positive_log_target(rg_tau),
                "log_torsion_transition_rate": positive_log_target(torsion_rate),
                "Rg_converged": False,
                "Re_converged": False,
                "tau_Rg_valid": False,
                "tau_Re_valid": False,
                "tau_1_valid": False,
                "tau_2_valid": False,
                "tau_3_valid": False,
                "tau_5_valid": False,
                "n_frames_used": n_frames,
                "quality_flag": "needs_trajectory_qc"
                if trajectory_sufficient_for_dynamics
                else "insufficient_frames_or_missing_trajectory",
            }
        )
    return pd.DataFrame(rows)


def write_qc_report(targets: pd.DataFrame, report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MD Quality Control Report",
        "",
        "Generated from precomputed descriptors only; raw trajectories/topologies were not available in this checkout.",
        "",
        f"Rows: {len(targets):,}",
        f"Rows with Rg targets: {int(_valid_positive(targets['Rg_mean']).sum()):,}",
        f"Rows with Re targets: {int(_valid_positive(targets['Re_mean']).sum()):,}",
        f"Rows with persistence length: {int(_valid_positive(targets['persistence_length']).sum()):,}",
        f"Rows with torsion transition rate: {int(_valid_positive(targets['torsion_transition_rate']).sum()):,}",
        "",
        "No convergence flags are marked true because block/fraction trajectory data are unavailable.",
        "Rouse modes, intrachain contacts, internal-distance scaling, and Re autocorrelation are not reported without ordered backbone/topology and trajectory coordinates.",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "data/all_chain_metrics.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/md_physics_targets.parquet")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/md_quality_control.md")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(message)s")
    targets = build_targets(args.metrics)
    _write_table(targets, args.output)
    write_qc_report(targets, args.report)
    logging.info("rows=%d", len(targets))


if __name__ == "__main__":
    main()

