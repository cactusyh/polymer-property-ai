#!/usr/bin/env python3
"""Resume-safe batch runner for the map-aware 200-system MD production."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CANARY_IDS = {
    "poly_0382ff574025b3c6",
    "poly_dddca9e52ab732dc",
    "poly_087c04b9d9cdf6d7",
    "poly_d2ce25761366483b",
    "poly_5b2e2fc8f3913f8b",
}
REQUIRED_COLUMNS = [
    "polymer_id",
    "canonical_psmiles",
    "raw_psmiles",
    "priority_rank",
    "selection_reason",
    "target_band_gap_if_available",
    "motif_flags",
    "status",
    "attempt_count",
    "construction_strategy_used",
    "structure_success",
    "forcefield_success",
    "md_success",
    "mapping_success",
    "descriptor_success",
    "failure_reason",
    "output_system_dir",
    "production_time_ps",
    "dump_interval_ps",
    "n_frames",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def make_manifest(candidates: pd.DataFrame, output_root: Path, existing: pd.DataFrame | None) -> pd.DataFrame:
    existing_by_id = {}
    if existing is not None and not existing.empty:
        existing_by_id = {str(row.polymer_id): row._asdict() for row in existing.itertuples(index=False)}
    rows = []
    for row in candidates.sort_values("priority_rank").itertuples(index=False):
        polymer_id = str(row.polymer_id)
        prior = existing_by_id.get(polymer_id, {})
        system_dir = output_root / "systems" / polymer_id
        canary = polymer_id in CANARY_IDS and system_dir.exists()
        generated_status = system_dir / "metadata" / "generation_status.json"
        validation_status = system_dir / "metadata" / "trajectory_validation.json"
        if canary and generated_status.exists() and validation_status.exists():
            generated = json.loads(generated_status.read_text())
            validation = json.loads(validation_status.read_text())
            prior.update(
                {
                    "status": "CANARY_REUSED",
                    "attempt_count": int(generated.get("construction_attempts", prior.get("attempt_count", 0) or 0)),
                    "construction_strategy_used": generated.get("construction_strategy_used"),
                    "structure_success": bool(generated.get("structure_generated")),
                    "forcefield_success": bool(generated.get("force_field_success")),
                    "md_success": bool(generated.get("md_success")),
                    "mapping_success": bool(validation.get("mapping_success")),
                    "descriptor_success": True,
                    "failure_reason": None,
                    "output_system_dir": str(system_dir),
                    "production_time_ps": 100.0,
                    "dump_interval_ps": 0.5,
                    "n_frames": int(validation.get("n_frames", 201)),
                }
            )
        values = {
            "polymer_id": polymer_id,
            "canonical_psmiles": row.canonical_psmiles,
            "raw_psmiles": getattr(row, "raw_psmiles", None),
            "priority_rank": int(row.priority_rank),
            "selection_reason": getattr(row, "selection_reason", ""),
            "target_band_gap_if_available": getattr(row, "band_gap_value", None),
            "motif_flags": ";".join(
                name
                for name in ("aromatic", "conjugated_bond", "sp2", "hetero_adjacent", "conjugated_linker")
                if bool_value(getattr(row, f"motif_{name}"))
            ),
            "status": prior.get("status", "PENDING"),
            "attempt_count": int(prior.get("attempt_count", 0) or 0),
            "construction_strategy_used": prior.get("construction_strategy_used"),
            "structure_success": bool_value(prior.get("structure_success", False)),
            "forcefield_success": bool_value(prior.get("forcefield_success", False)),
            "md_success": bool_value(prior.get("md_success", False)),
            "mapping_success": bool_value(prior.get("mapping_success", False)),
            "descriptor_success": bool_value(prior.get("descriptor_success", False)),
            "failure_reason": prior.get("failure_reason"),
            "output_system_dir": prior.get("output_system_dir", str(system_dir)),
            "production_time_ps": prior.get("production_time_ps"),
            "dump_interval_ps": prior.get("dump_interval_ps"),
            "n_frames": prior.get("n_frames"),
            "dataset_label": "MAP_AWARE_MD_200_METHOD_DEVELOPMENT_SET",
            "canary_reused": canary,
            "batch_id": prior.get("batch_id", f"batch_{(int(row.priority_rank) - 1) // 10 + 1:02d}"),
            "updated_at": utc_now(),
        }
        values.update({key: prior[key] for key in prior if key not in values})
        rows.append(values)
    return pd.DataFrame(rows)


def successful_existing(row: pd.Series, root: Path) -> bool:
    polymer_id = str(row.polymer_id)
    if polymer_id in CANARY_IDS:
        return False
    system_dir = root / "systems" / polymer_id
    status_path = system_dir / "metadata" / "generation_status.json"
    validation_path = system_dir / "metadata" / "trajectory_validation.json"
    if not status_path.exists() or not validation_path.exists():
        return False
    status = json.loads(status_path.read_text())
    validation = json.loads(validation_path.read_text())
    return bool(status.get("md_success")) and bool(validation.get("mapping_success"))


def update_from_system(row: pd.Series, root: Path) -> dict:
    polymer_id = str(row.polymer_id)
    system_dir = root / "systems" / polymer_id
    status_path = system_dir / "metadata" / "generation_status.json"
    validation_path = system_dir / "metadata" / "trajectory_validation.json"
    result = row.to_dict()
    result["output_system_dir"] = str(system_dir)
    result["updated_at"] = utc_now()
    if not status_path.exists():
        result["status"] = "FAILED"
        result["failure_reason"] = "generation status missing"
        return result
    status = json.loads(status_path.read_text())
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
    result.update(
        {
            "status": "MD_COMPLETE" if status.get("md_success") else "FAILED",
            "attempt_count": int(status.get("construction_attempts", 0) or 0),
            "construction_strategy_used": status.get("construction_strategy_used"),
            "structure_success": bool(status.get("structure_generated")),
            "forcefield_success": bool(status.get("force_field_success")),
            "md_success": bool(status.get("md_success")),
            "mapping_success": bool(validation.get("mapping_success")),
            "descriptor_success": False,
            "failure_reason": validation.get("failure_reason") or status.get("failure_reason"),
        }
    )
    if result["md_success"] and result["mapping_success"]:
        metadata_path = system_dir / "metadata" / "system_metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        result["production_time_ps"] = float(metadata.get("production_time_ps", 100.0))
        result["dump_interval_ps"] = float(metadata.get("dump_every_steps", 500)) * float(metadata.get("timestep_fs", 1.0)) / 1000.0
        result["n_frames"] = validation.get("n_frames")
        result["status"] = "MAPPING_VALID"
        result["failure_reason"] = None
    return result


def save_manifest(manifest: pd.DataFrame, path: Path) -> None:
    for column in REQUIRED_COLUMNS:
        if column not in manifest:
            manifest[column] = None
    ordered = REQUIRED_COLUMNS + [column for column in manifest.columns if column not in REQUIRED_COLUMNS]
    manifest[ordered].sort_values("priority_rank").to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/map_aware_md_expansion_200_candidates.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/map_aware_md_200_production_manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_200"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-systems", type=int, default=200)
    parser.add_argument("--production-steps", type=int, default=100000)
    parser.add_argument("--dump-every", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--lammps", default="/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/lmp")
    parser.add_argument("--lammps-wrapper", default=None)
    parser.add_argument("--only-batch", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 10 or args.batch_size > 20:
        raise SystemExit("batch size must be between 10 and 20")
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "logs"
    inputs = logs / "batch_inputs"
    logs.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates).drop_duplicates("polymer_id").sort_values("priority_rank").head(args.max_systems).copy()
    if len(candidates) != min(args.max_systems, 200):
        raise SystemExit(f"expected up to 200 unique candidates, found {len(candidates)}")
    existing = pd.read_csv(args.manifest) if args.manifest.exists() else None
    manifest = make_manifest(candidates, args.output_root, existing)
    save_manifest(manifest, args.manifest)

    pending = manifest[~manifest.apply(lambda row: successful_existing(row, args.output_root), axis=1)].copy()
    pending = pending[~pending.canary_reused.fillna(False)]
    if args.only_batch is not None:
        pending = pending[pending.batch_id == f"batch_{args.only_batch:02d}"]
    batches = [pending.iloc[start : start + args.batch_size].copy() for start in range(0, len(pending), args.batch_size)]
    if not batches:
        print(json.dumps({"status": "nothing_pending", "candidate_count": len(candidates), "canary_reused": int(manifest.canary_reused.sum())}, indent=2))
        return
    generator = Path(__file__).resolve().with_name("generate_map_aware_single_chain_md.py")
    for index, batch in enumerate(batches, start=1):
        batch_id = str(batch.batch_id.iloc[0]) if batch.batch_id.nunique() == 1 else f"run_{index:03d}"
        batch_csv = inputs / f"{batch_id}.csv"
        batch.to_csv(batch_csv, index=False)
        log_path = logs / f"{batch_id}.log"
        started = time.time()
        command = [
            sys.executable,
            str(generator),
            "--pilot-csv",
            str(batch_csv),
            "--target-atoms",
            "300",
            "--equil-steps",
            "1000",
            "--production-steps",
            str(args.production_steps),
            "--dump-every",
            str(args.dump_every),
            "--timeout",
            str(args.timeout),
            "--output-root",
            str(args.output_root),
        ]
        if args.lammps_wrapper:
            command[command.index("--output-root"):command.index("--output-root")] = ["--lammps-wrapper", args.lammps_wrapper]
        else:
            command[command.index("--output-root"):command.index("--output-root")] = ["--lammps", args.lammps]
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"START {batch_id} {utc_now()}\nCOMMAND {' '.join(command)}\n")
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=handle, stderr=subprocess.STDOUT, text=True)
            elapsed = time.time() - started
            handle.write(f"END {batch_id} returncode={completed.returncode} elapsed_seconds={elapsed:.1f} {utc_now()}\n")
        manifest = pd.read_csv(args.manifest)
        updated = []
        for row in manifest.itertuples(index=False):
            if str(row.polymer_id) in set(batch.polymer_id.astype(str)):
                updated.append(update_from_system(pd.Series(row._asdict()), args.output_root))
            else:
                updated.append(row._asdict())
        manifest = pd.DataFrame(updated)
        save_manifest(manifest, args.manifest)
        batch_rows = manifest[manifest.polymer_id.isin(batch.polymer_id)]
        batch_summary = {
            "batch_id": batch_id,
            "systems_attempted": len(batch_rows),
            "systems_md_success": int(batch_rows.md_success.fillna(False).sum()),
            "systems_mapping_valid": int(batch_rows.mapping_success.fillna(False).sum()),
            "systems_failed": int((~batch_rows.md_success.fillna(False)).sum()),
            "construction_strategy_usage": batch_rows.construction_strategy_used.value_counts(dropna=False).to_dict(),
            "dominant_failure_mode": batch_rows.failure_reason.dropna().value_counts().index[0] if batch_rows.failure_reason.notna().any() else None,
            "elapsed_wall_seconds": elapsed,
            "storage_bytes": sum(path.stat().st_size for sid in batch_rows.polymer_id for path in (args.output_root / "systems" / str(sid)).rglob("*") if path.is_file()),
        }
        (logs / f"{batch_id}_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
        print(json.dumps(batch_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
