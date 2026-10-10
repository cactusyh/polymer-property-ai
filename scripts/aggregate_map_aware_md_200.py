#!/usr/bin/env python3
"""Aggregate validated canary and production systems without modifying trajectories."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


CANARY_IDS = {
    "poly_0382ff574025b3c6",
    "poly_dddca9e52ab732dc",
    "poly_087c04b9d9cdf6d7",
    "poly_d2ce25761366483b",
    "poly_5b2e2fc8f3913f8b",
}
PROJECT = Path(__file__).resolve().parents[1]


def make_staging(root: Path) -> Path:
    staging = root / "analysis_input_200"
    staging_systems = staging / "systems"
    staging_systems.mkdir(parents=True, exist_ok=True)
    for link in staging_systems.iterdir():
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
    for source_root in (root / "systems", root / "canary" / "systems"):
        if not source_root.exists():
            continue
        for system in source_root.iterdir():
            target = staging_systems / system.name
            if not target.exists():
                target.symlink_to(system, target_is_directory=True)
    reports = staging / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    statuses = []
    for system in sorted(staging_systems.iterdir()):
        status_path = system / "metadata" / "generation_status.json"
        validation_path = system / "metadata" / "trajectory_validation.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
            status["mapping_success"] = bool(validation.get("mapping_success"))
            status["trajectory_valid"] = bool(validation.get("trajectory_valid"))
            statuses.append(status)
    pd.DataFrame(statuses).to_csv(reports / "generation_status.csv", index=False)
    pd.DataFrame(
        [json.loads((system / "metadata/trajectory_validation.json").read_text()) for system in sorted(staging_systems.iterdir()) if (system / "metadata/trajectory_validation.json").exists()]
    ).to_parquet(reports / "validation_registry.parquet", index=False)
    return staging


def static_worker(system_path: str) -> dict:
    sys.path.insert(0, str(PROJECT / "scripts"))
    from extract_conjugation_aware_md_descriptors import extract_system

    return extract_system(Path(system_path))


def v2_worker(system_path: str) -> dict:
    sys.path.insert(0, str(PROJECT / "scripts"))
    import extract_conjugation_aware_md_descriptors_v2 as extractor

    extractor.ACTIVE_ROOT = Path(system_path).parents[1]
    row = pd.Series({"polymer_id": Path(system_path).name, "selection_group": "production_200"})
    return extractor.extract_system(row)


def run_workers(staging: Path, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    systems = [str(path) for path in sorted((staging / "systems").iterdir()) if path.is_dir()]
    context = mp.get_context("spawn")
    static_rows = []
    v2_rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        static_futures = {pool.submit(static_worker, path): path for path in systems}
        v2_futures = {pool.submit(v2_worker, path): path for path in systems}
        for future in as_completed(static_futures):
            path = static_futures[future]
            try:
                static_rows.append(future.result())
            except Exception as exc:
                static_rows.append({"polymer_id": Path(path).name, "descriptor_status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"})
        for future in as_completed(v2_futures):
            path = v2_futures[future]
            try:
                v2_rows.append(future.result())
            except Exception as exc:
                v2_rows.append({"polymer_id": Path(path).name, "descriptor_v2_status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(static_rows), pd.DataFrame(v2_rows)


def runtime_storage_rows(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples(index=False):
        system = Path(item.output_system_dir)
        if not system.exists():
            continue
        status_path = system / "metadata/generation_status.json"
        validation_path = system / "metadata/trajectory_validation.json"
        metadata_path = system / "metadata/system_metadata.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        validation = json.loads(validation_path.read_text()) if validation_path.exists() else {}
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        log_candidates = [system / "lammps/lammps.stdout.log", root / "logs/gpu_workers" / f"{item.polymer_id}.log"]
        runtime_s = np.nan
        for log_path in log_candidates:
            if log_path.exists():
                text = log_path.read_text(errors="ignore")
                matches = re.findall(r"Total wall time:\s*(\d+):(\d+):(\d+)", text)
                if matches:
                    h, m, s = map(int, matches[-1])
                    runtime_s = h * 3600 + m * 60 + s
                    break
                matches = re.findall(r"elapsed_seconds[= ]([0-9.]+)", text)
                if matches:
                    runtime_s = float(matches[-1])
                    break
        trajectory = system / "trajectory/trajectory.lammpstrj"
        total_bytes = sum(path.stat().st_size for path in system.rglob("*") if path.is_file())
        rows.append(
            {
                "polymer_id": item.polymer_id,
                "runtime_s": runtime_s,
                "trajectory_bytes": trajectory.stat().st_size if trajectory.exists() else 0,
                "total_bytes": total_bytes,
                "n_atoms": status.get("n_atoms"),
                "n_frames": validation.get("n_frames"),
                "production_time_ps": metadata.get("production_time_ps"),
                "dump_interval_ps": float(metadata.get("dump_every_steps", 500)) * float(metadata.get("timestep_fs", 1.0)) / 1000.0,
                "timestep_fs": metadata.get("timestep_fs"),
                "mapping_success": validation.get("mapping_success", False),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT / "data/processed/map_aware_md_200_production_manifest.csv")
    parser.add_argument("--output-root", type=Path, default=Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_200"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    staging = make_staging(args.output_root)
    static, v2 = run_workers(staging, args.workers)
    static = static.sort_values("polymer_id").drop_duplicates("polymer_id")
    v2 = v2.sort_values("polymer_id").drop_duplicates("polymer_id")
    static_path = args.output_root / "reports/map_aware_md_200_static_descriptors.parquet"
    v2_path = args.output_root / "reports/map_aware_md_200_v2_descriptors.parquet"
    static.to_parquet(static_path, index=False)
    v2.to_parquet(v2_path, index=False)
    manifest = pd.read_csv(args.manifest)
    runtime = runtime_storage_rows(args.output_root, manifest)
    runtime.to_parquet(args.output_root / "reports/map_aware_md_200_runtime_storage.parquet", index=False)
    static_ok = static.set_index("polymer_id").descriptor_status.eq("OK")
    v2_ok = v2.set_index("polymer_id").descriptor_v2_status.eq("OK")
    manifest = manifest.set_index("polymer_id")
    manifest["descriptor_success"] = [bool(static_ok.get(sid, False) and v2_ok.get(sid, False)) for sid in manifest.index]
    manifest["status"] = np.where(manifest["descriptor_success"], np.where(manifest["canary_reused"], "CANARY_REUSED", "COMPLETE"), manifest["status"])
    manifest.reset_index().to_csv(args.manifest, index=False)
    print(json.dumps({"systems": len(manifest), "static_success": int(static_ok.sum()), "v2_success": int(v2_ok.sum()), "full_descriptor_success": int(manifest.descriptor_success.sum()), "staging_root": str(staging)}, indent=2))


if __name__ == "__main__":
    main()
