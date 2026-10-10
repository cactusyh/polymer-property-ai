#!/usr/bin/env python3
"""Resume-safe independent-process GPU production queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


CANARY_IDS = {
    "poly_0382ff574025b3c6",
    "poly_dddca9e52ab732dc",
    "poly_087c04b9d9cdf6d7",
    "poly_d2ce25761366483b",
    "poly_5b2e2fc8f3913f8b",
}


def is_valid_existing(row: pd.Series, root: Path) -> bool:
    polymer_id = str(row.polymer_id)
    if polymer_id in CANARY_IDS:
        return False
    status_path = root / "systems" / polymer_id / "metadata/generation_status.json"
    validation_path = root / "systems" / polymer_id / "metadata/trajectory_validation.json"
    if not status_path.exists() or not validation_path.exists():
        return False
    status = json.loads(status_path.read_text())
    validation = json.loads(validation_path.read_text())
    return bool(status.get("md_success")) and bool(validation.get("mapping_success"))


def run_one(row: dict, args: argparse.Namespace) -> dict:
    polymer_id = str(row["polymer_id"])
    batch_id = str(row.get("batch_id", "batch_gpu"))
    batch_dir = args.output_root / "logs" / "batch_inputs" / "gpu_workers"
    batch_dir.mkdir(parents=True, exist_ok=True)
    row_csv = batch_dir / f"{polymer_id}.csv"
    pd.DataFrame([row]).to_csv(row_csv, index=False)
    log_path = args.output_root / "logs" / "gpu_workers" / f"{polymer_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("run_map_aware_md_single_system.py")),
        "--row-csv",
        str(row_csv),
        "--output-root",
        str(args.output_root),
        "--production-steps",
        str(args.production_steps),
        "--dump-every",
        str(args.dump_every),
        "--equil-steps",
        "1000",
        "--target-atoms",
        str(args.target_atoms),
        "--timeout",
        str(args.timeout),
        "--box-padding",
        "15.0",
        "--lammps-wrapper",
        str(args.lammps_wrapper),
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"START {polymer_id} {time.time()} batch={batch_id}\n")
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], stdout=handle, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - started
        handle.write(f"END {polymer_id} returncode={completed.returncode} elapsed_seconds={elapsed:.1f}\n")
    status_path = args.output_root / "systems" / polymer_id / "metadata/generation_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"failure_reason": "status missing"}
    status["queue_elapsed_seconds"] = elapsed
    status["batch_id"] = batch_id
    status["gpu_backend"] = True
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/map_aware_md_200_production_manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_200"))
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--production-steps", type=int, default=100000)
    parser.add_argument("--dump-every", type=int, default=500)
    parser.add_argument("--target-atoms", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--lammps-wrapper", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.concurrency < 1 or args.concurrency > 5:
        raise SystemExit("concurrency must be 1-5")
    manifest = pd.read_csv(args.manifest)
    pending = manifest[~manifest.apply(lambda row: is_valid_existing(row, args.output_root), axis=1)]
    pending = pending[~pending.polymer_id.isin(CANARY_IDS)]
    if args.limit is not None:
        pending = pending.head(args.limit)
    rows = pending.to_dict("records")
    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, row, args) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"polymer_id": result.get("polymer_id"), "md_success": result.get("md_success"), "strategy": result.get("construction_strategy_used"), "elapsed_seconds": result.get("queue_elapsed_seconds")}), flush=True)
    summary = {
        "concurrency": args.concurrency,
        "systems_attempted": len(rows),
        "systems_md_success": int(sum(bool(item.get("md_success")) for item in results)),
        "systems_failed": int(sum(not bool(item.get("md_success")) for item in results)),
        "elapsed_wall_seconds": time.time() - started,
        "systems_per_hour": len(rows) / max(time.time() - started, 1e-9) * 3600,
        "mean_worker_seconds": sum(float(item.get("queue_elapsed_seconds", 0)) for item in results) / max(len(results), 1),
        "failure_reasons": pd.Series([item.get("failure_reason") for item in results]).dropna().value_counts().to_dict(),
    }
    out = args.output_root / "logs" / f"gpu_queue_concurrency_{args.concurrency}_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
