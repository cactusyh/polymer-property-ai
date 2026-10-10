#!/usr/bin/env python3
"""Use fixed full-DP RadonPy construction tiers for four original walk failures."""

import csv
import json
import multiprocessing as mp
import signal
import time
from pathlib import Path

from build_bulk_md_tg_cohort_150 import ROOT, build_one


MANIFEST = ROOT / "data/processed/bulk_md_tg_cohort_150_split.csv"
SNAPSHOT = ROOT / "results/bulk_md_tg_150_build_registry_pre_rescue.csv"
OUTPUT = ROOT / "results/bulk_md_tg_150_construction_rescue.csv"
TIERS = [
    ("original_single_chain", {"retry": 1, "retry_step": 20, "opt": None, "dist_min": 0.7}),
    ("robust_random_walk", {"retry": 20, "retry_step": 200, "opt": "rdkit", "dist_min": 0.7}),
    ("relaxed_overlap_random_walk", {"retry": 40, "retry_step": 300, "opt": "rdkit", "dist_min": 0.5}),
    ("extended_initialization", {"retry": 60, "retry_step": 500, "opt": "rdkit", "dist_min": 0.45}),
]
FIELDS = ["polymer_id", "split", "retry_wall_time_seconds", "build_status",
          "construction_tier", "construction_attempts", "forcefield_status",
          "export_status", "failure_reason"]


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def worker(row):
    def alarm(signum, frame):
        raise TimeoutError("tiered construction exceeded 7200 s wall-clock limit")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(7200)
    started = time.monotonic()
    try:
        result = build_one(type("BuildRow", (), row)(), construction_tiers=TIERS)
    finally:
        signal.alarm(0)
    return {"polymer_id": row["polymer_id"], "split": row["split"],
            "retry_wall_time_seconds": round(time.monotonic() - started, 3),
            "build_status": result["build_status"],
            "construction_tier": result.get("construction_tier", ""),
            "construction_attempts": json.dumps(result.get("construction_attempts", [])),
            "forcefield_status": result["forcefield_status"],
            "export_status": result["export_status"], "failure_reason": result["failure_reason"]}


def save(completed, order):
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(completed.values(), key=lambda row: order[row["polymer_id"]]))
    temporary.replace(OUTPUT)


def main():
    manifest = rows(MANIFEST)
    original = {row["polymer_id"]: row for row in rows(SNAPSHOT)}
    selected = [row for row in manifest if original[row["polymer_id"]]["failure_reason"].startswith(
                ("RecursionError:", "RadonPyError:Reached maximum number of retrying polymerize_rw"))]
    if len(selected) != 4:
        raise RuntimeError(f"Expected 4 recursion/walk failures; found {len(selected)}")
    completed = {row["polymer_id"]: row for row in rows(OUTPUT)} if OUTPUT.exists() else {}
    order = {row["polymer_id"]: index for index, row in enumerate(manifest)}
    pending = [row for row in selected if row["polymer_id"] not in completed]
    print(json.dumps({"original_failures": 4, "completed": len(completed), "pending": len(pending)}), flush=True)
    if pending:
        with mp.Pool(min(4, len(pending))) as pool:
            for result in pool.imap_unordered(worker, pending):
                completed[result["polymer_id"]] = result
                save(completed, order)
                print("CONSTRUCTION_RESCUE_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
