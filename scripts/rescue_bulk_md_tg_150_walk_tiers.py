#!/usr/bin/env python3
"""Complete the validated full-DP walk tiers after the first tier timed out."""

import csv
import json
import multiprocessing as mp
import time

from build_bulk_md_tg_cohort_150 import ROOT, build_one


MANIFEST = ROOT / "data/processed/bulk_md_tg_cohort_150_split.csv"
SNAPSHOT = ROOT / "results/bulk_md_tg_150_build_registry_pre_rescue.csv"
FIRST = ROOT / "results/bulk_md_tg_150_construction_rescue.csv"
OUTPUT = ROOT / "results/bulk_md_tg_150_walk_tier_rescue.csv"
TIERS = [
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
    started = time.monotonic()
    result = build_one(type("BuildRow", (), row)(), construction_tiers=TIERS,
                       tier_timeout_seconds=1200)
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
    first = {row["polymer_id"]: row for row in rows(FIRST)}
    selected = [row for row in manifest if original[row["polymer_id"]]["failure_reason"].startswith(
        "RadonPyError:Reached maximum number of retrying polymerize_rw")]
    if len(selected) != 2 or not {row["polymer_id"] for row in selected}.issubset(first):
        raise RuntimeError("Two original walk failures must complete their first-tier attempts")
    completed = {row["polymer_id"]: row for row in rows(OUTPUT)} if OUTPUT.exists() else {}
    pending = [row for row in selected if row["polymer_id"] not in completed]
    order = {row["polymer_id"]: index for index, row in enumerate(manifest)}
    print(json.dumps({"walk_failures": 2, "completed": len(completed), "pending": len(pending)}), flush=True)
    if pending:
        with mp.Pool(len(pending)) as pool:
            for result in pool.imap_unordered(worker, pending):
                completed[result["polymer_id"]] = result
                save(completed, order)
                print("WALK_TIER_RESCUE_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
