#!/usr/bin/env python3
"""Retry the locked cohort's original 30-minute build timeouts at 120 minutes.

This intentionally calls the original build worker without changing its chemistry,
seed, chain count, DP, construction algorithm, force field, packing, or QC.
No molecular dynamics is run.
"""

import csv
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

from build_bulk_md_tg_cohort_150 import ROOT, build_worker


MANIFEST = ROOT / "data/processed/bulk_md_tg_cohort_150_split.csv"
ORIGINAL = ROOT / "data/processed/bulk_md_tg_150_build_registry.csv"
SNAPSHOT = ROOT / "results/bulk_md_tg_150_build_registry_pre_rescue.csv"
OUTPUT = ROOT / "results/bulk_md_tg_150_timeout_retry.csv"
FIELDS = [
    "polymer_id", "split", "original_timeout_minutes", "retry_timeout_minutes",
    "retry_wall_time_seconds", "actual_completion_time_seconds", "build_status",
    "retry_class", "forcefield_status", "export_status", "failure_reason",
]


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def worker(row):
    os.environ["BULK_BUILD_TIMEOUT_SECONDS"] = "7200"
    start = time.monotonic()
    result = build_worker(row)
    elapsed = time.monotonic() - start
    reason = result.get("failure_reason", "") or ""
    if result["build_status"] == "PASS":
        retry_class = "BUILD_PASS_AFTER_TIMEOUT_RETRY"
    elif reason.startswith("BuildTimeout:"):
        retry_class = "BUILD_TIMEOUT_120MIN"
    else:
        retry_class = reason.split(":", 1)[0] or "BUILD_FAILURE_OTHER"
    return {
        "polymer_id": row["polymer_id"], "split": row["split"],
        "original_timeout_minutes": 30, "retry_timeout_minutes": 120,
        "retry_wall_time_seconds": round(elapsed, 3),
        "actual_completion_time_seconds": round(elapsed, 3) if result["build_status"] == "PASS" else "",
        "build_status": result["build_status"], "retry_class": retry_class,
        "forcefield_status": result["forcefield_status"],
        "export_status": result["export_status"], "failure_reason": reason,
    }


def save(rows, order):
    temp = OUTPUT.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: order[row["polymer_id"]]))
    temp.replace(OUTPUT)


def main():
    manifest = load_csv(MANIFEST)
    source = load_csv(ORIGINAL)
    if len(manifest) != 150 or len(source) != 150 or len({r["polymer_id"] for r in manifest}) != 150:
        raise RuntimeError("Locked 150-system manifest or build registry is invalid")
    by_id = {r["polymer_id"]: r for r in source}
    if set(by_id) != {r["polymer_id"] for r in manifest}:
        raise RuntimeError("Manifest identities differ from original build registry")
    timeouts = [r for r in manifest if by_id[r["polymer_id"]]["failure_reason"] ==
                "BuildTimeout:build exceeded 1800 s wall-clock limit"]
    if len(timeouts) != 27:
        raise RuntimeError(f"Expected exactly 27 original 30-minute timeouts, found {len(timeouts)}")
    if not SNAPSHOT.exists():
        shutil.copy2(ORIGINAL, SNAPSHOT)
    if OUTPUT.exists():
        previous = load_csv(OUTPUT)
        completed = {r["polymer_id"]: r for r in previous}
    else:
        completed = {}
    valid_ids = {r["polymer_id"] for r in timeouts}
    if not set(completed).issubset(valid_ids):
        raise RuntimeError("Retry file contains IDs outside the timeout group")
    pending = [r for r in timeouts if r["polymer_id"] not in completed]
    order = {r["polymer_id"]: i for i, r in enumerate(manifest)}
    print(json.dumps({"timeout_total": 27, "already_attempted": len(completed),
                      "pending": len(pending), "workers": min(8, len(pending))}), flush=True)
    if pending:
        with mp.Pool(min(8, len(pending))) as pool:
            for result in pool.imap_unordered(worker, pending):
                completed[result["polymer_id"]] = result
                save(completed, order)
                print("TIMEOUT_RETRY_DONE", json.dumps(result, ensure_ascii=False), flush=True)
    print("TIMEOUT_RETRY_FINAL", json.dumps({
        "attempted": len(completed),
        "rescued": sum(r["retry_class"] == "BUILD_PASS_AFTER_TIMEOUT_RETRY" for r in completed.values()),
        "still_timed_out": sum(r["retry_class"] == "BUILD_TIMEOUT_120MIN" for r in completed.values()),
    }), flush=True)


if __name__ == "__main__":
    main()
