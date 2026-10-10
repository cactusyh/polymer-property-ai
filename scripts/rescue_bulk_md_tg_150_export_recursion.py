#!/usr/bin/env python3
"""Repair the two confirmed RadonPy recursive molecule-ID export failures."""

import csv
import json
import multiprocessing as mp
import signal
import time

from build_bulk_md_tg_cohort_150 import ROOT, build_one


MANIFEST = ROOT / "data/processed/bulk_md_tg_cohort_150_split.csv"
SNAPSHOT = ROOT / "results/bulk_md_tg_150_build_registry_pre_rescue.csv"
CONSTRUCTION = ROOT / "results/bulk_md_tg_150_construction_rescue.csv"
OUTPUT = ROOT / "results/bulk_md_tg_150_export_recursion_rescue.csv"
FIELDS = ["polymer_id", "split", "retry_wall_time_seconds", "build_status",
          "forcefield_status", "export_status", "failure_reason", "repair"]


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def worker(row):
    def alarm(signum, frame):
        raise TimeoutError("iterative export repair exceeded 7200 s")
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(7200)
    started = time.monotonic()
    try:
        result = build_one(type("BuildRow", (), row)(), iterative_mol_id=True)
    finally:
        signal.alarm(0)
    return {"polymer_id": row["polymer_id"], "split": row["split"],
            "retry_wall_time_seconds": round(time.monotonic() - started, 3),
            "build_status": result["build_status"], "forcefield_status": result["forcefield_status"],
            "export_status": result["export_status"], "failure_reason": result["failure_reason"],
            "repair": "iterative molecule-ID connected-component traversal at export"}


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
    selected = [row for row in manifest if original[row["polymer_id"]]["failure_reason"].startswith("RecursionError:")]
    if len(selected) != 2:
        raise RuntimeError(f"Expected two original export recursion failures, found {len(selected)}")
    # The construction-tier job writes to the same system directories. Retry
    # each ID only after its first attempt is recorded; this can run in stages.
    first_attempts = {row["polymer_id"]: row for row in rows(CONSTRUCTION)}
    completed = {row["polymer_id"]: row for row in rows(OUTPUT)} if OUTPUT.exists() else {}
    order = {row["polymer_id"]: index for index, row in enumerate(manifest)}
    pending = [row for row in selected if row["polymer_id"] in first_attempts and row["polymer_id"] not in completed]
    print(json.dumps({"export_recursion_total": 2, "completed": len(completed),
                      "ready": len(pending), "awaiting_first_attempt": 2-len(completed)-len(pending)}), flush=True)
    if pending:
        with mp.Pool(len(pending)) as pool:
            for result in pool.imap_unordered(worker, pending):
                completed[result["polymer_id"]] = result
                save(completed, order)
                print("EXPORT_RECURSION_RESCUE_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
