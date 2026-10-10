#!/usr/bin/env python3
"""Trace frozen GAFF2_mod assignment failures without changing parameters."""

import contextlib
import csv
import json
import multiprocessing as mp
import os
import signal
import time
import traceback
from pathlib import Path

import numpy as np
from radonpy.core import poly, utils
from radonpy.ff.gaff2_mod import GAFF2_mod

from build_bulk_md_tg_cohort_150 import ROOT, radonpy_single_isomer_build_mode, set_lammps_styles


MANIFEST = ROOT / "data/processed/bulk_md_tg_cohort_150_split.csv"
SNAPSHOT = ROOT / "results/bulk_md_tg_150_build_registry_pre_rescue.csv"
OUTDIR = ROOT / "results/bulk_md_tg_150_gaff_diagnostics"
OUTPUT = ROOT / "results/bulk_md_tg_150_gaff_diagnostics.csv"
FIELDS = ["polymer_id", "split", "diagnostic_status", "wall_time_seconds",
          "failed_stage", "stage_results", "error_message", "log_path"]


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def worker(row):
    poly_id = row["polymer_id"]
    log_path = OUTDIR / f"{poly_id}.log"
    started = time.monotonic()
    stages = []
    status, error = "DIAGNOSTIC_INCOMPLETE", ""
    def alarm_handler(signum, frame):
        raise TimeoutError("GAFF diagnostic exceeded 2700 seconds")
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(2700)
    try:
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                np.random.seed(int(poly_id[-8:], 16) % (2**32 - 1))
                with radonpy_single_isomer_build_mode():
                    monomer = utils.mol_from_smiles(row["canonical_psmiles"])
                if monomer is None:
                    raise RuntimeError("mol_from_smiles returned None")
                chain = poly.polymerize_rw(monomer, int(row["selected_DP"]), tacticity="isotactic",
                                           retry=20, retry_step=300, opt="rdkit", dist_min=0.7)
                if chain is None:
                    raise RuntimeError("polymerize_rw returned None")
                set_lammps_styles(chain)
                ff = GAFF2_mod()
                for name in ("assign_ptypes", "assign_btypes", "assign_atypes", "assign_dtypes", "assign_itypes"):
                    original = getattr(ff, name)
                    def traced(molecule, _name=name, _original=original):
                        print(f"DIAGNOSTIC_STAGE_BEGIN {_name}", flush=True)
                        result = _original(molecule)
                        stages.append({"stage": _name, "success": bool(result)})
                        print(f"DIAGNOSTIC_STAGE_END {_name} {bool(result)}", flush=True)
                        return result
                    setattr(ff, name, traced)
                assigned = ff.ff_assign(chain)
                status = "ASSIGN_PASS_ON_REPLAY" if assigned else "ASSIGN_FAIL_REPRODUCED"
                print(f"DIAGNOSTIC_FINAL {status}", flush=True)
            except Exception as exc:
                status = "DIAGNOSTIC_EXCEPTION"
                error = f"{type(exc).__name__}:{exc}"
                traceback.print_exc()
    finally:
        signal.alarm(0)
    failed = [stage["stage"] for stage in stages if not stage["success"]]
    return {"polymer_id": poly_id, "split": row["split"],
            "diagnostic_status": status, "wall_time_seconds": round(time.monotonic() - started, 3),
            "failed_stage": ";".join(dict.fromkeys(failed)),
            "stage_results": json.dumps(stages), "error_message": error,
            "log_path": str(log_path.relative_to(ROOT))}


def save(completed, order):
    temp = OUTPUT.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(completed.values(), key=lambda row: order[row["polymer_id"]]))
    temp.replace(OUTPUT)


def main():
    manifest = rows(MANIFEST)
    original = {row["polymer_id"]: row for row in rows(SNAPSHOT)}
    selected = [row for row in manifest if original[row["polymer_id"]]["failure_reason"] ==
                "RuntimeError:GAFF2_mod ff_assign returned False"]
    if len(selected) != 22:
        raise RuntimeError(f"Expected 22 original GAFF2_mod failures, found {len(selected)}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    completed = {row["polymer_id"]: row for row in rows(OUTPUT)} if OUTPUT.exists() else {}
    pending = [row for row in selected if row["polymer_id"] not in completed]
    order = {row["polymer_id"]: index for index, row in enumerate(manifest)}
    print(json.dumps({"gaff_total": 22, "already_diagnosed": len(completed),
                      "pending": len(pending), "workers": min(4, len(pending))}), flush=True)
    if pending:
        with mp.Pool(min(4, len(pending))) as pool:
            for result in pool.imap_unordered(worker, pending):
                completed[result["polymer_id"]] = result
                save(completed, order)
                print("GAFF_DIAGNOSTIC_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
