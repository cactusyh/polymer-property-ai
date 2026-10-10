#!/usr/bin/env python3
"""Reconcile structure-only rescue attempts with the locked cohort registry."""

import csv
import json
import shutil
import statistics
from pathlib import Path

import pandas as pd

from build_bulk_md_tg_cohort_150 import ROOT, lammps_data_qc, write_report
from audit_bulk_md_tg_150_build_failures import failure_class


MANIFEST = ROOT / "data/processed/bulk_md_tg_cohort_150_split.csv"
SNAPSHOT = ROOT / "results/bulk_md_tg_150_build_registry_pre_rescue.csv"
REGISTRY = ROOT / "data/processed/bulk_md_tg_150_build_registry.csv"
RETRY = ROOT / "results/bulk_md_tg_150_timeout_retry.csv"
CONSTRUCTION = ROOT / "results/bulk_md_tg_150_construction_rescue.csv"
WALK_TIERS = ROOT / "results/bulk_md_tg_150_walk_tier_rescue.csv"
EXPORT_RECURSION = ROOT / "results/bulk_md_tg_150_export_recursion_rescue.csv"
GAFF = ROOT / "results/bulk_md_tg_150_gaff_diagnostics.csv"
OUTROOT = ROOT / "bulk_md_tg_150"
BUILD_REPORT = ROOT / "reports/bulk_md_tg_150_build_qc.md"
BUILD_REPORT_PRE_RESCUE = ROOT / "reports/bulk_md_tg_150_build_qc_pre_rescue.md"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(path, records, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def main():
    manifest = rows(MANIFEST)
    original_rows = rows(SNAPSHOT)
    original = {row["polymer_id"]: row for row in original_rows}
    timeout_rows = rows(RETRY)
    construction_rows = rows(CONSTRUCTION)
    walk_rows = rows(WALK_TIERS)
    export_rows = rows(EXPORT_RECURSION)
    diagnostic_rows = rows(GAFF)
    if len(manifest) != 150 or len(original) != 150:
        raise RuntimeError("Original fixed cohort is incomplete")
    if len(timeout_rows) != 27 or len(construction_rows) != 4 or len(walk_rows) != 2 or \
            len(export_rows) != 2 or len(diagnostic_rows) != 22:
        raise RuntimeError(f"Rescue/diagnostics incomplete: timeout={len(timeout_rows)}, "
                           f"construction={len(construction_rows)}, walk={len(walk_rows)}, "
                           f"export={len(export_rows)}, GAFF={len(diagnostic_rows)}")
    if len({row["polymer_id"] for row in manifest}) != 150 or \
            {row["polymer_id"] for row in manifest} != set(original):
        raise RuntimeError("Polymer identities changed")
    selected = {row["polymer_id"]: row for row in timeout_rows + construction_rows}
    for row in walk_rows:
        if row["polymer_id"] not in selected or not original[row["polymer_id"]]["failure_reason"].startswith(
                "RadonPyError:Reached maximum number of retrying polymerize_rw"):
            raise RuntimeError("Walk-tier group differs from original failures")
        selected[row["polymer_id"]] = row
    for row in export_rows:
        if row["polymer_id"] not in selected or not original[row["polymer_id"]]["failure_reason"].startswith("RecursionError:"):
            raise RuntimeError("Export recursion repair group differs from original failures")
        selected[row["polymer_id"]] = row
    if len(selected) != 31:
        raise RuntimeError("Rescue groups overlap")
    for row in timeout_rows:
        reason = row["failure_reason"]
        if row["build_status"] == "PASS":
            retry_class = "BUILD_PASS_AFTER_TIMEOUT_RETRY"
        elif reason.startswith("BuildTimeout:build exceeded 7200 s"):
            retry_class = "BUILD_TIMEOUT_120MIN"
        else:
            retry_class = failure_class(reason)
        row["retry_class"] = retry_class
    save(RETRY, timeout_rows, list(timeout_rows[0]))
    final_rows = []
    for manifest_row in manifest:
        pid = manifest_row["polymer_id"]
        record = original[pid].copy()
        if record["canonical_psmiles"] != manifest_row["canonical_psmiles"] or \
                record["split"] != manifest_row["split"] or \
                int(float(record["DP"])) != int(manifest_row["selected_DP"]) or \
                int(float(record["number_of_chains"])) != int(manifest_row["selected_chain_count"]):
            raise RuntimeError(f"Frozen identity or construction selection changed: {pid}")
        if pid in selected:
            attempt = selected[pid]
            metadata_path = OUTROOT / pid / "build_metadata.json"
            if not metadata_path.exists():
                raise RuntimeError(f"Missing rescue metadata: {pid}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for key in ("polymer_id", "canonical_psmiles", "split"):
                if metadata.get(key) != record[key]:
                    raise RuntimeError(f"Rescue changed frozen {key}: {pid}")
            for key in ("DP", "number_of_chains", "chemical_cluster_id", "estimated_total_atoms"):
                if int(metadata[key]) != int(float(record[key])):
                    raise RuntimeError(f"Rescue changed frozen {key}: {pid}")
            if metadata["build_status"] != attempt["build_status"]:
                raise RuntimeError(f"Rescue metadata/status disagreement: {pid}")
            for key in record:
                if key in metadata:
                    value = metadata[key]
                    record[key] = "" if value is None else value
            if record["build_status"] == "PASS":
                record["traceback"] = ""
                data_path = OUTROOT / pid / "structure.data"
                if not data_path.exists():
                    raise RuntimeError(f"Missing rescued structure.data: {pid}")
                actual_atoms = int(record["actual_total_atoms"])
                chains = int(record["number_of_chains"])
                errors, counts, molecules = lammps_data_qc(data_path, actual_atoms, chains)
                if errors or counts.get("atoms") != actual_atoms or molecules != chains:
                    raise RuntimeError(f"Rescued topology QC failed {pid}: {errors}")
                if record["forcefield_status"] != "PASS" or record["export_status"] != "PASS":
                    raise RuntimeError(f"Rescued forcefield/export QC failed: {pid}")
            else:
                record["actual_total_atoms"] = ""
                record["forcefield_status"] = metadata.get("forcefield_status", "NOT_RUN")
                record["export_status"] = metadata.get("export_status", "NOT_RUN")
                record["failure_reason"] = metadata.get("failure_reason", "")
                if str(record["failure_reason"]).startswith("RuntimeError:GAFF2_mod ff_assign returned False"):
                    record["forcefield_status"] = "FAIL"
                elif str(record["failure_reason"]).startswith(("RuntimeError:forcefield_qc_failed",
                                                               "RuntimeError:cell_forcefield_qc_failed")):
                    record["forcefield_status"] = "FAIL"
                elif str(record["failure_reason"]).startswith(("RecursionError:", "RuntimeError:LAMMPS data export failed",
                                                               "RuntimeError:export_qc_failed")):
                    record["forcefield_status"] = "PASS"
                    record["export_status"] = "FAIL"
        final_rows.append(record)
    if len(final_rows) != 150:
        raise RuntimeError("Final registry must contain exactly 150 polymers")
    save(REGISTRY, final_rows, list(original_rows[0]))
    if BUILD_REPORT.exists() and not BUILD_REPORT_PRE_RESCUE.exists():
        shutil.copy2(BUILD_REPORT, BUILD_REPORT_PRE_RESCUE)
    write_report(pd.read_csv(REGISTRY))
    successful = [float(row["actual_completion_time_seconds"]) for row in timeout_rows
                  if row["retry_class"] == "BUILD_PASS_AFTER_TIMEOUT_RETRY"]
    print(json.dumps({
        "timeout_attempted": len(timeout_rows),
        "timeout_rescued": len(successful),
        "timeout_120min": sum(row["retry_class"] == "BUILD_TIMEOUT_120MIN" for row in timeout_rows),
        "successful_build_time_median_seconds": statistics.median(successful) if successful else None,
        "successful_build_time_max_seconds": max(successful) if successful else None,
        "total_build_pass": sum(row["build_status"] == "PASS" for row in final_rows),
        "development_build_pass": sum(row["build_status"] == "PASS" and row["split"] == "DEVELOPMENT" for row in final_rows),
        "locked_holdout_build_pass": sum(row["build_status"] == "PASS" and row["split"] == "LOCKED_HOLDOUT" for row in final_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
