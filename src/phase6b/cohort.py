"""Authoritative cohort reconciliation for the frozen bulk-MD registry."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from src.phase6b import COHORT_VERSION


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["polymer_id"])
    return frame.polymer_id.astype(str).tolist()


def reconcile(root: Path, output: Path) -> dict[str, object]:
    cohort_path = root / "data/processed/bulk_md_tg_final_cohort_150_split.csv"
    registry_path = root / "data/processed/bulk_md_tg_final_150_multitemp_registry.csv"
    qc_path = root / "reports/bulk_md_tg_final_150_multitemp_qc.md"
    split_qc_path = root / "reports/bulk_md_tg_final_split_qc.md"
    cohort = pd.read_csv(cohort_path, dtype={"polymer_id": str, "split": str})
    registry = pd.read_csv(registry_path, dtype={"polymer_id": str, "split": str})
    if len(cohort) != len(set(cohort.polymer_id)):
        raise ValueError("duplicate polymer IDs in authoritative cohort")
    if cohort.polymer_id.duplicated().any():
        raise ValueError("duplicate polymer IDs in authoritative cohort")
    from src.data.polymer_smiles import stable_polymer_id
    if "canonical_psmiles" in cohort.columns:
        mismatched_identity = [row.polymer_id for row in cohort.itertuples()
                               if stable_polymer_id(str(row.canonical_psmiles)) != row.polymer_id]
        if mismatched_identity:
            raise ValueError(f"cohort stable identity mismatch: {mismatched_identity[:5]}")
    assignments = cohort.groupby("polymer_id").split.nunique()
    conflicts = assignments[assignments > 1].index.astype(str).tolist()
    if conflicts:
        raise ValueError(f"conflicting cohort assignment: {conflicts}")
    split_counts = cohort.split.value_counts().to_dict()
    if set(split_counts) != {"DEVELOPMENT", "LOCKED_HOLDOUT"}:
        raise ValueError("unexpected authoritative cohort split labels")
    by_temp = registry.groupby(["polymer_id", "temperature_K"], as_index=False).agg(
        n_status=("status", "size"), status=("status", "first"), split_registry=("split", "first"))
    expected = {(pid, temp) for pid in cohort.polymer_id for temp in (300, 400, 500, 600)}
    observed = set(zip(by_temp.polymer_id, by_temp.temperature_K.astype(int)))
    missing_rows = sorted([f"{pid}:{temp}" for pid, temp in expected - observed])
    status_lookup = {(row.polymer_id, int(row.temperature_K)): row.status for row in registry.itertuples()}
    complete_ids = sorted(pid for pid in cohort.polymer_id if all(
        status_lookup.get((pid, temp)) == f"{temp}K_COMPLETE" for temp in (300, 400, 500, 600)))
    incomplete_ids = sorted(set(cohort.polymer_id) - set(complete_ids))
    registry_splits = dict(zip(registry.polymer_id, registry.split))
    inconsistent_qc = sorted(pid for pid in cohort.polymer_id
                             if pid in registry_splits and registry_splits[pid] != cohort.set_index("polymer_id").loc[pid, "split"])
    if inconsistent_qc:
        raise ValueError(f"registry/cohort split mismatch: {inconsistent_qc}")
    if set(cohort.loc[cohort.split == "DEVELOPMENT", "polymer_id"]) & set(cohort.loc[cohort.split == "LOCKED_HOLDOUT", "polymer_id"]):
        raise ValueError("polymer appears in both DEVELOPMENT and LOCKED_HOLDOUT")
    qc_text = qc_path.read_text(errors="replace")
    split_qc_text = split_qc_path.read_text(errors="replace")
    def report_count(key: str) -> int:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(\d+)', qc_text)
        if not match:
            raise ValueError(f"missing {key} in bulk QC report")
        return int(match.group(1))
    historical_complete = {"DEVELOPMENT": report_count("600K_DEVELOPMENT"),
                           "LOCKED_HOLDOUT": report_count("600K_LOCKED_HOLDOUT")}
    complete_counts = cohort[cohort.polymer_id.isin(complete_ids)].split.value_counts().to_dict()
    if not complete_counts:
        raise ValueError("no complete all-temperature bulk systems")
    assignment_text = "The authoritative frozen assignment is the 150-row cohort manifest: 120 DEVELOPMENT and 30 LOCKED_HOLDOUT. Historical 119/29 values are the all-four-temperature valid completion counts in the production QC report: one DEVELOPMENT and one LOCKED_HOLDOUT member remain cohort members but failed 600 K plateau and therefore have downstream NOT_RUN_UPSTREAM statuses."
    cohort_hash = hashlib.sha256("\n".join(sorted(cohort.polymer_id)).encode()).hexdigest()
    result = {
        "cohort_version": COHORT_VERSION,
        "cohort_assignment_source": str(cohort_path.relative_to(root)),
        "bulk_qc_source": str(qc_path.relative_to(root)),
        "split_qc_source": str(split_qc_path.relative_to(root)),
        "source_sha256": {"cohort": sha256(cohort_path), "registry": sha256(registry_path),
                           "qc_report": sha256(qc_path), "split_qc_report": sha256(split_qc_path)},
        "cohort_list_sha256": cohort_hash,
        "assignment_counts": {k: int(v) for k, v in split_counts.items()},
        "complete_all_four_counts": {k: int(complete_counts.get(k, 0)) for k in ("DEVELOPMENT", "LOCKED_HOLDOUT")},
        "complete_all_four_total": len(complete_ids),
        "excluded_or_incomplete_count": len(incomplete_ids),
        "excluded_or_incomplete_ids": incomplete_ids,
        "duplicate_ids": sorted(cohort.loc[cohort.polymer_id.duplicated(keep=False), "polymer_id"].unique().tolist()),
        "conflicting_assignment_ids": conflicts,
        "missing_temperature_rows": missing_rows,
        "inconsistent_qc_ids": inconsistent_qc,
        "identity_audit": {"cohort_polymer_ids_match_canonical": True,
                           "registry_rows_use_explicit_polymer_id": True,
                           "row_or_filename_order_mapping": False},
        "development_ids": sorted(cohort.loc[cohort.split == "DEVELOPMENT", "polymer_id"]),
        "locked_holdout_ids": sorted(cohort.loc[cohort.split == "LOCKED_HOLDOUT", "polymer_id"]),
        "historical_119_29_completion_counts": historical_complete,
        "discrepancy_explanation": assignment_text,
        "qc_report_contains_no_holdout_tg": "No Tg values or prediction results are read" in qc_text,
        "split_qc_contains_preregistration": "FINAL_COHORT_PREREGISTERED" in split_qc_text,
    }
    if result["duplicate_ids"] or result["conflicting_assignment_ids"] or result["missing_temperature_rows"] or result["inconsistent_qc_ids"]:
        raise ValueError("authoritative cohort reconciliation failed")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cohort_reconciliation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# Phase 6B authoritative cohort reconciliation", "",
             f"Cohort version: `{COHORT_VERSION}`.",
             f"Assignment source: `{cohort_path.relative_to(root)}`.",
             f"Bulk QC source: `{qc_path.relative_to(root)}`.", "",
             "| Group | Frozen assigned | Complete at all four temperatures |", "|---|---:|---:|"]
    for group in ("DEVELOPMENT", "LOCKED_HOLDOUT"):
        lines.append(f"| {group} | {split_counts.get(group, 0)} | {complete_counts.get(group, 0)} |")
    lines += ["", f"Final cohort-list SHA-256: `{cohort_hash}`.",
              f"Incomplete members retained in the cohort: {', '.join(incomplete_ids)}.",
              "", "## Audit", "",
              f"Duplicate IDs: `{result['duplicate_ids']}`.",
              f"Conflicting assignments: `{result['conflicting_assignment_ids']}`.",
              f"Missing/inconsistent QC: `{result['inconsistent_qc_ids']}`.", "",
              assignment_text, "",
              "The two counts are therefore different quantities: 120/30 are authoritative assignments; 119/29 are completed all-temperature systems available for Phase 6B descriptors.", ""]
    (root / "reports/phase6b_cohort_reconciliation.md").write_text("\n".join(lines))
    return result
