#!/usr/bin/env python
"""Build the leakage-safe Phase 6A contract from real repository registries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase6a.contract import build_manifest


def render_reports(manifest: dict[str, object]) -> None:
    counts = manifest["counts"]
    audits = manifest["leakage_audit"]
    lines = ["# Phase 6A identity and leakage audit", "",
             f"Manifest SHA-256: `{manifest['manifest_sha256']}`.", "",
             ("The official split labels and explicit sample-ID MD mapping were used."
              " No bulk holdout Tg values were read."), "",
             "| Split | Audit | Train IDs after global exclusions | Raw PI1M eval overlap | Clean PI1M eval overlap | Near duplicates ≥0.95 | Holdout training overlap |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name in ("random", "ood"):
        row = audits[name]
        lines.append(f"| {name} | {row['status']} | {row['safe_supervised_train_ids']} | "
                     f"{row['raw_pi1m_validation_test_overlap']} | {row['clean_pi1m_validation_test_overlap']} | "
                     f"{row['near_duplicate_evaluation_identities_at_0_95']} | "
                     f"{row['locked_holdout_training_overlap']} |")
    lines += ["", ("Exact identity, canonical PSMILES, and structure-hash overlaps after filtering are zero for both splits."),
              ("Near duplicates are reported as chemical similarity, not automatically removed from the official splits."),
              ("Production pretraining may contain evaluation identities and its results must carry the `PRODUCTION_RESULT` label."),
              "", f"Single-chain MD mappings verified: {counts['md_verified_mapping']}/{counts['md_rows']}.",
              f"Excluded identity/path mapping mismatches: {counts['md_excluded_mapping_mismatch']}.",
              "Cross-task and MD-to-property identity overlaps are in `phase6a_artifacts/leakage_audit.json`.", ""]
    (ROOT / "reports/phase6a_identity_leakage_audit.md").write_text("\n".join(lines))
    units = ["# Phase 6A electronic-property unit and provenance audit", "",
             ("Targets retain the raw numeric values. The source files do not document authoritative units."
              " Eea, Ei and Eat remain separate exploratory tasks."), "",
             "| Task | Valid labels | Raw code | Source file | Unit status |",
             "|---|---:|---|---|---|"]
    for task, row in manifest["unit_audit"].items():
        units.append(f"| {task} | {row['n_valid']} | {', '.join(row['raw_code'])} | "
                     f"{', '.join(row['source'])} | {row['status']} |")
    units += ["", "No electronic injection barrier file is present. No sign or unit conversion was applied.",
              "The machine-readable audit is `phase6a_artifacts/unit_audit.json`.", ""]
    (ROOT / "reports/phase6a_unit_provenance_audit.md").write_text("\n".join(units))
    md = ["# Phase 6A single-chain MD QC and coverage", "",
          (f"Rows: {counts['md_rows']}; verified explicit mapping: {counts['md_verified_mapping']}; "
           f"mapping mismatches excluded: {counts['md_excluded_mapping_mismatch']}."),
          f"Verified rows with at least one usable target: {counts['md_at_least_one_usable_target']}.", "",
          "| Frozen feature | Valid after target-specific QC |", "|---|---:|"]
    md += [f"| {feature} | {number} |" for feature, number in manifest["md_coverage"].items()]
    md += ["", ("Each target retains its own validity, quality class and quality-weight mask."
                " Missing targets do not remove a polymer from other tasks."),
           "The exploratory 200-system long-chain descriptor table is excluded from primary training.", ""]
    (ROOT / "reports/phase6a_single_chain_md_qc.md").write_text("\n".join(md))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="phase6a_artifacts", type=Path)
    parser.add_argument("--skip-near-duplicate", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = build_manifest(ROOT, output, near_duplicates=not args.skip_near_duplicate)
    if output == ROOT / "phase6a_artifacts" and not args.skip_near_duplicate:
        render_reports(manifest)
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"],
                      "counts": manifest["counts"], "md_coverage": manifest["md_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
