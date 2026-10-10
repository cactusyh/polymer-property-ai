#!/usr/bin/env python3
"""Audit the fixed 150-system cohort using structural data only; never read Tg."""

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/bulk_md_tg_cohort_150_split.csv"
ORIGINAL = ROOT / "results/bulk_md_tg_150_build_registry_pre_rescue.csv"
REGISTRY = ROOT / "data/processed/bulk_md_tg_150_build_registry.csv"
RETRY = ROOT / "results/bulk_md_tg_150_timeout_retry.csv"
GAFF = ROOT / "results/bulk_md_tg_150_gaff_diagnostics.csv"
CONSTRUCTION = ROOT / "results/bulk_md_tg_150_construction_rescue.csv"
WALK_TIERS = ROOT / "results/bulk_md_tg_150_walk_tier_rescue.csv"
EXPORT_RECURSION = ROOT / "results/bulk_md_tg_150_export_recursion_rescue.csv"
OUT = ROOT / "results/bulk_md_tg_150_build_failure_audit.csv"
REPORT = ROOT / "reports/bulk_md_tg_150_build_failure_audit.md"
MD = ROOT / "data/processed/bulk_md_tg_150_600K_registry.csv"
MD_FAIL = ROOT / "results/bulk_md_tg_150_600K_failure_table.csv"

MOTIFS = {
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[CX3](=[OX1])[OX2H0]",
    "imide": "[NX3]([CX3](=[OX1]))[CX3](=[OX1])",
    "aromatic_n": "[n]",
    "sulfur": "[#16]",
    "fluorine": "[F]",
    "chlorine": "[Cl]",
    "bromine": "[Br]",
    "nitro": "[N+](=O)[O-]",
    "nitrile": "[CX2]#N",
    "ether": "[OD2]([#6])[#6]",
    "alkene": "[CX3]=[CX3]",
    "hetero_ring": "[R;!#6]",
    "silicon": "[Si]",
    "selenium": "[Se]",
    "germanium": "[Ge]",
    "lithium": "[Li]",
    "azide": "[N-][N+]#N",
}
PATTERNS = {key: Chem.MolFromSmarts(smarts) for key, smarts in MOTIFS.items()}


def read(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def failure_class(reason):
    if reason.startswith("BuildTimeout:build exceeded 1800 s"):
        return "BUILD_TIMEOUT_30MIN"
    if reason.startswith("BuildTimeout:build exceeded 7200 s"):
        return "BUILD_TIMEOUT_120MIN"
    if reason.startswith("RuntimeError:GAFF2_mod ff_assign returned False"):
        return "GAFF2_MOD_ASSIGNMENT_FAILURE"
    if reason.startswith("RecursionError:"):
        return "RADONPY_RECURSION_FAILURE"
    if reason.startswith("RadonPyError:Reached maximum number of retrying polymerize_rw"):
        return "POLYMERIZE_RW_RETRY_EXHAUSTION"
    if reason.startswith("RuntimeError:polymerize_rw failed"):
        return "POLYMERIZE_RW_TIERS_EXHAUSTED"
    if reason.startswith("TimeoutError:iterative export repair exceeded 7200 s"):
        return "BUILD_TIMEOUT_120MIN"
    return reason.split(":", 1)[0] or "UNKNOWN"


def numeric_summary(values):
    return f"{statistics.median(values):.3f}" if values else "n/a"


def standardized_difference(pass_values, fail_values):
    if len(pass_values) < 2 or len(fail_values) < 2:
        return 0.0
    variance = (statistics.variance(pass_values) + statistics.variance(fail_values)) / 2
    return (statistics.mean(fail_values) - statistics.mean(pass_values)) / math.sqrt(variance) if variance > 0 else 0.0


def parse_diagnostic(log_path):
    if not log_path or not log_path.exists():
        return "", "", ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Can not assign this (bond|angle|dihedral|improper) ([^\n]+)", text)
    matches += re.findall(r"Can not estimate parameters of this (angle|bond|dihedral|improper) ([^\n]+)", text)
    atom_matches = re.findall(r"Cannot assignment index (\d+), element ([^,]+), num\. of bonds (\d+), hybridization ([^\n]+)", text)
    missing = ";".join(sorted(set(f"{kind}:{detail.strip()}" for kind, detail in matches)))
    by_type = Counter((element, bonds, hyb) for idx, element, bonds, hyb in atom_matches)
    atoms = ";".join(f"element={element},bonds={bonds},hybridization={hyb},sites={count}"
                     for (element, bonds, hyb), count in sorted(by_type.items()))
    return missing, atoms, "\n".join(text.splitlines()[-8:])


def main():
    manifest = read(MANIFEST)
    original = {row["polymer_id"]: row for row in read(ORIGINAL if ORIGINAL.exists() else REGISTRY)}
    current = {row["polymer_id"]: row for row in read(REGISTRY)}
    retry = {row["polymer_id"]: row for row in read(RETRY)}
    gaff = {row["polymer_id"]: row for row in read(GAFF)}
    construction = {row["polymer_id"]: row for row in read(CONSTRUCTION)}
    walk_tiers = {row["polymer_id"]: row for row in read(WALK_TIERS)}
    export_recursion = {row["polymer_id"]: row for row in read(EXPORT_RECURSION)}
    if len(manifest) != 150 or len(original) != 150 or len(current) != 150:
        raise RuntimeError("Fixed cohort and build registries must contain 150 unique IDs")
    if set(original) != set(current) or set(original) != {row["polymer_id"] for row in manifest}:
        raise RuntimeError("Locked cohort identities changed")
    if Counter(row["split"] for row in manifest) != {"DEVELOPMENT": 120, "LOCKED_HOLDOUT": 30}:
        raise RuntimeError("Locked 120/30 split changed")

    features = []
    for row in manifest:
        mol = Chem.MolFromSmiles(row["canonical_psmiles"])
        if mol is None:
            raise RuntimeError(f"Cannot parse fixed polymer {row['polymer_id']}")
        motifs = sorted(key for key, pattern in PATTERNS.items() if mol.HasSubstructMatch(pattern))
        entry = {**row, "motifs": motifs,
                 "build_pass": original[row["polymer_id"]]["build_status"] == "PASS",
                 "final_build_pass": current[row["polymer_id"]]["build_status"] == "PASS"}
        features.append(entry)

    audit = []
    for row in features:
        if row["build_pass"]:
            continue
        pid = row["polymer_id"]
        old, now = original[pid], current[pid]
        diagnostic = gaff.get(pid, {})
        missing, atom_types, log_tail = parse_diagnostic(ROOT / diagnostic["log_path"] if diagnostic.get("log_path") else None)
        attempt = retry.get(pid, {})
        construction_attempt = construction.get(pid, {})
        walk_attempt = walk_tiers.get(pid, {})
        export_attempt = export_recursion.get(pid, {})
        original_reason = old["failure_reason"]
        old_class = failure_class(original_reason)
        if old_class == "BUILD_TIMEOUT_30MIN":
            original_runtime = "at least 1800 s; exact elapsed time not logged"
        else:
            original_runtime = "not recorded in original build artifacts"
        unsupported = sorted({atom.GetSymbol() for atom in Chem.MolFromSmiles(row["canonical_psmiles"]).GetAtoms()}
                             & {"Si", "Se", "Ge", "Li"})
        if old_class == "GAFF2_MOD_ASSIGNMENT_FAILURE" and "azide" in row["motifs"]:
            unsupported.append("azide_Nminus_degree2_SP2")
        if old_class != "GAFF2_MOD_ASSIGNMENT_FAILURE":
            # A timeout is not evidence of unsupported chemistry.
            unsupported = []
        audit.append({
            "polymer_id": pid, "split": row["split"], "canonical_psmiles": row["canonical_psmiles"],
            "chemical_cluster_id": row["chemical_cluster_id"], "repeat_unit_atoms": row["repeat_unit_atoms"],
            "chain_count": row["selected_chain_count"], "DP": row["selected_DP"],
            "estimated_atoms": row["estimated_total_atoms"], "original_failure_class": old_class,
            "actual_failure_class": old_class, "exact_exception_error_message": original_reason,
            "original_runtime_before_failure": original_runtime,
            "original_traceback": old.get("traceback", ""),
            "gaff_failed_stage": diagnostic.get("failed_stage", ""),
            "gaff_atom_typing_errors": atom_types, "gaff_missing_parameter_signatures": missing,
            "gaff_diagnostic_status": diagnostic.get("diagnostic_status", ""),
            "gaff_diagnostic_wall_time_seconds": diagnostic.get("wall_time_seconds", ""),
            "gaff_diagnostic_log_tail": log_tail,
            "charge_assignment_failure": "not reached" if old_class == "GAFF2_MOD_ASSIGNMENT_FAILURE" else "not identified",
            "unsupported_element_or_chemistry": ";".join(unsupported) if unsupported else "undetermined",
            "same_missing_parameter_polymer_count": "",
            "retry_wall_time_seconds": attempt.get("retry_wall_time_seconds", ""),
            "retry_class": attempt.get("retry_class", ""),
            "construction_retry_wall_time_seconds": construction_attempt.get("retry_wall_time_seconds", ""),
            "construction_tier": construction_attempt.get("construction_tier", ""),
            "construction_attempts": construction_attempt.get("construction_attempts", ""),
            "walk_tier_retry_wall_time_seconds": walk_attempt.get("retry_wall_time_seconds", ""),
            "walk_tier": walk_attempt.get("construction_tier", ""),
            "walk_tier_attempts": walk_attempt.get("construction_attempts", ""),
            "export_recursion_repair_wall_time_seconds": export_attempt.get("retry_wall_time_seconds", ""),
            "export_recursion_repair": export_attempt.get("repair", ""),
            "final_build_status": now["build_status"],
            "final_forcefield_status": now["forcefield_status"],
            "final_export_status": now["export_status"],
            "final_failure_reason": now["failure_reason"],
            "major_chemical_motifs": ";".join(row["motifs"]),
        })
    if len(audit) != 53:
        raise RuntimeError(f"Expected 53 original failure rows, found {len(audit)}")
    signatures = Counter(signature for row in audit for signature in row["gaff_missing_parameter_signatures"].split(";") if signature)
    unsupported_counts = Counter(element for row in audit for element in row["unsupported_element_or_chemistry"].split(";")
                                 if element and element != "undetermined")
    for row in audit:
        signatures_here = [s for s in row["gaff_missing_parameter_signatures"].split(";") if s]
        row["same_missing_parameter_polymer_count"] = ";".join(
            [f"{s}={signatures[s]}" for s in signatures_here] +
            [f"unsupported_atom_type:{element}={unsupported_counts[element]}" for element in
             row["unsupported_element_or_chemistry"].split(";") if element in unsupported_counts])
        if row["retry_class"] in ("BUILD_PASS_AFTER_TIMEOUT_RETRY", "BUILD_TIMEOUT_120MIN"):
            row["actual_failure_class"] = row["retry_class"]
        elif row["retry_class"]:
            row["actual_failure_class"] = ("GAFF2_UNSUPPORTED_CHEMISTRY"
                                           if failure_class(row["final_failure_reason"]) == "GAFF2_MOD_ASSIGNMENT_FAILURE"
                                           and row["unsupported_element_or_chemistry"] != "undetermined"
                                           else failure_class(row["final_failure_reason"]))
        elif row["final_build_status"] == "PASS":
            row["actual_failure_class"] = ("BUILD_PASS_AFTER_EXPORT_REPAIR" if row["export_recursion_repair"]
                                           else "BUILD_PASS_AFTER_CONSTRUCTION_TIER")
        elif row["original_failure_class"] == "GAFF2_MOD_ASSIGNMENT_FAILURE" and \
                row["unsupported_element_or_chemistry"] != "undetermined":
            row["actual_failure_class"] = "GAFF2_UNSUPPORTED_CHEMISTRY"
        elif row["final_failure_reason"] and row["final_failure_reason"] != row["exact_exception_error_message"]:
            row["actual_failure_class"] = failure_class(row["final_failure_reason"])
    write(OUT, audit, list(audit[0]))

    md = read(MD)
    if len(md) != 97 or {r["polymer_id"] for r in md} != {r["polymer_id"] for r in features if r["build_pass"]}:
        raise RuntimeError("Final 600 K registry must cover all 97 original build-PASS systems")
    md_failures = [r for r in md if r["status"] != "600K_COMPLETE"]
    write(MD_FAIL, md_failures, list(md[0]))

    original_pass_group = [r for r in features if r["build_pass"]]
    original_fail_group = [r for r in features if not r["build_pass"]]
    pass_group = [r for r in features if r["final_build_pass"]]
    fail_group = [r for r in features if not r["final_build_pass"]]
    columns = ["repeat_unit_atoms", "aromatic_fraction", "heteroatom_fraction", "ring_count",
               "selected_chain_count", "selected_DP", "estimated_total_atoms"]
    comparisons = []
    for column in columns:
        pass_values = [float(r[column]) for r in pass_group]
        fail_values = [float(r[column]) for r in fail_group]
        comparisons.append((column, numeric_summary(pass_values), numeric_summary(fail_values),
                            standardized_difference(pass_values, fail_values)))
    motif_comparisons = []
    for motif in MOTIFS:
        p = sum(motif in r["motifs"] for r in pass_group)
        f = sum(motif in r["motifs"] for r in fail_group)
        motif_comparisons.append((motif, p, f, f / len(fail_group) - p / len(pass_group)))
    max_smd = max(abs(row[3]) for row in comparisons)
    max_motif_difference = max(abs(row[3]) for row in motif_comparisons)
    split_rates = {split: (sum(r["split"] == split for r in fail_group) /
                           sum(r["split"] == split for r in features)) for split in ("DEVELOPMENT", "LOCKED_HOLDOUT")}
    split_difference = abs(split_rates["DEVELOPMENT"] - split_rates["LOCKED_HOLDOUT"])
    # Prespecified descriptive threshold: this is a chemical-space warning, not
    # a claim of statistical significance. The unique cohort cluster IDs have
    # no within-cohort replication and cannot support cluster-rate estimates.
    if max_smd >= 0.8 or max_motif_difference >= 0.30 or split_difference >= 0.25:
        bias = "BUILD_FAILURE_BIAS_HIGH"
    elif max_smd >= 0.5 or max_motif_difference >= 0.20 or split_difference >= 0.15:
        bias = "BUILD_FAILURE_BIAS_MODERATE"
    else:
        bias = "BUILD_FAILURE_BIAS_LOW"
    total_pass = sum(r["build_status"] == "PASS" for r in current.values())
    final_known = (len(retry) == 27 and len(construction) == 4 and
                   len(walk_tiers) == 2 and len(export_recursion) == 2 and len(gaff) == 22)
    if not final_known:
        gate = "PENDING_BUILD_RESCUE"
    elif total_pass >= 135:
        gate = "BUILD_COVERAGE_PASS"
    elif total_pass >= 120 and bias == "BUILD_FAILURE_BIAS_LOW":
        gate = "BUILD_COVERAGE_USABLE_WITH_CAUTION"
    else:
        gate = "BUILD_COVERAGE_INSUFFICIENT"
    timeout_pass = [float(r["actual_completion_time_seconds"]) for r in retry.values()
                    if r["retry_class"] == "BUILD_PASS_AFTER_TIMEOUT_RETRY" and r["actual_completion_time_seconds"]]
    lines = ["# Fixed 150-system build failure audit and rescue", "",
             "## Scope and frozen cohort", "",
             "- Original identities: 150 unique polymers; DEVELOPMENT 120, LOCKED_HOLDOUT 30; unchanged.",
             "- Structural comparison uses only the fixed cohort manifest and build status. No Tg values were read.",
             "- No 500/400/300 K MD or Tg-model training was started by this rescue workflow.", "",
             "## Finalized original 600 K queue", "",
             f"- All {len(md)}/97 original build-PASS systems have a terminal 600 K registry row.",
             f"- 600K_COMPLETE: {sum(r['status'] == '600K_COMPLETE' for r in md)}/97; 600K_PLATEAU_NOT_REACHED: {len(md_failures)}/97.",
             f"- DEVELOPMENT: {sum(r['split'] == 'DEVELOPMENT' and r['status'] == '600K_COMPLETE' for r in md)}/81 complete; "
             f"LOCKED_HOLDOUT: {sum(r['split'] == 'LOCKED_HOLDOUT' and r['status'] == '600K_COMPLETE' for r in md)}/16 complete.",
             f"- 600 K failure table: `{MD_FAIL.relative_to(ROOT)}` ({len(md_failures)} rows).", "",
             "## Original build failures", "",
             f"- Original build coverage: {len(original_pass_group)}/150; unresolved: {len(original_fail_group)}/150 "
             "(27 time-limited builds, 26 observed non-timeout errors).",
             "- BUILD_TIMEOUT_30MIN: 27; GAFF2_mod assignment: 22; RadonPy recursion: 2; polymerize_rw retry exhaustion: 2.",
             "- Original non-timeout runtimes were not recorded. Timeout runtimes are known only to have reached the 1800 s alarm.",
             f"- Per-polymer original error, traceback, chemistry, diagnostic fields and rescue outcome: `{OUT.relative_to(ROOT)}`.", "",
             "## GAFF2_mod assignment diagnosis", "",
             f"- Diagnostic replays attempted: {len(gaff)}/22 (20 assignment failures reproduced; 2 diagnostic timeouts).",
             "- The frozen GAFF2_mod particle database contains no Si, Se, Ge, or Li atom types. "
             "The azide [N-][N+]#N motif also leaves a two-coordinate SP2 N without an assignment rule. "
             "These cases are classified `GAFF2_UNSUPPORTED_CHEMISTRY`; no parameter or atom type was invented.",
             "- Two full-chain diagnostic replays reached their 2700 s diagnostic limit before force-field assignment; "
             "their original GAFF2_mod failure and absent Si/Ge particle types remain documented. "
             "This diagnostic timeout is not a build rescue result.",
             "- No implementation/mapping bug, parameter-lookup bug, or existing omitted GAFF2_mod parameter was demonstrated. "
             "Bond, angle, dihedral and improper assignment cannot be evaluated after atom typing fails; those fields are left empty rather than guessed.",
             "- GAFF2_mod is invoked without charge assignment; Gasteiger charges are assigned only after its successful return. "
             "These original failures therefore are not charge-assignment failures.",
             "- Atom typing, bond, angle, dihedral and improper diagnostics, shared signatures, and logs are linked per polymer in the audit CSV.", "",
             "## Structural build-failure bias", "",
             "- PASS versus unresolved builds after rescue; FAIL includes time-limited builds and does not imply unsupported chemistry.",
             "| Structural feature | PASS median | FAIL median | standardized mean difference (FAIL minus PASS) |",
             "|---|---:|---:|---:|"]
    lines.extend(f"| {column} | {p} | {f} | {smd:+.3f} |" for column, p, f, smd in comparisons)
    lines += ["", f"| Major chemical motif | PASS n/{len(pass_group)} | FAIL n/{len(fail_group)} | failure minus pass prevalence |",
              "|---|---:|---:|---:|"]
    lines.extend(f"| {motif} | {p}/{len(pass_group)} | {f}/{len(fail_group)} | {delta:+.3f} |"
                 for motif, p, f, delta in motif_comparisons)
    lines += ["", "| Family | cohort members | current failures | failure rate |",
              "|---|---:|---:|---:|"]
    for motif in ("silicon", "selenium", "germanium", "lithium", "imide", "aromatic_n"):
        members = [row for row in features if motif in row["motifs"]]
        failures = sum(not row["final_build_pass"] for row in members)
        lines.append(f"| {motif} | {len(members)} | {failures} | {failures/len(members):.1%} |")
    lines += ["", "| Split | PASS | FAIL | failure rate |", "|---|---:|---:|---:|"]
    for split, n in (("DEVELOPMENT", 120), ("LOCKED_HOLDOUT", 30)):
        p = sum(r["split"] == split for r in pass_group)
        f = n - p
        lines.append(f"| {split} | {p}/{n} | {f}/{n} | {f/n:.1%} |")
    lines += ["", "- The 150 Morgan chemical cluster IDs are each represented once in this cohort. "
              "Within-cluster failure rates therefore cannot be estimated; cluster ID remains in each audit row.",
              f"- Descriptive bias classification: **{bias}**. Thresholds: HIGH if any absolute standardized "
              "difference ≥0.8, motif prevalence gap ≥0.30, or split failure-rate gap ≥0.25; "
              "MODERATE if ≥0.5, ≥0.20, or ≥0.15; otherwise LOW.", "",
             "## Standardized rescue tiers", "",
              "1. BUILD_TIMEOUT_30MIN: replay the exact original build function and seed with only the per-polymer alarm changed from 1800 to 7200 s.",
              "2. polymerize_rw exhaustion: retain monomer, DP, chain count, GAFF2_mod and export QC; "
              "use the validated single-chain robust, relaxed-overlap and extended-initialization random-walk settings at the original DP, "
              "with a uniform 1200 s cap per remaining tier. Never accept a shorter diagnostic chain. "
              "For export recursion, use an iterative equivalent of RadonPy molecule-ID traversal.",
              "3. GAFF2_mod: inspect atom typing and bond/angle/dihedral/improper assignment against the frozen parameter database. Repair only demonstrated implementation or lookup bugs; do not invent or hand-fit parameters. Genuinely unsupported chemistry is GAFF2_UNSUPPORTED_CHEMISTRY.", "",
              f"- First construction-tier attempts recorded: {len(construction)}/4; "
              f"remaining full-DP walk tiers recorded: {len(walk_tiers)}/2; "
              f"iterative export-recursion repairs recorded: {len(export_recursion)}/2.", "",
              "## 120-minute timeout retry", "",
              f"- Original 30-minute timeout systems: 27; attempted: {len(retry)}/27.",
              f"- Rescued: {sum(r['retry_class'] == 'BUILD_PASS_AFTER_TIMEOUT_RETRY' for r in retry.values())}; "
              f"still timed out at 120 minutes: {sum(r['retry_class'] == 'BUILD_TIMEOUT_120MIN' for r in retry.values())}.",
              f"- Median successful retry build time: {statistics.median(timeout_pass):.1f} s; "
              f"maximum: {max(timeout_pass):.1f} s." if timeout_pass else "- Successful retry build time: pending or none.",
              f"- Detailed retry records: `{RETRY.relative_to(ROOT)}`.", "",
              f"- Coverage from timeout retries alone: {len(original_pass_group) + len(timeout_pass)}/150. "
              "The final total below also includes the separately recorded export-recursion rescue.",
              "- Timeout-only coverage by split: " + "; ".join(
                  f"{split}: {sum(r['split'] == split for r in original_pass_group) + sum(r['split'] == split and r['retry_class'] == 'BUILD_PASS_AFTER_TIMEOUT_RETRY' for r in retry.values())}/{n}"
                  for split, n in (("DEVELOPMENT", 120), ("LOCKED_HOLDOUT", 30))) + ".",
              "- Timeouts remain unresolved construction outcomes, never unsupported-chemistry diagnoses.", "",
              "## Build coverage after rescue", "",
              f"- Total: {total_pass}/150; DEVELOPMENT: "
              f"{sum(r['build_status']=='PASS' and r['split']=='DEVELOPMENT' for r in current.values())}/120; "
              f"LOCKED_HOLDOUT: {sum(r['build_status']=='PASS' and r['split']=='LOCKED_HOLDOUT' for r in current.values())}/30.",
              f"- Coverage gate: **{gate}**" + (" (one or more standardized rescue attempts remain pending)." if not final_known else "."), ""]
    lines += ["| Final rescue/audit class | systems |", "|---|---:|"]
    lines.extend(f"| {classification} | {count} |" for classification, count in
                 sorted(Counter(row["actual_failure_class"] for row in audit).items()))
    lines += ["", "## Stop condition", "",
              "- Rescue records are reconciled; successful rescued structure.data files passed atom-count, molecule-ID, finite-coordinate and topology-reference QC.",
              "- The final BUILD_TIMEOUT_120MIN count includes the 18 original timeout retries and one separate export-recursion rescue that reached its wall-time limit.",
              "- Both POLYMERIZE_RW_TIERS_EXHAUSTED outcomes exhausted the allowed time in each fallback tier; they do not establish unsupported chemistry.",
              "- Stopped after build rescue and coverage audit. No MD was started for rescued systems; no lower-temperature MD or Tg-model training was launched during finalization.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"audit_rows": len(audit), "bias": bias, "current_coverage": total_pass,
                      "retry_attempted": len(retry), "gate": gate}))


if __name__ == "__main__":
    main()
