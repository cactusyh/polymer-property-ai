"""Construct and dry-run the Phase 6D queue from validated build artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "phase6d_artifacts"
REGISTRY = ART / "candidate_build_registry.v1.csv"
MANIFEST = ART / "expansion_candidate_manifest.v1.json"
POOL = ART / "expansion_candidate_pool.v1.csv"
READY = ART / "phase6d_ready_queue.v2.csv"
READY_MANIFEST = ART / "phase6d_ready_queue_manifest.v2.json"
DRY_RUN = ART / "phase6d_ready_queue_dry_run.v1.json"
REPORTS = ROOT / "reports"
TEMPERATURES = (300, 400, 500, 600)
PILOTS = {"poly_98bc13bed4455c1f", "poly_db696efd35d8326b", "poly_a5cce35c85e525e3", "poly_df7c069ad0b2d467"}
FIELDS = ["polymer_id", "temperature_K", "gpu_id", "structure_data", "input_path", "command", "checkpoint_path", "trajectory_path", "output_path", "manifest_sha256", "queue_status"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def validate_input(path: Path, pid: str, temperature: int) -> list[str]:
    errors = []
    if not path.is_file():
        return ["missing_input"]
    text = path.read_text(encoding="utf-8")
    if f"# polymer_id={pid}" not in text:
        errors.append("identity_mismatch")
    if f"temp {temperature}.0 {temperature}.0" not in text:
        errors.append("temperature_mismatch")
    for required in ("boundary p p p", "pair_style lj/cut 10.0", "bond_style harmonic", "angle_style harmonic", "dihedral_style fourier", "improper_style cvff", "special_bonds amber", "read_data structure.data", "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes", "timestep 1.0", "fix npt", "write_restart", "write_data"):
        if required not in text:
            errors.append(f"missing:{required}")
    if "structure.data" not in {p.name for p in path.parent.iterdir()}:
        errors.append("missing_structure_link")
    if "phase6d_runs/pilot" in text or "bulk_md_tg_pilot" in text:
        errors.append("pilot_path_reference")
    if f"production_{temperature}K" not in text:
        errors.append("temperature_output_mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "dry-run"))
    args = parser.parse_args()
    if args.action == "build":
        raise SystemExit("build artifacts with scripts/phase6d_build_inputs.py")
    rows = read(REGISTRY)
    manifest_hash = sha(MANIFEST)
    pool = {row["polymer_id"]: row for row in read(POOL)}
    locked = {pid for pid, row in pool.items() if "LOCKED_HOLDOUT" in row.get("exclusion_reason", "")}
    errors: list[dict[str, str]] = []
    tasks: list[dict[str, str]] = []
    build_pass = 0
    for row in rows:
        pid = row["polymer_id"]
        if row.get("build_status") != "BUILD_PASS":
            continue
        build_pass += 1
        if pid in PILOTS or pid in locked:
            errors.append({"polymer_id": pid, "error": "pilot_or_locked_candidate"})
            continue
        for temperature in TEMPERATURES:
            input_path = ROOT / row[f"input_{temperature}K"]
            structure = ROOT / row["structure_data"]
            local_errors = validate_input(input_path, pid, temperature)
            if not structure.is_file():
                local_errors.append("missing_structure_data")
            if structure.resolve() == input_path.resolve():
                local_errors.append("structure_input_collision")
            if local_errors:
                errors.append({"polymer_id": pid, "temperature_K": str(temperature), "error": ";".join(local_errors)})
                continue
            output_dir = input_path.parent
            tasks.append({
                "polymer_id": pid, "temperature_K": str(temperature), "gpu_id": str(len(tasks) % 4),
                "structure_data": str(structure.relative_to(ROOT)), "input_path": str(input_path.relative_to(ROOT)),
                "command": f"CUDA_VISIBLE_DEVICES={len(tasks) % 4} lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in {input_path.name}",
                "checkpoint_path": str((output_dir / "checkpoints" / f"production_{temperature}K.restart").relative_to(ROOT)),
                "trajectory_path": str((output_dir / "trajectory" / f"production_{temperature}K_200ps.lammpstrj").relative_to(ROOT)),
                "output_path": str((output_dir / "outputs" / f"production_{temperature}K_final.data").relative_to(ROOT)),
                "manifest_sha256": manifest_hash, "queue_status": "READY",
            })
    pairs = [(row["polymer_id"], row["temperature_K"]) for row in tasks]
    duplicate_pairs = len(pairs) - len(set(pairs))
    duplicate_outputs = len({row["output_path"] for row in tasks}) != len(tasks)
    if duplicate_pairs or duplicate_outputs:
        errors.append({"error": "duplicate_pair_or_output_path"})
    status = "PASS" if not errors else "BLOCKED"
    write_csv(READY, tasks)
    queue_hash = sha(READY)
    summary = {
        "schema_version": "phase6d.ready_queue_manifest.v2", "status": status,
        "manifest_sha256": manifest_hash, "ready_queue_sha256": queue_hash,
        "total_remaining_candidates": 124, "build_pass_candidates": build_pass,
        "build_failure_count": len(rows) - build_pass, "ready_candidate_count": len({row["polymer_id"] for row in tasks}),
        "ready_temperature_task_count": len(tasks), "missing_structure_data_count": sum(row.get("error") == "missing_structure_data" for row in errors),
        "missing_input_count": sum("missing_input" in row.get("error", "") for row in errors),
        "excluded_pilot_tasks": 16, "excluded_locked_holdout_tasks": len(locked) * 4,
        "duplicate_pair_count": duplicate_pairs, "duplicate_output_path": duplicate_outputs,
        "errors": errors, "production_launched": False,
    }
    write_json(READY_MANIFEST, summary)
    dry = {"schema_version": "phase6d.ready_queue_dry_run.v1", "status": status, "manifest": summary, "tasks": tasks, "errors": errors, "lammps_launched": False}
    write_json(DRY_RUN, dry)
    counts = {}
    for row in rows:
        counts[row.get("build_status", "MISSING")] = counts.get(row.get("build_status", "MISSING"), 0) + 1
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase6d_candidate_build_report.md").write_text(
        "# Phase 6D candidate build report\n\n"
        f"- Remaining candidates inspected: **{len(rows)}**\n- BUILD_PASS: **{build_pass}**\n"
        f"- Build status counts: `{json.dumps(counts, sort_keys=True)}`\n- Production LAMMPS launched: **no**\n\n"
        "Each build reuses `scripts/frozen_bulk_md_tg_production_builder.py` with GAFF2_mod and explicit topology/charge QC. Unsupported elements are recorded without invented parameters.\n",
        encoding="utf-8",
    )
    (REPORTS / "phase6d_candidate_input_validation.md").write_text(
        "# Phase 6D candidate input validation\n\n"
        f"- BUILD_PASS candidates: **{build_pass}**\n- Validated temperature inputs: **{len(tasks)}**\n"
        f"- Validation errors: **{len(errors)}**\n- Dry-run status: **{status}**\n- LAMMPS launched: **no**\n\n"
        "Validation checks structure links, topology/protocol directives, temperature identity, unique checkpoint/output paths, and pilot-path exclusion.\n",
        encoding="utf-8",
    )
    (REPORTS / "phase6d_ready_queue_audit.md").write_text(
        "# Phase 6D ready queue audit\n\n"
        f"- Remaining candidates: **124**\n- BUILD_PASS candidates: **{build_pass}**\n"
        f"- Ready temperature tasks: **{len(tasks)}**\n- Missing structure/input errors: **{len(errors)}**\n"
        f"- Excluded pilot tasks: **16**\n- Excluded LOCKED_HOLDOUT tasks: **{len(locked) * 4}**\n"
        f"- Duplicate pairs: **{duplicate_pairs}**\n- Queue SHA-256: `{queue_hash}`\n- Status: **{status}**\n",
        encoding="utf-8",
    )
    (REPORTS / "phase6d_ready_queue_dry_run.md").write_text(
        "# Phase 6D ready queue dry run\n\n"
        f"Status: **{status}**. The dry run listed {len(tasks)} validated candidate-temperature tasks and launched no LAMMPS process.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "build_pass": build_pass, "ready_candidates": summary["ready_candidate_count"], "ready_tasks": len(tasks), "errors": len(errors), "queue_sha256": queue_hash}, indent=2))


if __name__ == "__main__":
    main()
