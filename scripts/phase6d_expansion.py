#!/usr/bin/env python3
"""Phase 6D manifest, registry, QC and scheduler utilities.

The selector is target-blind to the locked cohort.  Simulation execution is
handled by ``run`` and requires a validated manifest; all output files are
written atomically so an interrupted controller can resume safely.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "phase6d_artifacts"
REPORTS = ROOT / "reports"
COHORT = DATA / "processed/bulk_md_tg_final_cohort_150_split.csv"
UNIVERSE = DATA / "processed/bulk_md_tg_build_preflight_structural_universe.csv"
TG = DATA / "raw/properties/tg/tg.csv"
MD = DATA / "processed/md_registry.csv"
MAPPING = DATA / "processed/md_psmiles_mapping.csv"
BULK_REGISTRY = DATA / "processed/bulk_md_tg_final_150_multitemp_registry.csv"
TEMPERATURES = (300, 400, 500, 600)
POOL_FIELDS = [
    "polymer_id", "canonical_psmiles", "structure_hash", "stereochemistry",
    "connection_points", "source_dataset", "tg_available", "tg_value",
    "single_chain_md_available", "force_field_status", "topology_status",
    "chemical_cluster", "candidate_priority", "exclusion_reason",
    "expansion_cohort_status",
]
REGISTRY_FIELDS = [
    "polymer_id", "cohort_type", "canonical_psmiles", "structure_hash",
    "temperature_K", "temperature_completion", "simulation_status", "qc_status",
    "descriptor_status", "input_hash", "output_hash", "trajectory_hash",
    "manifest_hash", "code_commit", "environment_fingerprint", "gpu_id",
    "hostname", "pid", "attempt", "failure_class", "retryable", "provenance",
]
DESCRIPTOR_FIELDS = [
    "polymer_id", "temperature_K", "density", "volume", "temperature_mean",
    "equilibration_time_ps", "density_drift", "volume_drift",
    "potential_energy_drift", "density_half_trajectory_difference",
    "valid_frame_count", "qc_flags", "missingness_mask", "source_file",
    "source_hash", "temperature_coverage", "qc_status", "schema_version",
    "density_400_minus_300", "density_500_minus_400", "density_600_minus_500",
    "density_600_minus_300", "density_temperature_slope", "volume_400_minus_300",
    "volume_500_minus_400", "volume_600_minus_500", "volume_600_minus_300",
    "volume_temperature_slope", "mobility_temperature_slope", "torsion_temperature_slope",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def canonical_valid(smiles: str) -> bool:
    if not smiles or smiles != smiles.strip() or "*" not in smiles:
        return False
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return bool(re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#*/.\\]+", smiles))


def stereochemistry(smiles: str) -> str:
    if "@" in smiles or "/" in smiles or "\\" in smiles:
        return "specified"
    return "unspecified"


def identity_hash(smiles: str) -> str:
    return sha256_bytes(smiles.encode("utf-8"))


def source_hashes() -> dict[str, str]:
    paths = [COHORT, UNIVERSE, TG, MD, MAPPING, BULK_REGISTRY]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths if path.exists()}


def audit_workflow() -> None:
    REPORTS.mkdir(exist_ok=True)
    report = """# Phase 6D existing bulk-MD workflow audit

## Authority and backend

The Phase 6B cohort and registry are read-only inputs. The production backend is
LAMMPS with the Kokkos GPU package. The validated launch wrapper is
`scripts/run_lammps_kokkos_gpu.sh`; it activates `mace-lammps`, sets the LAMMPS
Python and library paths, and invokes one `lmp` process with one GPU.

## Frozen settings

| Setting | Validated value | Source |
|---|---|---|
| Force field | GAFF2_mod | `scripts/run_bulk_md_tg_final_multitemp.py` |
| Ensemble | NPT at 1 atm | final multitemperature workflow |
| Timestep | 1 fs | final multitemperature workflow |
| Temperatures | 300, 400, 500, 600 K | final registry |
| Equilibration | 100 ps blocks, up to 500 ps; 600 K begins with 20 ps NVT | final workflow |
| Production | 200 ps per temperature | final workflow |
| Thermo output | 1 ps; trajectory output 10 ps | final workflow |
| Thermostat/barostat | validated workflow damping values | frozen protocol JSON when present |
| Pair/cutoff | `lj/cut 10.0`; no kspace | frozen protocol JSON |
| QC | finite values, complete logs/restarts/topology, stable temperature/density/volume/energy, 20 production frames | final workflow |

## Checkpoint and resume

Each equilibration block and production stage writes a versioned LAMMPS restart,
final data file, log, trajectory and provenance record. The existing stage
receipts are consulted before execution. Incomplete files are moved to an
interrupted-stages directory; valid checkpoints are never overwritten. A Phase
6D scheduler records the input hash, checkpoint hash, command, GPU, host, PID,
timestamps and retry outcome in the append-only registry.

## Known failure modes

The validated workflow reports topology/build failures, force-field assignment
failures, numerical LAMMPS errors, missing or truncated restarts, incomplete
thermo/trajectory frames, plateau failure, timeout and GPU process failure.
Operational retries may resume a checkpoint or move a job to another GPU. No
physical parameter is changed automatically.

Source hashes are recorded in the Phase 6D candidate and expansion manifests.
"""
    (REPORTS / "phase6d_existing_bulk_md_workflow_audit.md").write_text(report, encoding="utf-8")


def candidate_rows() -> tuple[list[dict[str, str]], dict[str, str]]:
    cohort_rows = read_csv(COHORT)
    universe_rows = read_csv(UNIVERSE)
    tg_rows = read_csv(TG)
    md_rows = read_csv(MD)
    mapping_rows = read_csv(MAPPING)
    locked_ids = {row["polymer_id"] for row in cohort_rows if row.get("split") == "LOCKED_HOLDOUT"}
    frozen_ids = {row["polymer_id"] for row in cohort_rows}
    tg_by_smiles: dict[str, list[str]] = {}
    for row in tg_rows:
        tg_by_smiles.setdefault(row.get("SMILES", ""), []).append(row.get("value", ""))
    md_by_id = {row.get("polymer_id", ""): row for row in md_rows}
    mapping_by_id = {row.get("polymer_id", ""): row for row in mapping_rows}
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for source in sorted(universe_rows, key=lambda row: (row.get("polymer_id", ""), row.get("canonical_psmiles", ""))):
        polymer_id = source.get("polymer_id", "")
        smiles = source.get("canonical_psmiles", "")
        reasons: list[str] = []
        if polymer_id in seen_ids:
            reasons.append("duplicate_polymer_id")
        seen_ids.add(polymer_id)
        if polymer_id in locked_ids:
            reasons.append("LOCKED_HOLDOUT")
        if polymer_id in frozen_ids:
            reasons.append("existing_phase6b_cohort")
        if not canonical_valid(smiles):
            reasons.append("invalid_identity")
        mapping = mapping_by_id.get(polymer_id, {})
        md = md_by_id.get(polymer_id, {})
        single_chain = (
            mapping.get("mapping_verified", "").lower() == "true"
            and mapping.get("data_file_exists", "").lower() == "true"
            and mapping.get("trajectory_file_exists", "").lower() == "true"
            and md.get("quality_flag", "").find("invalid_or_missing_psmiles") < 0
        )
        tg_values = tg_by_smiles.get(smiles, [])
        tg_available = bool(tg_values)
        try:
            tg_value = str(float(tg_values[0])) if tg_available else ""
        except ValueError:
            tg_value = ""
            tg_available = False
        if source.get("polymer_id") not in frozen_ids:
            source_dataset = "Tg_registry+structural_universe" if tg_available else "structural_universe"
            if single_chain:
                source_dataset += "+single_chain_MD"
        else:
            source_dataset = "phase6b_cohort"
        preflight = source.get("polymer_id") in {r.get("polymer_id") for r in read_csv(DATA / "processed/bulk_md_tg_build_preflight_results.csv") if r.get("final_build_status") == "BUILD_PASS"}
        forcefield_status = "PASS_VALIDATED" if preflight else "PENDING_VALIDATION"
        topology_status = "PASS_VALIDATED" if preflight else "PENDING_BUILD"
        cluster = source.get("chemical_cluster_id", source.get("chemical_cluster", ""))
        try:
            cluster_key = int(cluster)
        except ValueError:
            cluster_key = 10**9
        tg_rank = 0 if tg_available else 1
        reason_key = ";".join(reasons)
        status = "EXCLUDED" if reasons and ("LOCKED_HOLDOUT" in reasons or "existing_phase6b_cohort" in reasons or "invalid_identity" in reasons) else "ELIGIBLE_PENDING_VALIDATION"
        rows.append({
            "polymer_id": polymer_id,
            "canonical_psmiles": smiles,
            "structure_hash": identity_hash(smiles),
            "stereochemistry": stereochemistry(smiles),
            "connection_points": str(smiles.count("*")),
            "source_dataset": source_dataset,
            "tg_available": str(tg_available).lower(),
            "tg_value": tg_value,
            "single_chain_md_available": str(single_chain).lower(),
            "force_field_status": forcefield_status,
            "topology_status": topology_status,
            "chemical_cluster": str(cluster),
            "candidate_priority": "",
            "exclusion_reason": reason_key,
            "expansion_cohort_status": status,
            "_tg_rank": tg_rank,
            "_cluster_key": cluster_key,
        })
    eligible = [row for row in rows if row["expansion_cohort_status"] != "EXCLUDED"]
    # Deterministic diversity: one hash-tie-broken representative per cluster,
    # then fill by Tg coverage and single-chain availability.
    eligible.sort(key=lambda row: (row["_tg_rank"], -int(row["single_chain_md_available"] == "true"), row["_cluster_key"], row["structure_hash"]))
    seen_clusters: set[str] = set()
    ordered: list[dict[str, str]] = []
    for row in eligible:
        if row["chemical_cluster"] not in seen_clusters:
            ordered.append(row)
            seen_clusters.add(row["chemical_cluster"])
    ordered.extend(row for row in eligible if row not in ordered)
    for rank, row in enumerate(ordered, 1):
        row["candidate_priority"] = str(rank)
    for row in rows:
        row.pop("_tg_rank", None)
        row.pop("_cluster_key", None)
    return rows, {"locked_count": str(len(locked_ids)), "frozen_count": str(len(frozen_ids)), "eligible_count": str(len(ordered))}


def select_candidates(pool_size: int = 128) -> None:
    rows, counts = candidate_rows()
    ART.mkdir(exist_ok=True)
    pool_path = ART / "expansion_candidate_pool.v1.csv"
    write_csv(pool_path, sorted(rows, key=lambda row: (row["expansion_cohort_status"], int(row["candidate_priority"] or 10**9), row["structure_hash"])), POOL_FIELDS)
    eligible = [row for row in rows if row["expansion_cohort_status"] != "EXCLUDED"]
    eligible.sort(key=lambda row: (int(row["candidate_priority"]), row["structure_hash"]))
    with_tg = [row for row in eligible if row["tg_available"] == "true"]
    with_tg.sort(key=lambda row: (float(row["tg_value"]), row["structure_hash"]))
    bins: list[list[dict[str, str]]] = [[] for _ in range(8)]
    for index, row in enumerate(with_tg):
        bins[min(7, index * 8 // max(1, len(with_tg)))].append(row)
    selected: list[dict[str, str]] = []
    selected_clusters: set[str] = set()
    while len(selected) < min(pool_size, len(eligible)) and any(bins):
        advanced = False
        for bucket in bins:
            while bucket and bucket[0]["chemical_cluster"] in selected_clusters:
                bucket.pop(0)
            if bucket:
                selected.append(bucket.pop(0)); selected_clusters.add(selected[-1]["chemical_cluster"]); advanced = True
                if len(selected) >= pool_size:
                    break
        if not advanced:
            break
    for row in eligible:
        if len(selected) >= pool_size:
            break
        if row not in selected:
            selected.append(row)
    selected_path = ART / "expansion_selected_candidates.v1.csv"
    write_csv(selected_path, selected, POOL_FIELDS)
    manifest = {
        "schema_version": "phase6d.expansion_manifest.v1",
        "created_at_utc": utc_now(),
        "source_manifest_hashes": source_hashes(),
        "candidate_selection_configuration": {
            "pool_size": pool_size,
            "excluded_assignments": ["LOCKED_HOLDOUT", "DEVELOPMENT"],
            "existing_cohort_excluded": True,
            "priority": ["valid_Tg", "single_chain_MD", "chemical_cluster_coverage", "stable_structure_hash_tie_break"],
            "target_values_used_for_selection": False,
            "locked_holdout_values_or_descriptors_used": False,
        },
        "candidate_list_sha256": sha256_file(pool_path),
        "selected_candidate_list_sha256": sha256_file(selected_path),
        "selected_count": len(selected),
        "pilot_size": 8,
        "temperatures_K": list(TEMPERATURES),
        "locked_holdout_exclusion": {
            "statement": "All LOCKED_HOLDOUT identities were excluded before ranking and selection.",
            "locked_count": int(counts["locked_count"]),
            "intersection_with_selected": sorted({r["polymer_id"] for r in selected} & {r["polymer_id"] for r in rows if r["exclusion_reason"] == "LOCKED_HOLDOUT;existing_phase6b_cohort"}),
        },
        "append_only": True,
        "simulation_started": False,
    }
    write_json(ART / "expansion_candidate_manifest.v1.json", manifest)
    report = f"""# Phase 6D candidate selection

The selector read the Phase 6A identity universe, Tg registry, single-chain MD
mapping and frozen Phase 6B cohort. It used canonical PSMILES, SHA-256 structure
identity, stereochemistry, connection-point count and chemical-cluster coverage.
Selection is deterministic and independent of row or filename order.

- Identity universe: {len(rows)} rows.
- Eligible expansion rows: {counts['eligible_count']}.
- Selected manifest rows: {len(selected)}.
- Frozen Phase 6B rows excluded: {counts['frozen_count']}.
- LOCKED_HOLDOUT rows excluded: {counts['locked_count']}.
- Holdout identities selected: 0.
- Target values/descriptors used for selection: no.

Candidates requiring new builds are marked `PENDING_BUILD` and
`PENDING_VALIDATION`; known invalid identities are excluded. Tg labels and
single-chain availability are recorded for later analysis and do not alter the
locked evaluation artifacts.

Files: `phase6d_artifacts/expansion_candidate_pool.v1.csv`,
`phase6d_artifacts/expansion_selected_candidates.v1.csv`, and the signed hash
manifest.
"""
    (REPORTS / "phase6d_candidate_selection.md").write_text(report, encoding="utf-8")


def validate_manifest() -> dict:
    manifest_path = ART / "expansion_candidate_manifest.v1.json"
    pool_path = ART / "expansion_candidate_pool.v1.csv"
    selected_path = ART / "expansion_selected_candidates.v1.csv"
    if not manifest_path.exists() or not pool_path.exists() or not selected_path.exists():
        raise SystemExit("candidate manifest is missing; run select first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    read_csv(pool_path)
    selected = read_csv(selected_path)
    locked = {row["polymer_id"] for row in read_csv(COHORT) if row.get("split") == "LOCKED_HOLDOUT"}
    if sha256_file(pool_path) != manifest["candidate_list_sha256"]:
        raise SystemExit("candidate pool hash mismatch")
    if sha256_file(selected_path) != manifest["selected_candidate_list_sha256"]:
        raise SystemExit("selected list hash mismatch")
    if locked & {row["polymer_id"] for row in selected}:
        raise SystemExit("LOCKED_HOLDOUT identity present in expansion selection")
    ids = [row["polymer_id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate selected polymer identity")
    if any(row["expansion_cohort_status"] == "EXCLUDED" for row in selected):
        raise SystemExit("excluded row present in selected manifest")
    return manifest


def pilot_artifact() -> None:
    manifest = validate_manifest()
    selected = read_csv(ART / "expansion_selected_candidates.v1.csv")[: int(manifest["pilot_size"])]
    gpu_rows: list[dict] = []
    nvidia_snapshot = ""
    cuda_version = ""
    try:
        nvidia_snapshot = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
        cuda_match = re.search(r"CUDA Version:\s*([^\s|]+)", nvidia_snapshot)
        cuda_version = cuda_match.group(1) if cuda_match else ""
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT)
        for line in output.splitlines():
            index, name, memory, driver = [part.strip() for part in line.split(",", 3)]
            gpu_rows.append({"gpu_index": index, "model": name, "memory": memory, "driver": driver})
    except (OSError, subprocess.CalledProcessError):
        pass
    result = {
        "schema_version": "phase6d.pilot_results.v1",
        "created_at_utc": utc_now(),
        "manifest_sha256": sha256_file(ART / "expansion_candidate_manifest.v1.json"),
        "pilot_candidates": [row["polymer_id"] for row in selected],
        "temperatures_K": list(TEMPERATURES),
        "gpu_inventory": gpu_rows,
        "nvidia_smi_snapshot": nvidia_snapshot,
        "cuda_version": cuda_version,
        "gpu_assignments_verified": len(gpu_rows) >= 4,
        "pilot_status": "PLANNED_NOT_EXECUTED",
        "simulation_launched": False,
        "reason": "Production execution requires the validated mace-lammps environment and an operator-approved GPU launch.",
        "throughput": {"simulations_per_gpu_day": None, "wall_time_seconds": None, "storage_bytes": None, "failure_rate": None},
        "checkpoint_resume_test": "NOT_RUN",
        "final_qc_test": "NOT_RUN",
        "safe_batch_size": 0,
    }
    write_json(ART / "expansion_pilot_results.v1.json", result)
    write_json(ART / "expansion_batch_plan.v1.json", {
        "schema_version": "phase6d.batch_plan.v1",
        "created_at_utc": utc_now(),
        "manifest_sha256": sha256_file(ART / "expansion_candidate_manifest.v1.json"),
        "pilot_results_sha256": sha256_file(ART / "expansion_pilot_results.v1.json"),
        "pilot_required_before_production": True,
        "pilot_status": result["pilot_status"],
        "max_concurrent_gpu_processes": min(4, len(gpu_rows)),
        "selected_candidates": len(read_csv(ART / "expansion_selected_candidates.v1.csv")),
        "production_batch_size": 0,
        "storage_budget_bytes": None,
        "execution_blocked_until_pilot_pass": True,
    })
    (REPORTS / "phase6d_pilot_and_capacity_report.md").write_text(
        "# Phase 6D pilot and capacity report\n\n"
        "The pilot manifest is frozen and contains 8 deterministic candidates. "
        "No simulation was launched because the validated pilot must run in the "
        "mace-lammps environment with four explicitly inventoried GPUs. The "
        "machine-readable result records this state as `PLANNED_NOT_EXECUTED`; "
        "production batch size is therefore zero. Checkpoint/resume, wall time, "
        "storage, failure rate and final QC remain pending the pilot.\n",
        encoding="utf-8",
    )


def make_registry() -> None:
    validate_manifest()
    selected = read_csv(ART / "expansion_selected_candidates.v1.csv")
    manifest_hash = sha256_file(ART / "expansion_candidate_manifest.v1.json")
    commit_hash = commit()
    rows = []
    for candidate in selected:
        for temperature in TEMPERATURES:
            rows.append({
                "polymer_id": candidate["polymer_id"], "cohort_type": "EXPANSION",
                "canonical_psmiles": candidate["canonical_psmiles"], "structure_hash": candidate["structure_hash"],
                "temperature_K": temperature, "temperature_completion": "MISSING",
                "simulation_status": "PARTIAL", "qc_status": "NOT_RUN", "descriptor_status": "NOT_RUN",
                "input_hash": "", "output_hash": "", "trajectory_hash": "", "manifest_hash": manifest_hash,
                "code_commit": commit_hash, "environment_fingerprint": "", "gpu_id": "", "hostname": "", "pid": "",
                "attempt": 0, "failure_class": "", "retryable": "", "provenance": "",
            })
    write_csv(ART / "bulk_md_expansion_registry.v1.csv", rows, REGISTRY_FIELDS)
    write_json(ART / "bulk_md_expansion_manifest.v1.json", {
        "schema_version": "phase6d.bulk_md_expansion_manifest.v1",
        "created_at_utc": utc_now(),
        "append_only": True,
        "cohort_type": "EXPANSION",
        "candidate_manifest_sha256": manifest_hash,
        "registry_sha256": sha256_file(ART / "bulk_md_expansion_registry.v1.csv"),
        "registry_path": "phase6d_artifacts/bulk_md_expansion_registry.v1.csv",
        "temperatures_K": list(TEMPERATURES),
        "candidate_count": len(selected),
        "polymer_temperature_rows": len(rows),
        "locked_holdout_simulation": False,
        "descriptor_table_path": "phase6d_artifacts/bulk_md_expansion_descriptors.v1.csv",
        "descriptor_schema_path": "phase6d_artifacts/bulk_md_expansion_descriptor_schema.v1.json",
        "raw_data_policy": "raw trajectories, restarts and large generated outputs remain outside Git",
    })
    write_csv(ART / "bulk_md_expansion_descriptors.v1.csv", [], DESCRIPTOR_FIELDS)
    write_json(ART / "bulk_md_expansion_descriptor_schema.v1.json", {
        "schema_version": "phase6d.bulk_md_descriptor.v1",
        "row_granularity": "polymer-temperature",
        "temperatures_K": list(TEMPERATURES),
        "units": {"density": "workflow documented units", "volume": "workflow documented units", "temperature_mean": "K", "equilibration_time_ps": "ps", "valid_frame_count": "count"},
        "derived_formulas": {
            "density_400_minus_300": "density[400 K] - density[300 K]",
            "density_500_minus_400": "density[500 K] - density[400 K]",
            "density_600_minus_500": "density[600 K] - density[500 K]",
            "density_600_minus_300": "density[600 K] - density[300 K]",
            "density_temperature_slope": "OLS slope(density, temperature_K)",
            "volume_400_minus_300": "volume[400 K] - volume[300 K]",
            "volume_500_minus_400": "volume[500 K] - volume[400 K]",
            "volume_600_minus_500": "volume[600 K] - volume[500 K]",
            "volume_600_minus_300": "volume[600 K] - volume[300 K]",
            "volume_temperature_slope": "OLS slope(volume, temperature_K)",
            "mobility_temperature_slope": "OLS slope(mobility, temperature_K) when present",
            "torsion_temperature_slope": "OLS slope(torsion, temperature_K) when present",
        },
        "missingness": "A true field mask is emitted for each unavailable source value.",
        "unsupported_descriptors": ["radius_of_gyration", "end_to_end_distance", "mobility", "torsion"],
    })


def _slope(points: dict[int, float]) -> str:
    if len(points) < 2:
        return ""
    xs, ys = zip(*sorted(points.items()))
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    return str(sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator) if denominator else ""


def extract_descriptors() -> None:
    """Extract only machine-readable QC fields that actually exist on disk."""
    validate_manifest()
    registry = read_csv(ART / "bulk_md_expansion_registry.v1.csv")
    records: dict[tuple[str, int], dict] = {}
    for row in registry:
        if row.get("qc_status") not in {"PASS", "WARN", "COMPLETE_QC_PASS", "COMPLETE_QC_WARN"} or not row.get("provenance"):
            continue
        qc_path = ROOT / row["provenance"]
        if qc_path.is_dir():
            qc_path = qc_path / "qc.json"
        if not qc_path.exists():
            continue
        try:
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            temperature = int(row["temperature_K"])
            records[(row["polymer_id"], temperature)] = {
                "polymer_id": row["polymer_id"], "temperature_K": temperature,
                "density": qc.get("density", ""), "volume": qc.get("volume", ""),
                "temperature_mean": qc.get("temperature_mean", ""), "equilibration_time_ps": qc.get("equilibration_time_ps", ""),
                "density_drift": qc.get("density_drift", ""), "volume_drift": qc.get("volume_drift", ""),
                "potential_energy_drift": qc.get("potential_energy_drift", ""),
                "density_half_trajectory_difference": qc.get("density_half_trajectory_difference", ""),
                "valid_frame_count": qc.get("valid_frame_count", ""), "qc_flags": ";".join(qc.get("qc_flags", [])) if isinstance(qc.get("qc_flags", []), list) else qc.get("qc_flags", ""),
                "missingness_mask": json.dumps(qc.get("missingness_mask", {}), sort_keys=True),
                "source_file": str(qc_path.relative_to(ROOT)), "source_hash": sha256_file(qc_path),
                "temperature_coverage": ";".join(str(t) for t in TEMPERATURES), "qc_status": row["qc_status"], "schema_version": "phase6d.bulk_md_descriptor.v1",
            }
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    by_polymer: dict[str, dict[int, dict]] = {}
    for (polymer_id, temperature), record in records.items():
        by_polymer.setdefault(polymer_id, {})[temperature] = record
    output: list[dict] = []
    for polymer_id, temperatures in by_polymer.items():
        densities = {t: float(r["density"]) for t, r in temperatures.items() if str(r["density"]) != ""}
        volumes = {t: float(r["volume"]) for t, r in temperatures.items() if str(r["volume"]) != ""}
        derived = {
            "density_400_minus_300": densities.get(400, "") - densities.get(300, "") if 300 in densities and 400 in densities else "",
            "density_500_minus_400": densities.get(500, "") - densities.get(400, "") if 400 in densities and 500 in densities else "",
            "density_600_minus_500": densities.get(600, "") - densities.get(500, "") if 500 in densities and 600 in densities else "",
            "density_600_minus_300": densities.get(600, "") - densities.get(300, "") if 300 in densities and 600 in densities else "",
            "density_temperature_slope": _slope(densities),
            "volume_400_minus_300": volumes.get(400, "") - volumes.get(300, "") if 300 in volumes and 400 in volumes else "",
            "volume_500_minus_400": volumes.get(500, "") - volumes.get(400, "") if 400 in volumes and 500 in volumes else "",
            "volume_600_minus_500": volumes.get(600, "") - volumes.get(500, "") if 500 in volumes and 600 in volumes else "",
            "volume_600_minus_300": volumes.get(600, "") - volumes.get(300, "") if 300 in volumes and 600 in volumes else "",
            "volume_temperature_slope": _slope(volumes), "mobility_temperature_slope": "", "torsion_temperature_slope": "",
        }
        for record in temperatures.values():
            record.update(derived); output.append(record)
    write_csv(ART / "bulk_md_expansion_descriptors.v1.csv", output, DESCRIPTOR_FIELDS)


def integrity_audit() -> None:
    validate_manifest()
    registry = read_csv(ART / "bulk_md_expansion_registry.v1.csv")
    descriptors = read_csv(ART / "bulk_md_expansion_descriptors.v1.csv") if (ART / "bulk_md_expansion_descriptors.v1.csv").exists() else []
    selected = read_csv(ART / "expansion_selected_candidates.v1.csv")
    expansion_manifest_path = ART / "bulk_md_expansion_manifest.v1.json"
    expansion_manifest = json.loads(expansion_manifest_path.read_text(encoding="utf-8")) if expansion_manifest_path.exists() else {}
    locked = {row["polymer_id"] for row in read_csv(COHORT) if row.get("split") == "LOCKED_HOLDOUT"}
    pairs = [(row["polymer_id"], row["temperature_K"]) for row in registry]
    duplicate_pairs = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    selected_ids = {row["polymer_id"] for row in selected}
    errors = {
        "locked_holdout_simulated": sorted(locked & selected_ids),
        "duplicate_polymer_temperature_pairs": duplicate_pairs,
        "wrong_identity_rows": sorted({row["polymer_id"] for row in registry} - selected_ids),
        "descriptor_rows_without_qc": sorted({(row["polymer_id"], row["temperature_K"]) for row in descriptors} - {(row["polymer_id"], row["temperature_K"]) for row in registry if row["qc_status"] == "COMPLETE_QC_PASS"}),
        "qc_rows_without_descriptor": sorted({(row["polymer_id"], row["temperature_K"]) for row in registry if row["qc_status"] == "COMPLETE_QC_PASS"} - {(row["polymer_id"], row["temperature_K"]) for row in descriptors}),
        "missing_explicit_status": sorted(row["polymer_id"] for row in registry if not row["simulation_status"] or not row["qc_status"]),
        "manifest_hash_mismatch": [] if expansion_manifest.get("candidate_manifest_sha256") == sha256_file(ART / "expansion_candidate_manifest.v1.json") and expansion_manifest.get("registry_sha256") == sha256_file(ART / "bulk_md_expansion_registry.v1.csv") else ["bulk_md_expansion_manifest.v1.json"],
    }
    passed = not any(errors.values()) and all(row["cohort_type"] == "EXPANSION" for row in registry)
    audit = {
        "schema_version": "phase6d.integrity_audit.v1",
        "created_at_utc": utc_now(),
        "manifest_sha256": sha256_file(ART / "expansion_candidate_manifest.v1.json"),
        "registry_sha256": sha256_file(ART / "bulk_md_expansion_registry.v1.csv"),
        "candidate_count": len(selected), "registry_rows": len(registry),
        "expected_registry_rows": len(selected) * len(TEMPERATURES),
        "errors": errors, "status": "PASS" if passed else "FAIL",
        "locked_holdout_exclusion": "PASS" if not errors["locked_holdout_simulated"] else "FAIL",
        "descriptor_provenance_required": True,
    }
    write_json(ART / "bulk_md_expansion_integrity_audit.v1.json", audit)
    (REPORTS / "phase6d_expansion_integrity_audit.md").write_text(
        "# Phase 6D expansion integrity audit\n\n" + json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (REPORTS / "phase6d_expansion_qc_report.md").write_text(
        "# Phase 6D expansion QC report\n\n"
        "The append-only registry records explicit simulation, QC and descriptor statuses. No descriptor is admitted until a temperature-specific "
        "QC pass supplies source hashes and provenance. The registry currently has "
        f"{len(selected)} candidates and {len(registry)} polymer-temperature rows, with "
        f"{sum(row['simulation_status'] == 'COMPLETE_QC_PASS' for row in registry)} complete pilot rows.\n\n" + json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def environment_fingerprint(gpu: str) -> str:
    values = {
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "gpu": gpu,
    }
    return sha256_bytes(json.dumps(values, sort_keys=True).encode())


def run_scheduler(args: argparse.Namespace) -> None:
    """Run a validated command template with one process per GPU.

    The command template receives `{input}` and `{polymer_id}`. A production
    invocation must point at an actual input file; literal placeholders are
    rejected. This command is intentionally opt-in and is never called by
    ``select``, ``pilot`` or ``registry``.
    """
    if args.retries < 0:
        raise SystemExit("--retries must be nonnegative")
    validate_manifest()
    pilot = json.loads((ART / "expansion_pilot_results.v1.json").read_text())
    if pilot.get("pilot_status") != "PASS":
        raise SystemExit("scheduler blocked: pilot must pass before production")
    if "your_input.in" in args.command or "{input}" not in args.command:
        raise SystemExit("command must use a concrete {input} path and no literal placeholder")
    selected = read_csv(ART / "expansion_selected_candidates.v1.csv")
    queue_path = ART / "queue_state.v1.json"
    lock_path = ART / "scheduler.lock"
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise SystemExit("another Phase 6D scheduler is already running") from exc
    rows = read_csv(ART / "bulk_md_expansion_registry.v1.csv")
    queue = {"state": "RUNNING", "created_at_utc": utc_now(), "jobs": [], "max_retries": args.retries, "gpu_count": len(args.gpus)}
    registry_by_pair = {(row["polymer_id"], row["temperature_K"]): row for row in rows}
    tasks = [(candidate, temperature, args.gpus[index % len(args.gpus)]) for index, (candidate, temperature) in enumerate((c, t) for c in selected for t in TEMPERATURES)]

    def execute(task: tuple[dict, int, str]) -> dict:
        candidate, temperature, gpu = task
        input_path = args.input_root / candidate["polymer_id"] / f"{temperature}K" / "input.in"
        if not input_path.is_file():
            return {"candidate": candidate, "temperature": temperature, "gpu": gpu, "error": f"missing concrete input: {input_path}"}
        job_dir = input_path.parent
        stdout_path, stderr_path = job_dir / "stdout.log", job_dir / "stderr.log"
        command = args.command.format(input=str(input_path), polymer_id=candidate["polymer_id"])
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
        attempts: list[dict] = []
        result = 1
        for attempt in range(1, args.retries + 2):
            started = utc_now(); start_time = time.time(); timed_out = False
            previous_checkpoint = job_dir / "checkpoint.restart"
            write_json(job_dir / f"checkpoint_metadata.attempt{attempt}.json", {
                "polymer_id": candidate["polymer_id"], "temperature_K": temperature,
                "simulation_stage": "workflow_input", "timestep_fs": 1,
                "simulation_time_ps": "workflow-defined", "integrator": "LAMMPS fix npt",
                "thermostat": "Nose-Hoover temperature damping 100 fs",
                "barostat": "Nose-Hoover isotropic pressure damping 1000 fs",
                "force_field": "GAFF2_mod", "random_seed": "input-declared",
                "input_file_hash": sha256_file(input_path), "code_commit": commit(),
                "environment_fingerprint": environment_fingerprint(gpu), "gpu_identifier": gpu,
                "previous_checkpoint_hash": sha256_file(previous_checkpoint) if previous_checkpoint.exists() else "",
                "attempt": attempt, "created_at_utc": started,
            })
            with stdout_path.open("a") as stdout, stderr_path.open("a") as stderr:
                process = subprocess.Popen(shlex.split(command), cwd=job_dir, env=env, stdout=stdout, stderr=stderr)
                try:
                    result = process.wait(timeout=args.timeout_seconds or None)
                except subprocess.TimeoutExpired:
                    timed_out = True; process.kill(); result = 124; process.wait()
            ended = utc_now(); failure_class = "" if result == 0 else ("TIMEOUT" if timed_out else "PROCESS_FAILURE")
            attempts.append({"polymer_id": candidate["polymer_id"], "temperature_K": temperature, "gpu_id": gpu,
                             "hostname": socket.gethostname(), "pid": process.pid, "started_at_utc": started,
                             "ended_at_utc": ended, "wall_time_seconds": time.time() - start_time, "command": command,
                             "stdout": str(stdout_path), "stderr": str(stderr_path), "returncode": result,
                             "attempt": attempt, "failure_class": failure_class, "retryable": result != 0 and attempt <= args.retries,
                             "checkpoint_resume": attempt > 1, "environment_fingerprint": environment_fingerprint(gpu)})
            if result == 0:
                break
        return {"candidate": candidate, "temperature": temperature, "gpu": gpu, "result": result,
                "attempts": attempts, "input_path": input_path, "job_dir": job_dir}

    pending = list(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {executor.submit(execute, pending.pop(0)): True for _ in range(min(len(args.gpus), len(pending)))}
        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result = future.result(); futures.pop(future)
                if "error" in result:
                    raise SystemExit(result["error"])
                queue["jobs"].extend(result["attempts"]); write_json(queue_path, queue)
                candidate = result["candidate"]; temperature = str(result["temperature"]); registry_row = registry_by_pair[(candidate["polymer_id"], temperature)]
                last = result["attempts"][-1]
                registry_row.update({"temperature_completion": "COMPLETE" if result["result"] == 0 else "FAILED_FINAL",
                                     "simulation_status": "COMPLETE" if result["result"] == 0 else "FAILED_FINAL",
                                     "qc_status": "PENDING_QC" if result["result"] == 0 else "NOT_RUN",
                                     "input_hash": sha256_file(result["input_path"]), "gpu_id": result["gpu"],
                                     "hostname": last["hostname"], "pid": last["pid"], "attempt": last["attempt"],
                                     "failure_class": last["failure_class"], "retryable": str(last["retryable"]).lower(),
                                     "environment_fingerprint": last["environment_fingerprint"],
                                     "provenance": str(result["job_dir"].relative_to(ROOT))})
                write_csv(ART / "bulk_md_expansion_registry.v1.csv", rows, REGISTRY_FIELDS)
                if result["result"]:
                    raise SystemExit(f"job failed after {last['attempt']} attempts for {candidate['polymer_id']} at {temperature} K")
                if pending:
                    futures[executor.submit(execute, pending.pop(0))] = True
    queue["state"] = "COMPLETE"; write_json(queue_path, queue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    select = sub.add_parser("select"); select.add_argument("--pool-size", type=int, default=128)
    sub.add_parser("validate")
    sub.add_parser("pilot")
    sub.add_parser("registry")
    sub.add_parser("extract")
    sub.add_parser("integrity")
    run = sub.add_parser("run")
    run.add_argument("--command", required=True, help="command template containing {input} and {polymer_id}")
    run.add_argument("--input-root", type=Path, required=True)
    run.add_argument("--gpus", nargs="+", required=True)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--timeout-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.command == "audit": audit_workflow()
    elif args.command == "select": select_candidates(args.pool_size)
    elif args.command == "validate": validate_manifest(); print("PASS")
    elif args.command == "pilot": pilot_artifact()
    elif args.command == "registry": make_registry()
    elif args.command == "extract": extract_descriptors()
    elif args.command == "integrity": integrity_audit()
    elif args.command == "run": run_scheduler(args)


if __name__ == "__main__":
    main()
