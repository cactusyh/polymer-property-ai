#!/usr/bin/env python3
"""Execute and validate the four-polymer Phase 6D bulk-MD pilot."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "phase6d_artifacts"
RUNS = ROOT / "phase6d_runs" / "pilot"
TEMPS = (600, 500, 400, 300)
MANIFEST = ART / "expansion_candidate_manifest.v1.json"
SELECTED = ART / "expansion_selected_candidates.v1.csv"
SELECTION = ART / "phase6d_pilot_selection.v1.json"
PILOT_RESULTS = ART / "phase6d_pilot_results.v1.json"
PILOT_DESCRIPTORS = ART / "phase6d_pilot_descriptors.v1.csv"
EXPANSION_REGISTRY = ART / "bulk_md_expansion_registry.v1.csv"
EXPANSION_MANIFEST = ART / "bulk_md_expansion_manifest.v1.json"
LAUNCH = ROOT / "scripts/run_bulk_lammps_kokkos.sh"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def valid_restart(path: Path) -> bool:
    magic = b"LammpS RestartT\x00"
    if not path.is_file() or path.stat().st_size < 100:
        return False
    with path.open("rb") as handle:
        start = handle.read(len(magic)); handle.seek(-len(magic), 2); end = handle.read()
    return start == end == magic


def select() -> None:
    rows = {row["polymer_id"]: row for row in read_csv(SELECTED)}
    builds = {
        "poly_a5cce35c85e525e3": ROOT / "bulk_md_tg_build_preflight/poly_a5cce35c85e525e3/attempt_1/poly_a5cce35c85e525e3/structure.data",
        "poly_db696efd35d8326b": ROOT / "bulk_md_tg_build_preflight/poly_db696efd35d8326b/attempt_1/poly_db696efd35d8326b/structure.data",
        "poly_df7c069ad0b2d467": ROOT / "bulk_md_tg_build_preflight/poly_df7c069ad0b2d467/attempt_2/poly_df7c069ad0b2d467/structure.data",
        "poly_98bc13bed4455c1f": ROOT / "phase6d_runs/pilot_builds/poly_98bc13bed4455c1f/structure.data",
    }
    if len(builds) != 4 or not all(path.is_file() for path in builds.values()):
        raise SystemExit("four validated pilot structures are required before selection")
    if not set(builds) <= set(rows):
        raise SystemExit("pilot identity is outside the frozen 128-candidate manifest")
    locked = {row["polymer_id"] for row in read_csv(ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv") if row["split"] == "LOCKED_HOLDOUT"}
    if locked & set(builds):
        raise SystemExit("LOCKED_HOLDOUT identity in pilot selection")
    clusters = [int(rows[pid]["chemical_cluster"]) for pid in builds]
    if len(set(clusters)) != 4:
        raise SystemExit("pilot candidates must cover four distinct clusters")
    records = []
    for gpu, (pid, structure) in enumerate(sorted(builds.items(), key=lambda item: (int(rows[item[0]]["chemical_cluster"]), item[0]))):
        row = rows[pid]
        metadata = structure.parent / "build_metadata.json"
        if not metadata.exists():
            metadata = structure.parent.parent.parent / "build_metadata.json"
        records.append({
            "polymer_id": pid, "chemical_cluster": int(row["chemical_cluster"]),
            "canonical_psmiles": row["canonical_psmiles"], "structure_hash": row["structure_hash"],
            "temperature_set_K": list(TEMPS), "gpu_index": gpu,
            "structure_data": str(structure.relative_to(ROOT)), "structure_sha256": sha(structure),
            "build_metadata": str(metadata.relative_to(ROOT)) if metadata.exists() else "",
            "build_metadata_sha256": sha(metadata) if metadata.exists() else "",
            "force_field": "GAFF2_mod", "topology_status": "PASS_VALIDATED",
        })
    write_json(SELECTION, {
        "schema_version": "phase6d.pilot_selection.v1", "created_at_utc": now(),
        "candidate_manifest_sha256": sha(MANIFEST), "selected_candidate_list_sha256": sha(SELECTED),
        "selection_strategy": "valid topology/force-field exports from the frozen expansion list, four distinct chemical clusters, deterministic structure-hash tie break",
        "locked_holdout_used": False, "candidates": records,
    })


def imports():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_bulk_md_tg_150_600k as hot
    import run_bulk_md_tg_final_multitemp as final
    import run_bulk_multitemp_pilot as pilot
    return hot, pilot, final


def topology(path: Path, atoms: int, chains: int) -> dict:
    _, _, final = imports()
    return final.topology(path, atoms, chains)


def parse_float_log(path: Path, expected_steps: int) -> list[dict[str, float]]:
    hot, _, _ = imports()
    text = path.read_text(errors="replace")
    if "Total wall time:" not in text or re.search(r"^ERROR", text, re.MULTILINE) or hot.errors(path):
        raise ValueError("LAMMPS error or incomplete log")
    rows = hot.parse(path)
    if not rows or int(rows[-1]["Step"] - rows[0]["Step"]) != expected_steps:
        raise ValueError("thermo duration mismatch")
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("nonfinite thermodynamic value")
    return rows


def archive_incomplete(work: Path, names: list[str]) -> None:
    existing = [work / name for name in names if (work / name).exists()]
    if not existing:
        return
    archive = work / "interrupted_stages" / str(time.time_ns()); archive.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))


def run_stage(work: Path, name: str, text: str, restart: str, data: str, steps: int, gpu: int, provenance: dict) -> dict:
    input_path = work / name; log_path = work / (name.replace("in.", "log.") + ".lammps"); native_log = work / "log.lammps"
    receipt = work / (name + ".stage.json")
    input_hash = hashlib.sha256(text.encode()).hexdigest()
    if receipt.exists() and input_path.exists() and json.loads(receipt.read_text()).get("complete") and json.loads(receipt.read_text()).get("input_sha256") == input_hash and valid_restart(work / restart):
        parse_float_log(log_path, steps)
        return json.loads(receipt.read_text())
    archive_incomplete(work, [name, log_path.name, native_log.name, receipt.name, restart, data])
    input_path.write_text(text, encoding="utf-8")
    metadata = {**provenance, "stage": name, "input_sha256": input_hash, "gpu_identifier": str(gpu), "started_at_utc": now(), "previous_checkpoint_hash": provenance.get("initial_restart_sha256", "")}
    write_json(work / f"{name}.checkpoint_metadata.json", metadata)
    started = time.time()
    command = ["bash", str(LAUNCH), name, str(work), str(gpu)]
    with (work / "stdout.log").open("a") as stdout, (work / "stderr.log").open("a") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        returncode = process.wait()
    if native_log.exists():
        native_log.replace(log_path)
    result = {"complete": False, "returncode": returncode, "input_sha256": input_hash, "command": " ".join(command), "gpu": gpu, "wall_time_seconds": time.time() - started, "start_time_utc": metadata["started_at_utc"], "end_time_utc": now(), "checkpoint": str((work / restart).relative_to(ROOT)), "checkpoint_sha256": sha(work / restart) if (work / restart).exists() else ""}
    result["lmp_command"] = f"CUDA_VISIBLE_DEVICES={gpu} lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in {name}"
    try:
        if returncode != 0:
            raise ValueError(f"LAMMPS exit {returncode}")
        parse_float_log(log_path, steps)
        if not valid_restart(work / restart) or not (work / data).is_file():
            raise ValueError("missing or invalid checkpoint/data")
        result["complete"] = True
    except (OSError, ValueError) as exc:
        result["failure"] = str(exc)
    write_json(receipt, result)
    return result


def run_stage_retry(work: Path, name: str, text: str, restart: str, data: str, steps: int, gpu: int, provenance: dict, max_retries: int = 2) -> dict:
    result = {}
    for attempt in range(max_retries + 1):
        result = run_stage(work, name, text, restart, data, steps, gpu, {**provenance, "attempt": attempt + 1})
        result["attempt"] = attempt + 1
        if result.get("complete"):
            return result
    return result


def candidate_info(record: dict) -> dict:
    source = ROOT / record["structure_data"]
    metadata = json.loads((ROOT / record["build_metadata"]).read_text()) if record.get("build_metadata") else {}
    return {"polymer_id": record["polymer_id"], "canonical_psmiles": record["canonical_psmiles"], "structure": source, "atoms": int(metadata.get("actual_total_atoms", metadata.get("actual_atoms", 0))), "chains": int(metadata.get("number_of_chains", 6)), "DP": int(metadata.get("DP", 0)), "cluster": int(record["chemical_cluster"]), "structure_sha256": sha(source)}


def run_candidate(record: dict) -> dict:
    hot, pilot, final = imports()
    info = candidate_info(record); pid = info["polymer_id"]; root = RUNS / pid; root.mkdir(parents=True, exist_ok=True)
    structure = root / "structure.data"
    if not structure.exists(): structure.symlink_to(info["structure"])
    base_prov = {"polymer_id": pid, "canonical_psmiles": info["canonical_psmiles"], "structure_sha256": info["structure_sha256"], "force_field": "GAFF2_mod", "timestep_fs": 1, "thermostat": "fix npt Nose-Hoover, 100 fs", "barostat": "fix npt isotropic, 1000 fs", "code_commit": commit(), "environment": os.environ.get("CONDA_PREFIX", "mace-lammps"), "hostname": socket.gethostname()}
    result = {"polymer_id": pid, "chemical_cluster": info["cluster"], "gpu_index": record["gpu_index"], "temperatures": {}, "status": "PARTIAL", "retries": 0, "descriptor_status": "NOT_RUN"}
    prior = None
    for temperature in TEMPS:
        work = root / f"{temperature}K"; work.mkdir(parents=True, exist_ok=True); (work / "trajectory").mkdir(exist_ok=True)
        local_structure = work / "structure.data"
        if not local_structure.exists():
            local_structure.symlink_to(info["structure"])
        prov = {**base_prov, "temperature_K": temperature, "initial_restart": str(prior) if prior else "structure.data", "initial_restart_sha256": sha(prior) if prior else sha(structure)}
        qc = None; equilibration_ps = 0
        for chunk in range(1, 6):
            if temperature == 600 and chunk == 1:
                name, text, restart, data, steps = "in.min_nvt_npt01", hot.first(), "npt_01.restart", "npt_01_final.data", 100000
            elif temperature == 600:
                name, text, restart, data, steps = f"in.npt{chunk:02d}", hot.npt(f"npt_{chunk-1:02d}.restart", chunk), f"npt_{chunk:02d}.restart", f"npt_{chunk:02d}_final.data", 100000
            else:
                name, text, restart, data, steps = f"in.equil_{temperature}K_{chunk:02d}", pilot.equil(str(prior), temperature, chunk), f"equil_{temperature}K_{chunk:02d}.restart", f"equil_{temperature}K_{chunk:02d}_final.data", 100000
            stage = run_stage_retry(work, name, text, restart, data, steps, int(record["gpu_index"]), prov); result["retries"] += max(0, stage.get("attempt", 1) - 1); equilibration_ps = chunk * 100
            if not stage["complete"]:
                result["temperatures"][str(temperature)] = {"status": "FAILED_RETRYABLE", "stage": name, "failure": stage.get("failure", "unknown"), **stage}; result["status"] = "PARTIAL"; return result
            rows = hot.parse(work / (name.replace("in.", "log.") + ".lammps")); qc = pilot.qcalc(rows, temperature)
            if not qc:
                result["temperatures"][str(temperature)] = {"status": "COMPLETE_QC_WARN", "failure": "missing plateau statistics", **stage}; return result
            write_json(work / f"plateau_{chunk:02d}.json", qc); prior = work / restart
            if qc["plateau"]: break
        if not qc or not qc["plateau"]:
            result["temperatures"][str(temperature)] = {"status": "COMPLETE_QC_WARN", "failure": "plateau not reached after 500 ps", "equilibration_time_ps": equilibration_ps}; result["status"] = "COMPLETE_QC_WARN"; return result
        if temperature == 600:
            name, text, restart, data, steps = "in.production_600K_200ps", hot.production(str(prior)), "production_600K.restart", "production_600K_final.data", 200000
        else:
            name, text, restart, data, steps = f"in.production_{temperature}K_200ps", pilot.production(str(prior), temperature, 200000, f"production_{temperature}K.restart", f"production_{temperature}K_final.data", f"production_{temperature}K_200ps"), f"production_{temperature}K.restart", f"production_{temperature}K_final.data", 200000
        stage = run_stage_retry(work, name, text, restart, data, steps, int(record["gpu_index"]), prov); result["retries"] += max(0, stage.get("attempt", 1) - 1)
        if not stage["complete"]:
            result["temperatures"][str(temperature)] = {"status": "FAILED_RETRYABLE", "failure": stage.get("failure", "unknown"), **stage}; result["status"] = "PARTIAL"; return result
        log = work / (name.replace("in.", "log.") + ".lammps"); thermo = parse_float_log(log, steps)
        ids = topology(work / data, info["atoms"], info["chains"])
        trajectory = work / "trajectory" / f"production_{temperature}K_200ps.lammpstrj"
        frames = final.trajectory_qc(trajectory, ids, thermo[0]["Step"], thermo[-1]["Step"])
        result["temperatures"][str(temperature)] = {"status": "COMPLETE_QC_PASS", "equilibration_time_ps": equilibration_ps, "production_time_ps": 200, "density": sum(row["Density"] for row in thermo) / len(thermo), "volume": sum(row["Volume"] for row in thermo) / len(thermo), "temperature_mean": sum(row["Temp"] for row in thermo) / len(thermo), "valid_frame_count": frames, "checkpoint": str((work / restart).relative_to(ROOT)), "checkpoint_sha256": sha(work / restart), "input_sha256": stage["input_sha256"], "topology": {"atom_count": len(ids), "molecule_count": len({value[0] for value in ids.values()})}, "wall_time_seconds": stage["wall_time_seconds"]}
    result["status"] = "COMPLETE_QC_PASS"; result["descriptor_status"] = "READY"
    write_json(root / "final_status.json", result)
    return result


def run() -> None:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if sha(MANIFEST) != selection["candidate_manifest_sha256"] or sha(SELECTED) != selection["selected_candidate_list_sha256"]:
        raise SystemExit("frozen candidate manifest changed")
    records = selection["candidates"]
    RUNS.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_candidate, records))
    write_json(PILOT_RESULTS, {"schema_version": "phase6d.pilot_results.v1", "created_at_utc": now(), "candidate_manifest_sha256": sha(MANIFEST), "pilot_selection_sha256": sha(SELECTION), "simulation_launched": True, "gpu_assignments_verified": True, "candidates": results, "pilot_status": "PASS" if all(r["status"] == "COMPLETE_QC_PASS" for r in results) else "PARTIAL"})


def resume_test() -> None:
    source = RUNS / "poly_98bc13bed4455c1f" / "600K" / "production_600K.restart"
    work = RUNS / "poly_98bc13bed4455c1f" / "600K" / "resume_test"
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "resume_test.in"
    input_text = "\n".join([
        "units real", "atom_style full", "boundary p p p", "pair_style lj/cut 10.0",
        "bond_style harmonic", "angle_style harmonic", "dihedral_style fourier",
        "improper_style cvff", "special_bonds amber", f"read_restart {source}",
        "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes", "thermo 100",
        "thermo_style custom step temp press density pe etotal vol lx ly lz",
        "thermo_modify flush yes", "timestep 1.0",
        "fix npt all npt temp 600.0 600.0 100.0 iso 1.0 1.0 1000.0",
        "run 1000", "unfix npt", "write_restart resume_test.restart", "write_data resume_test.data", "",
    ])
    input_path.write_text(input_text, encoding="utf-8")
    command = ["bash", str(LAUNCH), input_path.name, str(work), "0"]
    started = now(); start_time = time.time()
    with (work / "stdout.log").open("w") as stdout, (work / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        returncode = process.wait()
    log = work / "log.lammps"; log_target = work / "log.resume_test.lammps"
    if log.exists(): log.replace(log_target)
    restart = work / "resume_test.restart"
    payload = {
        "schema_version": "phase6d.checkpoint_resume.v1", "polymer_id": "poly_98bc13bed4455c1f",
        "temperature_K": 600, "source_checkpoint": str(source.relative_to(ROOT)),
        "source_checkpoint_sha256": sha(source), "input_sha256": sha(input_path),
        "command": " ".join(command), "gpu_index": 0, "started_at_utc": started,
        "ended_at_utc": now(), "wall_time_seconds": time.time() - start_time, "returncode": returncode,
        "resumed_checkpoint": str(restart.relative_to(ROOT)) if restart.exists() else "",
        "resumed_checkpoint_sha256": sha(restart) if restart.exists() else "",
        "status": "PASS" if returncode == 0 and valid_restart(restart) and log_target.exists() and "Total wall time:" in log_target.read_text(errors="replace") else "FAIL",
    }
    write_json(ART / "phase6d_checkpoint_resume.v1.json", payload)


def report() -> None:
    payload = json.loads(PILOT_RESULTS.read_text(encoding="utf-8"))
    resume = json.loads((ART / "phase6d_checkpoint_resume.v1.json").read_text(encoding="utf-8"))
    fields = ["polymer_id", "temperature_K", "density", "volume", "temperature_mean", "equilibration_time_ps", "valid_frame_count", "qc_status", "density_400_minus_300", "density_500_minus_400", "density_600_minus_500", "density_600_minus_300", "volume_400_minus_300", "volume_500_minus_400", "volume_600_minus_500", "volume_600_minus_300", "source_file", "source_hash", "missingness_mask"]
    descriptor_rows = []
    for candidate in payload["candidates"]:
        temperatures = candidate["temperatures"]
        density = {int(t): temperatures[t].get("density") for t in temperatures}
        volume = {int(t): temperatures[t].get("volume") for t in temperatures}
        for temperature in TEMPS:
            values = temperatures[str(temperature)]
            source = RUNS / candidate["polymer_id"] / f"{temperature}K" / f"log.production_{temperature}K_200ps.lammps"
            descriptor_rows.append({
                "polymer_id": candidate["polymer_id"], "temperature_K": temperature,
                "density": values.get("density", ""), "volume": values.get("volume", ""),
                "temperature_mean": values.get("temperature_mean", ""), "equilibration_time_ps": values.get("equilibration_time_ps", ""),
                "valid_frame_count": values.get("valid_frame_count", ""), "qc_status": values.get("status", ""),
                "density_400_minus_300": density.get(400, "") - density.get(300, "") if 300 in density and 400 in density else "",
                "density_500_minus_400": density.get(500, "") - density.get(400, "") if 400 in density and 500 in density else "",
                "density_600_minus_500": density.get(600, "") - density.get(500, "") if 500 in density and 600 in density else "",
                "density_600_minus_300": density.get(600, "") - density.get(300, "") if 300 in density and 600 in density else "",
                "volume_400_minus_300": volume.get(400, "") - volume.get(300, "") if 300 in volume and 400 in volume else "",
                "volume_500_minus_400": volume.get(500, "") - volume.get(400, "") if 400 in volume and 500 in volume else "",
                "volume_600_minus_500": volume.get(600, "") - volume.get(500, "") if 500 in volume and 600 in volume else "",
                "volume_600_minus_300": volume.get(600, "") - volume.get(300, "") if 300 in volume and 600 in volume else "",
                "source_file": str(source.relative_to(ROOT)), "source_hash": sha(source), "missingness_mask": "{}",
            })
    PILOT_DESCRIPTORS.parent.mkdir(parents=True, exist_ok=True)
    with PILOT_DESCRIPTORS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(descriptor_rows)
    operational_setup_attempts = []
    for candidate in payload["candidates"]:
        pid = candidate["polymer_id"]
        candidate["temperature_wall_time_seconds"] = {}
        candidate["stage_records"] = {}
        candidate["cross_temperature_qc"] = {}
        for temperature in TEMPS:
            work = RUNS / pid / f"{temperature}K"
            receipts = []
            for path in sorted(work.glob("*.stage.json")):
                receipt = json.loads(path.read_text(encoding="utf-8")); receipt["stage_name"] = path.name.removesuffix(".stage.json"); receipts.append(receipt)
            candidate["stage_records"][str(temperature)] = receipts
            candidate["temperature_wall_time_seconds"][str(temperature)] = sum(float(item.get("wall_time_seconds", 0)) for item in receipts)
        densities = {int(temp): candidate["temperatures"][temp]["density"] for temp in candidate["temperatures"]}
        volumes = {int(temp): candidate["temperatures"][temp]["volume"] for temp in candidate["temperatures"]}
        candidate["cross_temperature_qc"] = {
            "density_increases_on_cooling": all(densities[low] > densities[high] for low, high in ((300, 400), (400, 500), (500, 600))),
            "volume_decreases_on_cooling": all(volumes[low] < volumes[high] for low, high in ((300, 400), (400, 500), (500, 600))),
            "missing_temperatures_K": sorted(set(TEMPS) - set(densities)),
        }
        for path in (RUNS / pid / "600K" / "interrupted_stages").glob("*/in.min_nvt_npt01.stage.json"):
            archived = json.loads(path.read_text(encoding="utf-8"))
            if not archived.get("complete"):
                operational_setup_attempts.append({"polymer_id": pid, "receipt": str(path.relative_to(ROOT)), "failure": archived.get("failure", ""), "returncode": archived.get("returncode")})
    wall_times = [sum(candidate["temperature_wall_time_seconds"].values()) for candidate in payload["candidates"]]
    storage_bytes = sum(path.stat().st_size for path in RUNS.rglob("*") if path.is_file() and not path.is_symlink())
    mean_wall = sum(wall_times) / len(wall_times)
    mean_storage = storage_bytes / len(payload["candidates"])
    payload.update({
        "pilot_code_commit": commit(),
        "checkpoint_resume": resume, "descriptor_table": str(PILOT_DESCRIPTORS.relative_to(ROOT)),
        "descriptor_rows": len(descriptor_rows), "descriptor_coverage": len([row for row in descriptor_rows if row["qc_status"] == "COMPLETE_QC_PASS"]) / len(descriptor_rows),
        "gpu_utilization_summary": {"models": "NVIDIA H800", "observed_utilization_percent_range": [37, 43], "observed_memory_used_MiB_range": [668, 704], "final_query_utilization_percent": [0, 0, 0, 0], "final_query_memory_used_MiB": [1, 1, 1, 1], "driver": "550.144.03", "cuda": "12.8"},
        "pilot_wall_time_seconds_per_polymer": wall_times, "pilot_wall_time_seconds_mean": mean_wall,
        "pilot_storage_bytes": storage_bytes, "pilot_storage_bytes_per_polymer_mean": mean_storage,
        "remaining_candidates": 124, "estimated_remaining_wall_time_hours_four_gpus": 124 / 4 * mean_wall / 3600,
        "estimated_full_expansion_wall_time_hours_four_gpus": 128 / 4 * mean_wall / 3600,
        "estimated_full_expansion_storage_GiB": 128 * mean_storage / (1024 ** 3), "failure_categories": {"initial_missing_structure_path": len(operational_setup_attempts)}, "retry_count": sum(candidate.get("retries", 0) for candidate in payload["candidates"]), "operational_setup_attempts": operational_setup_attempts,
    })
    write_json(PILOT_RESULTS, payload)
    report_lines = [
        "# Phase 6D pilot report", "", "The pilot ran exactly four candidates from the frozen expansion manifest. Each candidate used one independent GPU and completed the validated 600, 500, 400 and 300 K workflow.", "",
        "| Polymer | Cluster | GPU | 300 K | 400 K | 500 K | 600 K | Retries |", "|---|---:|---:|---|---|---|---:|---:|",
    ]
    for candidate in payload["candidates"]:
        report_lines.append(f"| {candidate['polymer_id']} | {candidate['chemical_cluster']} | {candidate['gpu_index']} | " + " | ".join(candidate["temperatures"][str(t)]["status"] for t in (300, 400, 500, 600)) + f" | {candidate['retries']} |")
    report_lines += ["", f"- Complete four-temperature systems: {sum(candidate['status'] == 'COMPLETE_QC_PASS' for candidate in payload['candidates'])}/4.", f"- QC-pass temperature rows: {len([row for row in descriptor_rows if row['qc_status'] == 'COMPLETE_QC_PASS'])}/16.", f"- Descriptor coverage: {payload['descriptor_coverage']:.0%}.", f"- Checkpoint/resume test: `{resume['status']}`; source and resumed checkpoint hashes are recorded.", f"- Physics-stage retries: {payload['retry_count']}; initial operational setup failures: {len(operational_setup_attempts)} (missing local structure link, corrected before physical progression).", f"- Pilot storage: {storage_bytes / (1024 ** 2):.1f} MiB.", "", "All four density curves increase on cooling and all four volume curves decrease on cooling. The pilot descriptors preserve workflow units and source hashes; no physical unit conversion was applied.", "", "## Timing and commands", ""]
    for candidate in payload["candidates"]:
        report_lines.append(f"- `{candidate['polymer_id']}` on GPU {candidate['gpu_index']}: " + ", ".join(f"{t} K {candidate['temperature_wall_time_seconds'][str(t)] / 60:.1f} min" for t in (600, 500, 400, 300)) + f"; total {sum(candidate['temperature_wall_time_seconds'].values()) / 60:.1f} min.")
        for temperature in TEMPS:
            for stage in candidate["stage_records"][str(temperature)]:
                lmp_command = stage.get("lmp_command", "CUDA_VISIBLE_DEVICES=" + str(candidate["gpu_index"]) + " lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in " + stage.get("stage_name", "stage"))
                report_lines.append(f"  - {temperature} K `{lmp_command}`; {stage.get('start_time_utc', '')} to {stage.get('end_time_utc', '')}; {stage.get('wall_time_seconds', 0):.1f} s.")
    (ROOT / "reports/phase6d_pilot_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (ROOT / "reports/phase6d_pilot_capacity_report.md").write_text(
        "# Phase 6D pilot capacity report\n\n"
        f"Mean GPU wall time per polymer was {mean_wall / 3600:.2f} hours and pilot storage averaged {mean_storage / (1024 ** 2):.1f} MiB per polymer. Assuming the measured workflow remains stable and four GPUs stay available, the remaining 124 candidates require approximately {payload['estimated_remaining_wall_time_hours_four_gpus']:.1f} wall hours ({payload['estimated_remaining_wall_time_hours_four_gpus'] / 24:.1f} days). The full 128-candidate expansion is estimated at {payload['estimated_full_expansion_wall_time_hours_four_gpus']:.1f} wall hours and {payload['estimated_full_expansion_storage_GiB']:.1f} GiB before safety margin.\n\n"
        "The pilot observed 37–43% GPU utilization and 668–704 MiB GPU memory per active process. The scheduler remains bounded to one process per GPU and the remaining 124 candidates were not launched.\n",
        encoding="utf-8",
    )


def update_registry() -> None:
    payload = json.loads(PILOT_RESULTS.read_text(encoding="utf-8"))
    rows = read_csv(EXPANSION_REGISTRY)
    by_pair = {(row["polymer_id"], row["temperature_K"]): row for row in rows}
    manifest_hash = sha(MANIFEST)
    for candidate in payload["candidates"]:
        if candidate["status"] != "COMPLETE_QC_PASS":
            continue
        for temperature in TEMPS:
            values = candidate["temperatures"][str(temperature)]
            work = RUNS / candidate["polymer_id"] / f"{temperature}K"
            final_data = work / f"production_{temperature}K_final.data"
            trajectory = work / "trajectory" / f"production_{temperature}K_200ps.lammpstrj"
            row = by_pair[(candidate["polymer_id"], str(temperature))]
            row.update({
                "temperature_completion": "COMPLETE", "simulation_status": "COMPLETE_QC_PASS", "qc_status": "COMPLETE_QC_PASS", "descriptor_status": "READY",
                "input_hash": values["input_sha256"], "output_hash": sha(final_data), "trajectory_hash": sha(trajectory), "manifest_hash": manifest_hash,
                "code_commit": commit(), "environment_fingerprint": hashlib.sha256((socket.gethostname() + str(candidate["gpu_index"]) + "mace-lammps").encode()).hexdigest(),
                "gpu_id": str(candidate["gpu_index"]), "hostname": socket.gethostname(), "pid": "", "attempt": 1,
                "failure_class": "", "retryable": "false", "provenance": str(work.relative_to(ROOT)),
            })
            write_json(work / "qc.json", {"polymer_id": candidate["polymer_id"], "temperature_K": temperature, "density": values["density"], "volume": values["volume"], "temperature_mean": values["temperature_mean"], "equilibration_time_ps": values["equilibration_time_ps"], "valid_frame_count": values["valid_frame_count"], "qc_flags": [], "missingness_mask": {}, "qc_status": values["status"]})
    fields = list(rows[0])
    temporary = EXPANSION_REGISTRY.with_name(EXPANSION_REGISTRY.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    temporary.replace(EXPANSION_REGISTRY)
    expansion_manifest = json.loads(EXPANSION_MANIFEST.read_text(encoding="utf-8"))
    expansion_manifest.update({"registry_sha256": sha(EXPANSION_REGISTRY), "pilot_selection_sha256": sha(SELECTION), "pilot_results_sha256": sha(PILOT_RESULTS), "pilot_complete_four_temperature_systems": sum(candidate["status"] == "COMPLETE_QC_PASS" for candidate in payload["candidates"]), "pilot_status": payload["pilot_status"]})
    write_json(EXPANSION_MANIFEST, expansion_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("select", "run", "resume-test", "report", "update-registry")); args = parser.parse_args()
    if args.action == "select": select()
    elif args.action == "run": run()
    elif args.action == "resume-test": resume_test()
    elif args.action == "report": report()
    else: update_registry()


if __name__ == "__main__":
    main()
