"""Build Phase 6D expansion topologies and validate production inputs.

This stage reuses the frozen RadonPy/GAFF2_mod builder and never launches
LAMMPS.  Generated structures and logs live below the ignored phase6d_runs
tree; only the small registry and audit artifacts are retained in Git.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "phase6d_artifacts"
BUILD_ROOT = ROOT / "phase6d_runs" / "expansion_builds"
INPUT_ROOT = ROOT / "phase6d_runs" / "expansion"
MANIFEST = ART / "expansion_candidate_manifest.v1.json"
POOL = ART / "expansion_candidate_pool.v1.csv"
SELECTED = ART / "expansion_selected_candidates.v1.csv"
REGISTRY = ART / "candidate_build_registry.v1.csv"
BUILD_MANIFEST = ART / "candidate_build_manifest.v1.json"
READY_QUEUE = ART / "phase6d_ready_queue.v2.csv"
READY_MANIFEST = ART / "phase6d_ready_queue_manifest.v2.json"
DRY_RUN = ART / "phase6d_ready_queue_dry_run.v1.json"
TEMPERATURES = (300, 400, 500, 600)
PILOTS = {"poly_98bc13bed4455c1f", "poly_db696efd35d8326b", "poly_a5cce35c85e525e3", "poly_df7c069ad0b2d467"}
REGISTRY_FIELDS = [
    "polymer_id", "canonical_psmiles", "structure_hash", "chemical_cluster",
    "selected_chain_count", "selected_DP", "estimated_total_atoms",
    "force_field", "force_field_source_hash", "charge_provenance",
    "structure_data", "structure_sha256", "atom_count", "bond_count",
    "angle_count", "dihedral_count", "improper_count", "box_xlo_xhi",
    "box_ylo_yhi", "box_zlo_zhi", "temperature_list", "input_300K",
    "input_400K", "input_500K", "input_600K", "input_hashes",
    "code_commit", "environment_fingerprint", "build_status", "failure_reason",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def candidates() -> list[dict[str, str]]:
    selected = {row["polymer_id"]: row for row in read_csv(SELECTED)}
    universe = {row["polymer_id"]: row for row in read_csv(ROOT / "data/processed/bulk_md_tg_build_preflight_structural_universe.csv")}
    pool = {row["polymer_id"]: row for row in read_csv(POOL)}
    locked = {row["polymer_id"] for row in read_csv(ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv") if row["split"] == "LOCKED_HOLDOUT"}
    if sha(MANIFEST) != "a18588e16d702fba82bead7128e4aa808ffe4f9169cdc79c6a0805b84e488509":
        raise RuntimeError("frozen candidate manifest hash changed")
    output = []
    for polymer_id, row in selected.items():
        if polymer_id in PILOTS or polymer_id in locked:
            continue
        source = universe.get(polymer_id)
        source_pool = pool.get(polymer_id)
        if source is None or source_pool is None or source["canonical_psmiles"] != row["canonical_psmiles"]:
            raise RuntimeError(f"identity mismatch for {polymer_id}")
        if text_sha(row["canonical_psmiles"]) != row["structure_hash"]:
            raise RuntimeError(f"structure hash mismatch for {polymer_id}")
        output.append({**row, **source, "chemical_cluster": row["chemical_cluster"], "locked_holdout": False})
    if len(output) != 124:
        raise RuntimeError(f"expected 124 remaining candidates, found {len(output)}")
    return output


def build_worker(row: dict[str, str]) -> dict[str, str]:
    """Run the existing builder in the current RadonPy environment."""
    import inspect

    import frozen_bulk_md_tg_production_builder as builder
    from radonpy.ff.gaff2_mod import GAFF2_mod
    from rdkit import Chem

    builder.OUTROOT = BUILD_ROOT
    target = BUILD_ROOT / row["polymer_id"]
    target.mkdir(parents=True, exist_ok=True)
    existing_result = target / "build_result.json"
    if existing_result.is_file():
        existing = json.loads(existing_result.read_text(encoding="utf-8"))
        if "particle database has no element types: *" not in existing.get("failure_reason", ""):
            return {**row, **existing}
    database = Path(inspect.getfile(GAFF2_mod)).parent / "ff_dat" / "gaff2_mod.json"
    supported = {item["elem"] for item in json.loads(database.read_text(encoding="utf-8"))["particle_types"]}
    molecule = Chem.MolFromSmiles(row["canonical_psmiles"])
    missing = sorted({atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 0} - supported) if molecule is not None else []
    if molecule is None:
        result = {**row, "build_status": "IDENTITY_MISMATCH", "failure_reason": "RDKit canonical PSMILES parse failed"}
        write_json(target / "build_result.json", result)
        return result
    if missing:
        result = {**row, "build_status": "UNSUPPORTED", "failure_reason": "GAFF2_mod particle database has no element types: " + ",".join(missing)}
        write_json(target / "build_result.json", result)
        return result
    build_row = SimpleNamespace(
        polymer_id=row["polymer_id"], canonical_psmiles=row["canonical_psmiles"],
        chemical_cluster_id=int(row["chemical_cluster"]), split="DEVELOPMENT_EXPANSION",
        selected_DP=int(row["selected_DP"]), selected_chain_count=int(row["selected_chain_count"]),
        estimated_total_atoms=int(row["estimated_total_atoms"]),
    )
    try:
        result = builder.build_one(build_row, iterative_mol_id=True)
        status = "BUILD_PASS" if result.get("build_status") == "PASS" else classify_failure(str(result.get("failure_reason", "")))
        result = {str(k): ("" if v is None else str(v)) for k, v in result.items()}
        result["build_status"] = status
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        result = {"polymer_id": row["polymer_id"], "failure_reason": f"{type(exc).__name__}:{exc}", "build_status": "BUILD_FAILURE"}
    write_json(target / "build_result.json", result)
    return {**row, **result}


def classify_failure(reason: str) -> str:
    lower = reason.lower()
    if "gaff2_mod" in lower or "forcefield" in lower or "ff_assign" in lower:
        return "FORCE_FIELD_FAILURE"
    if "charge" in lower:
        return "CHARGE_FAILURE"
    if "unsupported" in lower or "element" in lower:
        return "UNSUPPORTED"
    if "timeout" in lower:
        return "BUILD_TIMEOUT"
    if "topology" in lower or "export" in lower or "atom" in lower or "box" in lower:
        return "TOPOLOGY_FAILURE"
    return "TOPOLOGY_FAILURE"


def parse_data(path: Path) -> dict[str, object]:
    counts = {key: 0 for key in ("atoms", "bonds", "angles", "dihedrals", "impropers")}
    sections: dict[str, list[list[str]]] = {}
    current = None
    box: dict[str, tuple[float, float]] = {}
    headers = {"Atoms", "Bonds", "Angles", "Dihedrals", "Impropers", "Velocities", "Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"(\d+) (atoms|bonds|angles|dihedrals|impropers)", line)
        if match:
            counts[match.group(2)] = int(match.group(1)); continue
        if line in headers:
            current = line
            if line in {"Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"}:
                sections[line] = []
            continue
        match = re.fullmatch(r"([-+0-9.eE]+) ([-+0-9.eE]+) ([xyz])lo ([xyz])hi", line)
        if match:
            box[match.group(3)] = (float(match.group(1)), float(match.group(2)))
        elif current in sections:
            sections[current].append(line.split())
    if not path.is_file() or counts["atoms"] <= 0:
        raise ValueError("missing or empty structure.data")
    if any(len(sections.get(name, [])) != counts[key] for name, key in (("Atoms", "atoms"), ("Bonds", "bonds"), ("Angles", "angles"), ("Dihedrals", "dihedrals"), ("Impropers", "impropers"))):
        raise ValueError("topology section count mismatch")
    if set(box) != {"x", "y", "z"} or any(not (math.isfinite(a) and math.isfinite(b) and b > a) for a, b in box.values()):
        raise ValueError("invalid periodic box")
    if any(not all(math.isfinite(float(value)) for value in atom[3:7]) for atom in sections["Atoms"]):
        raise ValueError("nonfinite atom charge or coordinate")
    return {**counts, "box": box}


def input_text(polymer_id: str, temperature: int) -> str:
    seed = int(hashlib.sha256(f"{polymer_id}:{temperature}".encode()).hexdigest()[:8], 16) % 2147483647
    return f"""# Phase 6D frozen production entrypoint; validation-only generation stage\n# polymer_id={polymer_id}\n# launch: CUDA_VISIBLE_DEVICES=<gpu> lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input_{temperature}K.in\nunits real\natom_style full\nboundary p p p\npair_style lj/cut 10.0\nbond_style harmonic\nangle_style harmonic\ndihedral_style fourier\nimproper_style cvff\nspecial_bonds amber\nread_data structure.data\nneighbor 2.0 bin\nneigh_modify every 1 delay 0 check yes\nthermo 1000\nthermo_style custom step temp press density pe etotal vol lx ly lz\nthermo_modify flush yes\ntimestep 1.0\nvelocity all create {temperature}.0 {seed} mom yes rot yes dist gaussian\nfix npt all npt temp {temperature}.0 {temperature}.0 100.0 iso 1.0 1.0 1000.0\ndump phase6d all custom 10000 trajectory/production_{temperature}K_200ps.lammpstrj id mol type xu yu zu\ndump_modify phase6d sort id\nrun 200000\nundump phase6d\nunfix npt\nwrite_restart checkpoints/production_{temperature}K.restart\nwrite_data outputs/production_{temperature}K_final.data\n"""


def generate_inputs(record: dict[str, str]) -> dict[str, str]:
    if record.get("build_status") != "BUILD_PASS":
        return record
    pid = record["polymer_id"]
    structure = BUILD_ROOT / pid / "structure.data"
    try:
        topology = parse_data(structure)
        candidate_root = INPUT_ROOT / pid
        candidate_root.mkdir(parents=True, exist_ok=True)
        ensure_link(candidate_root / "structure.data", structure)
        hashes = {}
        paths = {}
        for temperature in TEMPERATURES:
            work = candidate_root / f"{temperature}K"
            work.mkdir(parents=True, exist_ok=True)
            ensure_link(work / "structure.data", structure)
            input_path = work / f"input_{temperature}K.in"
            input_path.write_text(input_text(pid, temperature), encoding="utf-8")
            ensure_link(work / "input.in", Path(input_path.name))
            (work / "trajectory").mkdir(exist_ok=True)
            (work / "checkpoints").mkdir(exist_ok=True)
            (work / "outputs").mkdir(exist_ok=True)
            hashes[str(temperature)] = sha(input_path)
            paths[str(temperature)] = str(input_path.relative_to(ROOT))
        metadata = json.loads((BUILD_ROOT / pid / "build_metadata.json").read_text(encoding="utf-8"))
        import inspect

        from radonpy.ff.gaff2_mod import GAFF2_mod
        force_field_source = Path(inspect.getfile(GAFF2_mod)).parent / "ff_dat" / "gaff2_mod.json"
        record.update({
            "force_field": "GAFF2_mod", "force_field_source_hash": sha(force_field_source),
            "charge_provenance": metadata.get("charge_method", ""),
            "structure_data": str(structure.relative_to(ROOT)), "structure_sha256": sha(structure),
            "atom_count": str(topology["atoms"]), "bond_count": str(topology["bonds"]),
            "angle_count": str(topology["angles"]), "dihedral_count": str(topology["dihedrals"]),
            "improper_count": str(topology["impropers"]),
            "box_xlo_xhi": json.dumps(topology["box"]["x"]), "box_ylo_yhi": json.dumps(topology["box"]["y"]),
            "box_zlo_zhi": json.dumps(topology["box"]["z"]), "temperature_list": json.dumps(list(TEMPERATURES)),
            "input_300K": paths["300"], "input_400K": paths["400"], "input_500K": paths["500"], "input_600K": paths["600"],
            "input_hashes": json.dumps(hashes, sort_keys=True), "code_commit": commit(),
            "environment_fingerprint": text_sha(f"{sys.executable}:radonpy:GAFF2_mod"), "build_status": "BUILD_PASS",
            "failure_reason": "",
        })
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record["build_status"] = classify_failure(f"{type(exc).__name__}:{exc}")
        record["failure_reason"] = f"{type(exc).__name__}:{exc}"
    return record


def ensure_link(link: Path, target: Path) -> None:
    expected = (link.parent / target).resolve() if not target.is_absolute() else target.resolve()
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == expected:
            return
        if link.is_file() and link.resolve() == expected:
            return
        raise FileExistsError(f"existing path does not match expected target: {link}")
    link.symlink_to(target)


def build_all(workers: int) -> list[dict[str, str]]:
    rows = candidates()
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(build_worker, row): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"polymer_id": result["polymer_id"], "status": result.get("build_status"), "failure": result.get("failure_reason", "")}), flush=True)
    results = [generate_inputs(result) for result in sorted(results, key=lambda row: row["polymer_id"])]
    write_csv(REGISTRY, results, REGISTRY_FIELDS)
    write_json(BUILD_MANIFEST, {
        "schema_version": "phase6d.candidate_build_manifest.v1", "manifest_sha256": sha(MANIFEST),
        "selected_candidate_list_sha256": sha(SELECTED), "candidate_count": len(results),
        "temperature_list": list(TEMPERATURES), "builder": "scripts/frozen_bulk_md_tg_production_builder.py",
        "force_field": "GAFF2_mod", "build_status_counts": {status: sum(row.get("build_status") == status for row in results) for status in sorted({row.get("build_status") for row in results})},
        "production_launched": False, "code_commit": commit(),
    })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "validate-inputs"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.action == "build":
        build_all(max(1, args.workers))
    else:
        rows = read_csv(REGISTRY)
        for row in rows:
            if row.get("build_status") != "BUILD_PASS":
                continue
            for temperature in TEMPERATURES:
                path = ROOT / row[f"input_{temperature}K"]
                text = path.read_text(encoding="utf-8")
                if f"temp {temperature}.0 {temperature}.0" not in text or f"production_{temperature}K" not in text:
                    raise SystemExit(f"temperature validation failed: {path}")
        print(json.dumps({"status": "PASS", "build_pass": sum(row.get("build_status") == "BUILD_PASS" for row in rows), "input_files": sum(row.get("build_status") == "BUILD_PASS" for row in rows) * 4}))


if __name__ == "__main__":
    main()
