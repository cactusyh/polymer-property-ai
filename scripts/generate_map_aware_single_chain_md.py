#!/usr/bin/env python3
"""Generate short, provenance-preserving single-chain MD pilot systems."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem


OUTPUT_ROOT = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_pilot")
RADONPY_ROOT = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/RadonPy")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_table(directory: Path, stem: str, rows: list[dict]) -> None:
    write_rows(directory / f"{stem}.csv", rows)
    try:
        pd.DataFrame(rows).to_parquet(directory / f"{stem}.parquet", index=False)
    except ImportError:
        pass


def set_source_indices(raw_mol: Chem.Mol, monomer: Chem.Mol) -> None:
    raw_heavy = [a.GetIdx() for a in raw_mol.GetAtoms() if a.GetSymbol() != "*"]
    heavy_idx = 0
    for atom in monomer.GetAtoms():
        if atom.GetSymbol() == "H":
            neighbors = list(atom.GetNeighbors())
            source = neighbors[0].GetIntProp("source_rdkit_atom_index") if neighbors and neighbors[0].HasProp("source_rdkit_atom_index") else -1
        else:
            source = raw_heavy[heavy_idx] if heavy_idx < len(raw_heavy) else -1
            heavy_idx += 1
        atom.SetIntProp("source_rdkit_atom_index", int(source))


def add_charges(mol) -> bool:
    from radonpy.core import calc

    try:
        ok = calc.assign_charges(mol, charge="gasteiger")
    except Exception:
        ok = False
    if not ok:
        for atom in mol.GetAtoms():
            atom.SetDoubleProp("AtomicCharge", 0.0)
        return False
    return all(atom.HasProp("AtomicCharge") for atom in mol.GetAtoms())


def construct_chain(poly, monomer, n_repeat, args):
    strategies = [
        ("original", n_repeat, {"retry": 1, "retry_step": 20, "opt": None, "dist_min": 0.7}),
        ("robust_random_walk", n_repeat, {"retry": 20, "retry_step": 200, "opt": "rdkit", "dist_min": 0.7}),
        ("relaxed_overlap_random_walk", n_repeat, {"retry": 40, "retry_step": 300, "opt": "rdkit", "dist_min": 0.5}),
        ("extended_initialization", n_repeat, {"retry": 60, "retry_step": 500, "opt": "rdkit", "dist_min": 0.45}),
    ]
    if args.allow_short_chain_diagnostic and n_repeat > 2:
        strategies.append(("technical_short_chain_only", max(2, n_repeat // 2), {"retry": 30, "retry_step": 300, "opt": "rdkit", "dist_min": 0.5}))
    attempts = []
    for strategy, repeat_count, kwargs in strategies:
        started = time.time()
        try:
            chain = poly.polymerize_rw(monomer, repeat_count, tacticity="isotactic", **kwargs)
            if chain is None or chain.GetNumAtoms() == 0:
                raise RuntimeError("polymerize_rw returned no molecule")
            attempts.append({"strategy": strategy, "n_repeat": repeat_count, "elapsed_seconds": time.time() - started, "success": True, "failure_reason": None})
            return chain, strategy, repeat_count, attempts
        except Exception as exc:
            attempts.append({"strategy": strategy, "n_repeat": repeat_count, "elapsed_seconds": time.time() - started, "success": False, "failure_reason": f"{type(exc).__name__}: {exc}"})
    error = RuntimeError("all construction strategies failed")
    error.attempts = attempts
    raise error


def build_system(row: dict, args: argparse.Namespace) -> dict:
    from radonpy.core import utils, poly
    from radonpy.ff.gaff2_mod import GAFF2_mod
    from radonpy.sim.lammps import MolToLAMMPSdata

    polymer_id = str(row["polymer_id"])
    output_root = Path(args.output_root)
    system_root = output_root / "systems" / polymer_id
    input_dir = system_root / "input"
    metadata_dir = system_root / "metadata"
    lammps_dir = system_root / "lammps"
    trajectory_dir = system_root / "trajectory"
    descriptors_dir = system_root / "descriptors"
    for directory in (input_dir, metadata_dir, lammps_dir, trajectory_dir, descriptors_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.repair and (metadata_dir / "generation_status.json").exists():
        repair_dir = system_root / "repair_attempts"
        repair_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_dir / "generation_status.json", repair_dir / "original_generation_status.json")
    status = {
        "polymer_id": polymer_id,
        "canonical_psmiles": row.get("canonical_psmiles"),
        "status": "FAILED",
        "structure_generated": False,
        "force_field_success": False,
        "md_success": False,
        "mapping_success": False,
        "construction_strategy_used": None,
        "construction_attempts": 0,
        "failure_reason": None,
    }
    attempts = []
    try:
        raw_smiles = str(row["canonical_psmiles"])
        raw_mol = Chem.MolFromSmiles(raw_smiles)
        if raw_mol is None or raw_mol.GetNumAtoms() == 0:
            raise RuntimeError("RDKit parse failed")
        monomer = utils.mol_from_smiles(raw_smiles)
        if monomer is None or monomer.GetNumAtoms() == 0:
            raise RuntimeError("RadonPy repeat-unit conversion failed")
        for atom in raw_mol.GetAtoms():
            atom.SetIntProp("source_rdkit_atom_index", atom.GetIdx())
        set_source_indices(raw_mol, monomer)
        repeat_atoms = monomer.GetNumAtoms()
        n_repeat = max(2, int(args.target_atoms / max(repeat_atoms, 1)))
        chain, strategy, actual_repeat_count, attempts = construct_chain(poly, monomer, n_repeat, args)
        n_repeat = actual_repeat_count
        status["construction_strategy_used"] = strategy
        status["construction_attempts"] = len(attempts)
        if strategy == "technical_short_chain_only":
            raise RuntimeError("short-chain diagnostic succeeded but is excluded from final workflow")
        status["structure_generated"] = True
        chain.SetProp("pair_style", "lj")
        chain.SetProp("bond_style", "harmonic")
        chain.SetProp("angle_style", "harmonic")
        chain.SetProp("dihedral_style", "fourier")
        chain.SetProp("improper_style", "cvff")
        ff = GAFF2_mod()
        if not ff.ff_assign(chain):
            raise RuntimeError("GAFF2_mod ff_assign returned False")
        status["force_field_success"] = True
        charge_ok = add_charges(chain)
        if not charge_ok:
            status["charge_assignment"] = "fallback_zero_charge"
        positions = chain.GetConformer(0).GetPositions()
        mins = positions.min(axis=0) - args.box_padding
        maxs = positions.max(axis=0) + args.box_padding
        chain.cell = utils.Cell(float(maxs[0]), float(mins[0]), float(maxs[1]), float(mins[1]), float(maxs[2]), float(mins[2]))
        for idx, atom in enumerate(chain.GetAtoms()):
            if not atom.HasProp("source_rdkit_atom_index"):
                atom.SetIntProp("source_rdkit_atom_index", -1)
            atom.SetIntProp("chain_index", 0)
            source_idx = int(atom.GetIntProp("source_rdkit_atom_index"))
            source_atom = raw_mol.GetAtomWithIdx(source_idx) if 0 <= source_idx < raw_mol.GetNumAtoms() else None
            atom.SetIntProp("source_is_aromatic", int(bool(source_atom and source_atom.GetIsAromatic())))
            atom.SetIntProp("source_is_sp2", int(bool(source_atom and (source_atom.GetIsAromatic() or source_atom.GetHybridization() == Chem.HybridizationType.SP2))))
            atom.SetIntProp("source_is_conjugated_atom", int(bool(source_atom and any(b.GetIsConjugated() for b in source_atom.GetBonds()))))
        data_path = lammps_dir / "structure.data"
        if not MolToLAMMPSdata(chain, str(data_path), velocity=False, temp=300):
            raise RuntimeError("MolToLAMMPSdata returned False")
        status["n_atoms"] = chain.GetNumAtoms()
        status["n_bonds"] = chain.GetNumBonds()
        atom_rows = []
        for idx, atom in enumerate(chain.GetAtoms(), start=1):
            residue = atom.GetPDBResidueInfo()
            atom_rows.append({
                "polymer_id": polymer_id, "chain_index": 0,
                "repeat_index": int(residue.GetResidueNumber()) if residue else 0,
                "rdkit_atom_index": int(atom.GetIntProp("source_rdkit_atom_index")),
                "rdkit_atom_symbol": atom.GetSymbol(),
                "rdkit_atom_map_number": int(atom.GetAtomMapNum()),
                "topology_atom_id": idx, "lammps_atom_id": idx, "trajectory_atom_id": idx,
                "lammps_atom_type": int(atom.GetIntProp("ff_type_num")),
                "partial_charge": float(atom.GetDoubleProp("AtomicCharge")),
                "is_backbone": bool(atom.HasProp("main_chain") and atom.GetBoolProp("main_chain")),
                "is_attachment_atom": bool(atom.HasProp("linker") and atom.GetBoolProp("linker")),
                "is_aromatic": bool(atom.GetIntProp("source_is_aromatic")),
                "is_sp2": bool(atom.GetIntProp("source_is_sp2")),
                "is_conjugated_atom": bool(atom.GetIntProp("source_is_conjugated_atom")),
                "ring_id": None, "conjugation_path_id": None,
            })
        write_table(metadata_dir, "atom_map", atom_rows)
        bond_rows = [{"bond_id": i + 1, "begin_atom_id": b.GetBeginAtomIdx() + 1, "end_atom_id": b.GetEndAtomIdx() + 1, "is_conjugated": bool(b.GetIsConjugated()), "is_aromatic": bool(b.GetIsAromatic())} for i, b in enumerate(chain.GetBonds())]
        write_table(metadata_dir, "bond_map", bond_rows)
        torsion_rows = []
        for bond in chain.GetBonds():
            j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if not (bond.GetIsConjugated() or (chain.GetAtomWithIdx(j).GetIsAromatic() and chain.GetAtomWithIdx(k).GetIsAromatic())):
                continue
            for i_atom in chain.GetAtomWithIdx(j).GetNeighbors():
                if i_atom.GetIdx() == k:
                    continue
                for l_atom in chain.GetAtomWithIdx(k).GetNeighbors():
                    if l_atom.GetIdx() in (j, i_atom.GetIdx()):
                        continue
                    torsion_rows.append({"torsion_id": len(torsion_rows) + 1, "atom_i": i_atom.GetIdx() + 1, "atom_j": j + 1, "atom_k": k + 1, "atom_l": l_atom.GetIdx() + 1, "central_bond_id": bond.GetIdx() + 1})
        write_table(metadata_dir, "torsion_map", torsion_rows)
        ring_rows = []
        repeat_to_atoms = {}
        for atom_row in atom_rows:
            repeat_to_atoms.setdefault(int(atom_row["repeat_index"]), {})[int(atom_row["rdkit_atom_index"])] = int(atom_row["lammps_atom_id"])
        raw_rings = [list(r) for r in raw_mol.GetRingInfo().AtomRings() if all(raw_mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
        for repeat_index, source_to_lammps in sorted(repeat_to_atoms.items()):
            for ring_index, ring in enumerate(raw_rings):
                if all(i in source_to_lammps for i in ring):
                    ring_rows.append({"ring_id": len(ring_rows), "repeat_index": repeat_index, "source_ring_index": ring_index, "atom_ids": json.dumps([source_to_lammps[i] for i in ring])})
        write_table(metadata_dir, "ring_map", ring_rows)
        path_rows = []
        for path_id, component in enumerate(sorted({int(r["rdkit_atom_index"]) for r in atom_rows if r["is_conjugated_atom"]} for _ in [])):
            path_rows.append({"path_id": path_id, "atom_ids": json.dumps(sorted(component))})
        write_table(metadata_dir, "conjugation_path_map", path_rows)
        write_table(metadata_dir, "motif_atom_map", [r for r in atom_rows if r["is_aromatic"] or r["is_conjugated_atom"]])
        (input_dir / "psmiles.txt").write_text(raw_smiles + "\n", encoding="utf-8")
        metadata = {"polymer_id": polymer_id, "canonical_psmiles": raw_smiles, "raw_psmiles": row.get("raw_psmiles"), "force_field": "GAFF2_mod", "charge_method": "Gasteiger with zero fallback", "target_atoms": args.target_atoms, "n_repeat": n_repeat, "units": "real", "timestep_fs": 1.0, "temperature_K": 300.0, "ensemble": "NVT", "equilibration_steps": args.equil_steps, "production_steps": args.production_steps, "production_time_ps": args.production_steps / 1000.0, "dump_every_steps": args.dump_every, "expected_frame_count": args.production_steps // args.dump_every + 1, "coordinates": "unwrapped xu yu zu", "mapping_source": "RadonPy atom provenance + explicit chain residue numbers", "mapping_status": "pending validation"}
        (metadata_dir / "system_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        script = lammps_dir / "in.map_aware"
        script.write_text("\n".join([
            "units real", "atom_style full", "boundary p p p", "bond_style harmonic", "angle_style harmonic", "dihedral_style fourier", "improper_style cvff", "special_bonds amber", "pair_style lj/cut 10.0", "read_data structure.data", "neighbor 2.0 bin", "neigh_modify every 1 delay 0 check yes", "thermo 200", "thermo_style custom step temp pe ke etotal press density", "min_style cg", "minimize 1.0e-4 1.0e-6 1000 10000", "reset_timestep 0", "timestep 1.0", "velocity all create 300.0 12345 mom yes rot yes dist gaussian", "fix NVT all nvt temp 300.0 300.0 100.0", f"run {args.equil_steps}", f"dump D1 all custom {args.dump_every} trajectory.lammpstrj id type xu yu zu", "dump_modify D1 sort id", f"run {args.production_steps}", "undump D1", "unfix NVT", "write_restart restart.map_aware", ""
        ]), encoding="utf-8")
        if args.prepare_only:
            status["status"] = "PREPARED"
            status["prepared_only"] = True
            status["md_success"] = False
            return status
        env = os.environ.copy()
        env["OMPI_MCA_btl"] = "self,tcp"
        env["OMPI_MCA_pml"] = "ob1"
        if args.lammps_wrapper:
            command = [args.lammps_wrapper, "in.map_aware"]
        else:
            command = [args.lammps, "-in", "in.map_aware"]
        completed = subprocess.run(command, cwd=lammps_dir, env=env, capture_output=True, text=True, timeout=args.timeout)
        (lammps_dir / "lammps.stdout.log").write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        trajectory = lammps_dir / "trajectory.lammpstrj"
        if completed.returncode != 0 or not trajectory.exists():
            raise RuntimeError(f"LAMMPS failed returncode={completed.returncode}")
        shutil.copy2(trajectory, trajectory_dir / "trajectory.lammpstrj")
        status["md_success"] = True
        status["status"] = "GENERATED"
    except Exception as exc:
        attempts = getattr(exc, "attempts", attempts)
        status["failure_reason"] = f"{type(exc).__name__}: {exc}"
    status["construction_attempts"] = len(attempts)
    (metadata_dir / "generation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    if attempts:
        attempt_dir = system_root / "repair_attempts"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        for index, attempt in enumerate(attempts, start=1):
            attempt["attempt_index"] = index
            attempt["polymer_id"] = polymer_id
        write_rows(attempt_dir / "construction_attempts.csv", attempts)
        try:
            pd.DataFrame(attempts).to_parquet(attempt_dir / "construction_attempts.parquet", index=False)
        except ImportError:
            pass
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-csv", type=Path, default=Path("data/processed/map_aware_md_pilot_20.csv"))
    parser.add_argument("--target-atoms", type=int, default=300)
    parser.add_argument("--equil-steps", type=int, default=1000)
    parser.add_argument("--production-steps", type=int, default=5000)
    parser.add_argument("--dump-every", type=int, default=500)
    parser.add_argument("--box-padding", type=float, default=15.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--lammps", default="/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/lmp")
    parser.add_argument("--lammps-wrapper", default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--allow-short-chain-diagnostic", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.pilot_csv).to_dict("records")
    statuses = [build_system(row, args) for row in rows]
    write_rows(args.output_root / "reports" / "generation_status.csv", statuses)
    print(json.dumps({"pilot_candidate_count": len(rows), "systems_attempted": len(statuses), "md_successes": sum(x["md_success"] for x in statuses)}, indent=2))


if __name__ == "__main__":
    main()
