#!/usr/bin/env python3
"""Build fixed bulk-Tg pilot structures with RadonPy; intentionally no MD."""
import json
import math
import re
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / "data/processed/bulk_md_tg_pilot_12_feasible.csv"
OUTROOT = ROOT / "bulk_md_tg_pilot"
REGISTRY = ROOT / "data/processed/bulk_md_tg_pilot_build_registry.csv"
REPORT = ROOT / "reports/bulk_md_tg_pilot_build_qc.md"
NCHAINS = 6
INITIAL_DENSITY = 0.20

from radonpy.core import calc, poly, utils
from radonpy.ff.gaff2_mod import GAFF2_mod
from radonpy.sim.lammps import MolToLAMMPSdata


@contextmanager
def radonpy_single_isomer_build_mode():
    """Retain the input connectivity/stereo while avoiding combinatorial pre-embedding.

    RadonPy's mol_from_smiles performs a separate embedded enumeration of all
    unspecified stereoisomers before its ordinary 3-D embedding.  The pilot
    structures use their canonical PSMILES connectivity; this scope retains
    that molecule as the single candidate and still lets mol_from_smiles add
    hydrogens and generate the normal 3-D conformer used by polymerize_rw.
    """
    enum_module = utils.Chem.EnumerateStereoisomers
    original = enum_module.EnumerateStereoisomers
    enum_module.EnumerateStereoisomers = lambda molecule, options=None: (molecule,)
    try:
        yield
    finally:
        enum_module.EnumerateStereoisomers = original


def finite(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return all(finite(v) for v in value)
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def required_finite(obj, names):
    return all(hasattr(obj, name) and finite(getattr(obj, name)) for name in names)


def set_lammps_styles(mol):
    for atom in mol.GetAtoms():
        atom.SetProp("pair_style", "lj")
        atom.SetProp("bond_style", "harmonic")
        atom.SetProp("angle_style", "harmonic")
        atom.SetProp("dihedral_style", "fourier")
        atom.SetProp("improper_style", "cvff")


def forcefield_qc(mol):
    """Check every GAFF2_mod term and charge needed by the LAMMPS exporter."""
    errors = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("ff_type") or not atom.GetProp("ff_type"):
            errors.append(f"untyped_atom:{atom.GetIdx()}")
        for prop in ("ff_epsilon", "ff_sigma", "AtomicCharge"):
            if not atom.HasProp(prop) or not finite(atom.GetDoubleProp(prop)):
                errors.append(f"invalid_atom_{prop}:{atom.GetIdx()}")
    for bond in mol.GetBonds():
        if not bond.HasProp("ff_type") or not bond.GetProp("ff_type"):
            errors.append(f"untyped_bond:{bond.GetIdx()}")
        for prop in ("ff_k", "ff_r0"):
            if not bond.HasProp(prop) or not finite(bond.GetDoubleProp(prop)):
                errors.append(f"invalid_bond_{prop}:{bond.GetIdx()}")
    for label, attrs in (("angle", ("k", "theta0")), ("dihedral", ("k", "d0", "m", "n")), ("improper", ("k", "d0", "n"))):
        terms = getattr(mol, label + "s", [])
        for index, term in enumerate(terms):
            if not hasattr(term, "ff") or not getattr(term.ff, "type", ""):
                errors.append(f"untyped_{label}:{index}")
            elif not required_finite(term.ff, attrs):
                errors.append(f"invalid_{label}_parameters:{index}")
    return errors


def lammps_data_qc(path, expected_atoms, expected_chains):
    """Validate exported topology, atom/molecule IDs, finite coordinates, box and counts."""
    lines = path.read_text(encoding="utf-8").splitlines()
    counts = {}
    for line in lines[:35]:
        m = re.match(r"\s*(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)\s*$", line)
        if m:
            counts[m.group(2)] = int(m.group(1))
    sections = {}
    # LAMMPS files also contain Velocities and coefficient sections.  A new
    # section header terminates topology collection, avoiding velocity rows
    # being interpreted as atom records.
    section_headers = {
        "Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
        "Dihedral Coeffs", "Improper Coeffs", "Atoms", "Bonds",
        "Angles", "Dihedrals", "Impropers", "Velocities",
    }
    current = None
    for line in lines:
        token = line.strip()
        if token in section_headers:
            if token in {"Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"}:
                current = token.lower()
                sections[current] = []
            else:
                current = None
        elif current and token and not token.startswith("#"):
            sections[current].append(token)
    errors = []
    if counts.get("atoms") != expected_atoms or len(sections.get("atoms", [])) != expected_atoms:
        errors.append("atom_count_mismatch")
    atom_ids, molecule_ids, coordinates = [], [], []
    for line in sections.get("atoms", []):
        fields = line.split()
        if len(fields) < 7:
            errors.append("malformed_atom_line")
            continue
        atom_ids.append(int(fields[0])); molecule_ids.append(int(fields[1]))
        coordinates.extend(float(v) for v in fields[4:7])
    if sorted(atom_ids) != list(range(1, expected_atoms + 1)):
        errors.append("duplicate_or_noncontiguous_atom_ids")
    if len(set(molecule_ids)) != expected_chains or min(molecule_ids, default=0) < 1:
        errors.append("invalid_molecule_ids")
    if not all(finite(v) for v in coordinates):
        errors.append("nonfinite_coordinates")
    for kind, atom_columns in (("bonds", 2), ("angles", 3), ("dihedrals", 4), ("impropers", 4)):
        terms = sections.get(kind, [])
        if counts.get(kind, 0) != len(terms):
            errors.append(f"{kind}_count_mismatch")
        seen = set()
        for line in terms:
            fields = line.split()
            if len(fields) < 2 + atom_columns:
                errors.append(f"malformed_{kind}_line")
                continue
            identifier = int(fields[0]); seen.add(identifier)
            indices = [int(v) for v in fields[2:2 + atom_columns]]
            if any(i < 1 or i > expected_atoms for i in indices):
                errors.append(f"invalid_{kind}_atom_index")
        if terms and seen != set(range(1, len(terms) + 1)):
            errors.append(f"duplicate_or_noncontiguous_{kind}_ids")
    return errors, counts, len(set(molecule_ids))


def build_one(row, seed_offset=0):
    out = OUTROOT / row.polymer_id
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "build_metadata.json"
    data_path = out / "structure.data"
    record = {
        "polymer_id": row.polymer_id,
        "canonical_psmiles": row.canonical_psmiles,
        "cluster_id": int(row.cluster_id),
        "DP": int(row.selected_DP),
        "number_of_chains": NCHAINS,
        "estimated_total_atoms": int(row.estimated_total_atoms),
        "actual_total_atoms": np.nan,
        "heavy_atoms": np.nan,
        "initial_density": np.nan,
        "box_x": np.nan, "box_y": np.nan, "box_z": np.nan,
        "forcefield_status": "NOT_RUN", "export_status": "NOT_RUN", "build_status": "FAIL",
        "failure_reason": "",
    }
    try:
        # Same chemistry/construction philosophy as the validated canary.
        np.random.seed((int(row.polymer_id[-8:], 16) + int(seed_offset)) % (2**32 - 1))
        with radonpy_single_isomer_build_mode():
            monomer = utils.mol_from_smiles(row.canonical_psmiles)
        if monomer is None:
            raise RuntimeError("RadonPy mol_from_smiles returned None")
        chain = poly.polymerize_rw(monomer, int(row.selected_DP), tacticity="isotactic", retry=20,
                                  retry_step=300, opt="rdkit", dist_min=0.7)
        if chain is None:
            raise RuntimeError("polymerize_rw failed")
        set_lammps_styles(chain)
        assigned = GAFF2_mod().ff_assign(chain)
        if not assigned:
            raise RuntimeError("GAFF2_mod ff_assign returned False")
        try:
            calc.assign_charges(chain, charge="gasteiger")
            charge_method = "gasteiger"
        except Exception:
            # The validated canary uses this explicit, recorded fallback; charges remain present.
            for atom in chain.GetAtoms():
                atom.SetDoubleProp("AtomicCharge", 0.0)
            charge_method = "zero_charge_fallback_after_gasteiger_failure"
        ff_errors = forcefield_qc(chain)
        if ff_errors:
            raise RuntimeError("forcefield_qc_failed:" + ";".join(ff_errors[:10]))
        cell = poly.amorphous_cell(chain, NCHAINS, density=INITIAL_DENSITY, retry=10,
                                  retry_step=300, threshold=2.0)
        set_lammps_styles(cell)
        cell_ff_errors = forcefield_qc(cell)
        if cell_ff_errors:
            raise RuntimeError("cell_forcefield_qc_failed:" + ";".join(cell_ff_errors[:10]))
        actual_atoms = cell.GetNumAtoms()
        heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in cell.GetAtoms())
        density = float(calc.mol_density(cell))
        c = cell.cell
        box = (float(c.dx), float(c.dy), float(c.dz))
        if not all(v > 0 and finite(v) for v in box) or not (0 < density < 2.0):
            raise RuntimeError("invalid_initial_box_or_density")
        if not MolToLAMMPSdata(cell, str(data_path), velocity=True, temp=600):
            raise RuntimeError("LAMMPS data export failed")
        data_errors, header_counts, molecule_count = lammps_data_qc(data_path, actual_atoms, NCHAINS)
        text = data_path.read_text(encoding="utf-8").lower()
        if "nan" in text or "inf" in text:
            data_errors.append("nan_or_inf_in_lammps_data")
        if data_errors:
            raise RuntimeError("export_qc_failed:" + ";".join(sorted(set(data_errors))))
        record.update({
            "actual_total_atoms": actual_atoms, "heavy_atoms": heavy_atoms, "initial_density": density,
            "box_x": box[0], "box_y": box[1], "box_z": box[2],
            "forcefield_status": "PASS", "export_status": "PASS", "build_status": "PASS",
        })
        metadata = {**{key: (None if pd.isna(value) else value) for key, value in record.items()},
                    "forcefield": "GAFF2_mod", "charge_method": charge_method,
                    "initial_packing": {"density_g_cm3_requested": INITIAL_DENSITY, "retry": 10,
                                        "retry_step": 300, "threshold_A": 2.0},
                    "lammps_header_counts": header_counts, "molecule_count": molecule_count,
                    "estimated_minus_actual_atoms": int(record["estimated_total_atoms"] - actual_atoms),
                    "md_performed": False}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        record["failure_reason"] = f"{type(exc).__name__}:{exc}"
        metadata_path.write_text(json.dumps({**record, "traceback": traceback.format_exc(), "md_performed": False}, indent=2, default=str) + "\n", encoding="utf-8")
    return record


def write_report(registry):
    complete = registry[registry.build_status.eq("PASS")]
    attempted, built = len(registry), len(complete)
    ff_ok = int(registry.forcefield_status.eq("PASS").sum())
    exported = int(registry.export_status.eq("PASS").sum())
    if built == 12:
        classification = "BUILD_STAGE_PASS"
    elif built >= 10:
        classification = "BUILD_STAGE_PARTIAL"
    else:
        classification = "BUILD_STAGE_FAIL"
    def triplet(series):
        return "n/a" if series.empty else f"{series.min():.3f} / {series.median():.3f} / {series.max():.3f}"
    box_range = "n/a" if complete.empty else "; ".join(f"{axis}: {complete[axis].min():.3f}–{complete[axis].max():.3f} Å" for axis in ("box_x", "box_y", "box_z"))
    failures = registry.loc[registry.failure_reason.ne(""), ["polymer_id", "failure_reason"]]
    lines = ["# Bulk MD Tg Pilot Build QC", "", f"- Systems attempted: **{attempted}/12**",
             f"- Systems successfully built: **{built}/12**", f"- Systems successfully force-field assigned: **{ff_ok}/12**",
             f"- Systems successfully exported: **{exported}/12**", f"- Actual atom-count min / median / max: **{triplet(complete.actual_total_atoms)}**",
             f"- DP min / median / max: **{triplet(complete.DP)}**", f"- Initial-density min / median / max (g/cm³): **{triplet(complete.initial_density)}**",
             f"- Box-size ranges: **{box_range}**", f"- Final classification: **{classification}**", "",
             "## Estimated versus actual atom counts", "", "| polymer_id | estimated | actual | actual − estimated |", "|---|---:|---:|---:|"]
    for row in registry.itertuples(index=False):
        actual = "" if pd.isna(row.actual_total_atoms) else int(row.actual_total_atoms)
        delta = "" if pd.isna(row.actual_total_atoms) else int(row.actual_total_atoms - row.estimated_total_atoms)
        lines.append(f"| {row.polymer_id} | {row.estimated_total_atoms} | {actual} | {delta} |")
    lines += ["", "## Force-field / topology failures", ""]
    lines += ([f"- `{row.polymer_id}`: {row.failure_reason}" for row in failures.itertuples(index=False)] if not failures.empty else ["None."])
    lines += ["", "No minimization, NVT, NPT, densification, or other MD was run.", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return classification


def main():
    selected = pd.read_csv(INPUT)
    required = {"polymer_id", "canonical_psmiles", "cluster_id", "selected_DP", "estimated_total_atoms"}
    if len(selected) != 12 or not required.issubset(selected.columns) or selected.polymer_id.nunique() != 12:
        raise RuntimeError("fixed feasible pilot CSV is not a unique 12-polymer manifest")
    OUTROOT.mkdir(exist_ok=True)
    records = []
    for row in selected.itertuples(index=False):
        print(f"BUILD_START {row.polymer_id}", flush=True)
        record = build_one(row)
        records.append(record)
        print(f"BUILD_END {row.polymer_id} {record['build_status']} {record['failure_reason']}", flush=True)
    registry = pd.DataFrame(records)
    registry.to_csv(REGISTRY, index=False)
    classification = write_report(registry)
    complete = registry[registry.build_status.eq("PASS")]
    print("FINAL_BUILD_SUMMARY", json.dumps({
        "systems_attempted": len(registry), "systems_built": len(complete),
        "forcefield_successes": int(registry.forcefield_status.eq("PASS").sum()),
        "lammps_export_successes": int(registry.export_status.eq("PASS").sum()),
        "actual_atom_count_range": [] if complete.empty else [int(complete.actual_total_atoms.min()), int(complete.actual_total_atoms.max())],
        "DP_range": [] if complete.empty else [int(complete.DP.min()), int(complete.DP.max())],
        "initial_density_range": [] if complete.empty else [float(complete.initial_density.min()), float(complete.initial_density.max())],
        "major_failures": registry.loc[registry.failure_reason.ne(""), "failure_reason"].tolist(),
        "final_classification": classification,
    }), flush=True)


if __name__ == "__main__":
    main()
