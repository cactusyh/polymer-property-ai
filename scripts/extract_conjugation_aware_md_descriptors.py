#!/usr/bin/env python3
"""Extract mapping-aware, exploratory conjugation descriptors from pilot dumps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_pilot")


def read_table(directory: Path, stem: str) -> pd.DataFrame:
    parquet = directory / f"{stem}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(directory / f"{stem}.csv")


def read_dump(path: Path):
    lines = path.read_text(errors="ignore").splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue
        step = int(lines[i + 1])
        n_atoms = int(lines[i + 3])
        box_start = i + 4
        box = np.array([[float(x) for x in lines[box_start + 1 + j].split()[:2]] for j in range(3)])
        header = lines[box_start + 4].split()[2:]
        index = {name: position for position, name in enumerate(header)}
        required = {"id", "xu", "yu", "zu"}
        if not required.issubset(index):
            raise ValueError(f"trajectory missing {sorted(required - set(index))}")
        values = []
        for row in lines[box_start + 5 : box_start + 5 + n_atoms]:
            fields = row.split()
            values.append((int(fields[index["id"]]), [float(fields[index[x]]) for x in ("xu", "yu", "zu")]))
        values.sort(key=lambda item: item[0])
        xyz = np.asarray([item[1] for item in values], dtype=float)
        if not np.isfinite(xyz).all():
            raise ValueError("non-finite trajectory coordinates")
        frames.append((step, box, xyz))
        i = box_start + 5 + n_atoms
    return frames


def dihedral(points: np.ndarray) -> float:
    b0 = -(points[1] - points[0])
    b1 = points[2] - points[1]
    b2 = points[3] - points[2]
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def acute_plane_angle(points_a: np.ndarray, points_b: np.ndarray) -> float:
    def normal(points):
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        vector = vh[-1]
        return vector / np.linalg.norm(vector)

    cosine = abs(float(np.dot(normal(points_a), normal(points_b))))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def summary(values, prefix):
    if not values:
        return {f"{prefix}_{suffix}": np.nan for suffix in ("mean", "std", "p50", "p90")}
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def theta(phi):
    return min(abs(phi) % 360.0, abs((phi - 180.0) % 360.0))


def min_image_distance(a, b, box):
    delta = b - a
    lengths = box[:, 1] - box[:, 0]
    delta -= lengths * np.round(delta / lengths)
    return float(np.linalg.norm(delta))


def frame_shape(xyz):
    centered = xyz - xyz.mean(axis=0)
    covariance = centered.T @ centered / max(len(xyz), 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    total = max(float(eigenvalues.sum()), 1e-12)
    anisotropy = float(1.5 * np.sum((eigenvalues / total - 1 / 3) ** 2))
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))), anisotropy


def extract_system(system: Path) -> dict:
    metadata = system / "metadata"
    result = {"polymer_id": system.name, "descriptor_status": "FAILED"}
    validation = json.loads((metadata / "trajectory_validation.json").read_text())
    if not validation.get("mapping_success"):
        result["failure_reason"] = validation.get("failure_reason", "mapping validation failed")
        return result

    atom_map = read_table(metadata, "atom_map")
    torsions = read_table(metadata, "torsion_map")
    rings = read_table(metadata, "ring_map")
    bonds = read_table(metadata, "bond_map")
    frames = read_dump(system / "trajectory" / "trajectory.lammpstrj")
    aromatic_ids = set(atom_map.loc[atom_map.is_aromatic.astype(bool), "lammps_atom_id"].astype(int))
    conjugated_ids = set(atom_map.loc[atom_map.is_conjugated_atom.astype(bool), "lammps_atom_id"].astype(int))
    bond_pairs = {(int(row.begin_atom_id), int(row.end_atom_id)) for row in bonds.itertuples()}
    bond_pairs |= {(b, a) for a, b in bond_pairs}
    torsion_rows = [tuple(int(getattr(row, key)) for key in ("atom_i", "atom_j", "atom_k", "atom_l")) for row in torsions.itertuples()]
    ring_atoms = [tuple(json.loads(value)) for value in rings.atom_ids.dropna()] if "atom_ids" in rings else []
    torsion_values = []
    ring_values = []
    rg_values = []
    anisotropy_values = []
    nonlocal_values = []
    conjugated_contacts = []
    aromatic_contacts = []
    stack_contacts = []
    per_frame_thetas = []

    for _, box, xyz in frames:
        values = [theta(dihedral(xyz[np.asarray(t) - 1])) for t in torsion_rows]
        torsion_values.extend(values)
        per_frame_thetas.append(values)
        rg, anisotropy = frame_shape(xyz)
        rg_values.append(rg)
        anisotropy_values.append(anisotropy)
        if len(xyz) > 1:
            distances = []
            conj_distances = []
            aromatic_distances = []
            for left in range(len(xyz)):
                for right in range(left + 1, len(xyz)):
                    pair = (left + 1, right + 1)
                    if pair in bond_pairs or right - left <= 3:
                        continue
                    distance = min_image_distance(xyz[left], xyz[right], box)
                    distances.append(distance)
                    if pair[0] in conjugated_ids and pair[1] in conjugated_ids:
                        conj_distances.append(distance)
                    if pair[0] in aromatic_ids and pair[1] in aromatic_ids:
                        aromatic_distances.append(distance)
            nonlocal_values.extend(distances)
            conjugated_contacts.append(conj_distances)
            aromatic_contacts.append(aromatic_distances)
        if len(ring_atoms) > 1:
            ring_centers = [xyz[np.asarray(ids) - 1].mean(axis=0) for ids in ring_atoms]
            for left in range(len(ring_atoms)):
                for right in range(left + 1, len(ring_atoms)):
                    distance = min_image_distance(ring_centers[left], ring_centers[right], box)
                    if distance < 5.0:
                        ring_values.append(acute_plane_angle(xyz[np.asarray(ring_atoms[left]) - 1], xyz[np.asarray(ring_atoms[right]) - 1]))
                        if ring_values[-1] < 30.0:
                            stack_contacts.append(1)

    hard_runs = []
    soft_lengths = []
    for values in per_frame_thetas:
        binary = [value <= 30.0 for value in values]
        runs = []
        current = 0
        for state in binary:
            current = current + 1 if state else 0
            runs.append(current)
        hard_runs.append(max(runs or [0]))
        couplings = np.abs(np.cos(np.radians(values)))
        best = 0.0
        for start in range(len(couplings)):
            product = 1.0
            length = 1.0
            for coupling in couplings[start:]:
                product *= coupling
                length += product
                best = max(best, length)
        soft_lengths.append(best)

    flat_conj = [distance for values in conjugated_contacts for distance in values]
    flat_aromatic = [distance for values in aromatic_contacts for distance in values]
    result.update({
        "descriptor_status": "OK",
        "n_frames": len(frames),
        "conj_torsion_count": len(torsion_rows),
        "ring_pair_count": len(ring_values),
        "Rg": float(np.mean(rg_values)),
        "shape_anisotropy": float(np.mean(anisotropy_values)),
        "conj_contact_fraction_4p5": float(np.mean([sum(x < 4.5 for x in values) / max(len(values), 1) for values in conjugated_contacts])) if conjugated_contacts else np.nan,
        "conj_contact_fraction_6p0": float(np.mean([sum(x < 6.0 for x in values) / max(len(values), 1) for values in conjugated_contacts])) if conjugated_contacts else np.nan,
        "aromatic_contact_fraction_5p0": float(np.mean([sum(x < 5.0 for x in values) / max(len(values), 1) for values in aromatic_contacts])) if aromatic_contacts else np.nan,
        "mean_conj_nonlocal_distance": float(np.mean(flat_conj)) if flat_conj else np.nan,
        "minimum_conj_nonlocal_distance_mean": float(np.mean([min(values) for values in conjugated_contacts if values])) if any(conjugated_contacts) else np.nan,
        "aromatic_stacklike_contact_fraction": float(np.mean(stack_contacts)) if stack_contacts else np.nan,
        "generic_nonlocal_contact": float(np.mean([sum(x < 6.0 for x in nonlocal_values) / max(len(nonlocal_values), 1)])) if nonlocal_values else np.nan,
    })
    result.update(summary(torsion_values, "conj_twist"))
    result.update(summary(ring_values, "ring_twist"))
    result.update({
        "conj_planarity_mean": float(np.mean(np.abs(np.cos(np.radians(torsion_values))))) if torsion_values else np.nan,
        "conj_planar_fraction_20deg": float(np.mean(np.asarray(torsion_values) <= 20.0)) if torsion_values else np.nan,
        "conj_planar_fraction_30deg": float(np.mean(np.asarray(torsion_values) <= 30.0)) if torsion_values else np.nan,
        "conj_planar_fraction_40deg": float(np.mean(np.asarray(torsion_values) <= 40.0)) if torsion_values else np.nan,
        "conj_orthogonal_fraction_60deg": float(np.mean(np.asarray(torsion_values) >= 60.0)) if torsion_values else np.nan,
        "conj_torsion_entropy": np.nan,
        "ring_coplanar_fraction_20deg": float(np.mean(np.asarray(ring_values) <= 20.0)) if ring_values else np.nan,
        "ring_coplanar_fraction_30deg": float(np.mean(np.asarray(ring_values) <= 30.0)) if ring_values else np.nan,
        "conj_longest_planar_run_mean": float(np.mean(hard_runs)) if hard_runs else np.nan,
        "conj_longest_planar_run_std": float(np.std(hard_runs)) if hard_runs else np.nan,
        "conj_longest_planar_run_p90": float(np.percentile(hard_runs, 90)) if hard_runs else np.nan,
        "conj_soft_length_mean": float(np.mean(soft_lengths)) if soft_lengths else np.nan,
        "conj_soft_length_std": float(np.std(soft_lengths)) if soft_lengths else np.nan,
        "conj_soft_length_p90": float(np.percentile(soft_lengths, 90)) if soft_lengths else np.nan,
        "conj_soft_length_max_mean": float(np.mean(soft_lengths)) if soft_lengths else np.nan,
        "conj_torsion_transition_rate_frame": np.nan,
        "conj_torsion_state_lifetime_frames": np.nan,
        "conj_torsion_autocorrelation_lag_1": np.nan,
        "sp2_segment_planarity_mean": np.nan,
        "Re_over_Rg": np.nan,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    rows = []
    for system in sorted((args.root / "systems").iterdir()):
        if system.is_dir() and (system / "metadata" / "generation_status.json").exists():
            try:
                row = extract_system(system)
            except Exception as exc:
                row = {"polymer_id": system.name, "descriptor_status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            if row.get("descriptor_status") == "OK":
                pd.DataFrame([row]).to_parquet(system / "descriptors" / "conjugation_aware_descriptors.parquet", index=False)
    output = pd.DataFrame(rows)
    processed = args.project_root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    output.to_parquet(processed / "conjugation_aware_md_descriptors_pilot.parquet", index=False)
    output.to_parquet(args.root / "reports" / "descriptor_registry.parquet", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
