"""Conservative LAMMPS data/trajectory helpers for polymer MD audits."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterator

import numpy as np


SECTION_NAMES = {
    "Masses",
    "Pair Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "Atoms",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
    "Velocities",
}


@dataclass(frozen=True)
class LammpsTopology:
    path: Path
    counts: dict[str, int]
    box: np.ndarray
    masses: dict[int, float]
    type_labels: dict[int, str]
    atoms: dict[int, dict[str, float | int]]
    bonds: list[tuple[int, int, int]]
    angles: list[tuple[int, int, int, int]]
    dihedrals: list[tuple[int, int, int, int, int]]
    atom_style: str

    @property
    def atom_ids(self) -> list[int]:
        return sorted(self.atoms)


def _clean_section_name(line: str) -> str | None:
    text = line.strip().split("#", 1)[0].strip()
    return text if text in SECTION_NAMES else None


def parse_lammps_data(path: str | Path) -> LammpsTopology:
    path = Path(path)
    lines = path.read_text(errors="ignore").splitlines()
    counts: dict[str, int] = {}
    box = np.full((3, 2), np.nan, dtype=float)
    for line in lines[:200]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] in {"atoms", "bonds", "angles", "dihedrals", "impropers"}:
            counts[parts[1]] = int(parts[0])
        if len(parts) >= 4 and parts[2:4] == ["xlo", "xhi"]:
            box[0] = [float(parts[0]), float(parts[1])]
        if len(parts) >= 4 and parts[2:4] == ["ylo", "yhi"]:
            box[1] = [float(parts[0]), float(parts[1])]
        if len(parts) >= 4 and parts[2:4] == ["zlo", "zhi"]:
            box[2] = [float(parts[0]), float(parts[1])]

    sections: dict[str, list[str]] = defaultdict(list)
    current: str | None = None
    for line in lines:
        sec = _clean_section_name(line)
        if sec is not None:
            current = sec
            continue
        if current is None:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        sections[current].append(line)

    masses: dict[int, float] = {}
    type_labels: dict[int, str] = {}
    for line in sections.get("Masses", []):
        tokens = line.split("#", 1)
        parts = tokens[0].split()
        if len(parts) >= 2:
            atom_type = int(parts[0])
            masses[atom_type] = float(parts[1])
            if len(tokens) > 1:
                type_labels[atom_type] = tokens[1].strip()

    atoms: dict[int, dict[str, float | int]] = {}
    atom_style = "unknown"
    for line in sections.get("Atoms", []):
        parts = line.split("#", 1)[0].split()
        if len(parts) < 5:
            continue
        atom_id = int(parts[0])
        if len(parts) >= 7:
            atom_style = "full"
            atoms[atom_id] = {
                "molecule_id": int(parts[1]),
                "type": int(parts[2]),
                "charge": float(parts[3]),
                "x": float(parts[4]),
                "y": float(parts[5]),
                "z": float(parts[6]),
            }
        else:
            atom_style = "atomic"
            atoms[atom_id] = {"molecule_id": 0, "type": int(parts[1]), "x": float(parts[2]), "y": float(parts[3]), "z": float(parts[4])}

    bonds: list[tuple[int, int, int]] = []
    for line in sections.get("Bonds", []):
        parts = line.split("#", 1)[0].split()
        if len(parts) >= 4:
            bonds.append((int(parts[1]), int(parts[2]), int(parts[3])))

    angles: list[tuple[int, int, int, int]] = []
    for line in sections.get("Angles", []):
        parts = line.split("#", 1)[0].split()
        if len(parts) >= 5:
            angles.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])))

    dihedrals: list[tuple[int, int, int, int, int]] = []
    for line in sections.get("Dihedrals", []):
        parts = line.split("#", 1)[0].split()
        if len(parts) >= 6:
            dihedrals.append((int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])))

    return LammpsTopology(path, counts, box, masses, type_labels, atoms, bonds, angles, dihedrals, atom_style)


def adjacency_from_bonds(bonds: list[tuple[int, int, int]], atom_filter: set[int] | None = None) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = defaultdict(set)
    for _, a, b in bonds:
        if atom_filter is not None and (a not in atom_filter or b not in atom_filter):
            continue
        graph[a].add(b)
        graph[b].add(a)
    return graph


def connected_components(graph: dict[int, set[int]]) -> list[list[int]]:
    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in graph:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in graph[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        comps.append(sorted(comp))
    return sorted(comps, key=len, reverse=True)


def shortest_path(graph: dict[int, set[int]], start: int, end: int | None = None) -> tuple[int, list[int]]:
    parent = {start: None}
    queue: deque[int] = deque([start])
    last = start
    while queue:
        node = queue.popleft()
        last = node
        if end is not None and node == end:
            break
        for nbr in graph[node]:
            if nbr not in parent:
                parent[nbr] = node
                queue.append(nbr)
    target = end if end is not None else last
    if target not in parent:
        return 0, []
    path = [target]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return len(path) - 1, path


def principal_heavy_chain(topology: LammpsTopology) -> tuple[bool, list[int], str]:
    heavy = {aid for aid, atom in topology.atoms.items() if topology.masses.get(int(atom["type"]), 0.0) > 1.2}
    graph = adjacency_from_bonds(topology.bonds, heavy)
    comps = connected_components(graph)
    if not comps:
        return False, [], "no heavy-atom bonded component"
    comp = comps[0]
    if len(comp) < 4:
        return False, comp, "principal heavy component too small"
    sub = {node: graph[node] & set(comp) for node in comp}
    _, path1 = shortest_path(sub, comp[0])
    if not path1:
        return False, comp, "failed first graph traversal"
    _, diameter = shortest_path(sub, path1[-1])
    if len(diameter) < 4:
        return False, diameter, "diameter path too short"
    branch_nodes = sum(1 for node in comp if len(sub[node]) > 2)
    leaf_nodes = sum(1 for node in comp if len(sub[node]) == 1)
    reason = f"heavy-atom graph diameter; branches={branch_nodes}; leaves={leaf_nodes}; component_atoms={len(comp)}"
    return True, diameter, reason


def principal_heavy_chain_assessment(topology: LammpsTopology, psmiles: str | None = None) -> dict[str, object]:
    """Assess whether the heavy-atom graph diameter is a credible backbone proxy.

    This is deliberately conservative: a graph diameter can define an ordered
    contour, but branched repeat units make the chemically unique polymer
    backbone ambiguous unless additional atom mapping is available.
    """

    heavy = {aid for aid, atom in topology.atoms.items() if topology.masses.get(int(atom["type"]), 0.0) > 1.2}
    graph = adjacency_from_bonds(topology.bonds, heavy)
    comps = connected_components(graph)
    valid, path, reason = principal_heavy_chain(topology)
    if not valid or not comps:
        return {
            "backbone_method": "heavy_atom_graph_diameter",
            "backbone_valid": False,
            "backbone_confidence": "invalid",
            "backbone_ambiguous": True,
            "branch_count": 0,
            "backbone_n_atoms": len(path),
            "backbone_reason": reason,
            "backbone_atom_ids": path,
        }
    comp = comps[0]
    sub = {node: graph[node] & set(comp) for node in comp}
    branch_count = sum(1 for node in comp if len(sub[node]) > 2)
    leaf_count = sum(1 for node in comp if len(sub[node]) == 1)
    coverage = len(path) / max(len(comp), 1)
    stars_ok = bool(psmiles and str(psmiles).count("*") == 2)
    terminal_stars = bool(psmiles and str(psmiles).startswith("*") and str(psmiles).endswith("*"))
    if branch_count == 0 and leaf_count == 2 and coverage > 0.95:
        confidence = "high"
        ambiguous = False
    elif stars_ok and coverage >= 0.35 and branch_count <= max(80, 0.25 * len(comp)):
        confidence = "medium"
        ambiguous = branch_count > 0 or not terminal_stars
    else:
        confidence = "low"
        ambiguous = True
    return {
        "backbone_method": "heavy_atom_graph_diameter_with_psmiles_star_qc",
        "backbone_valid": True,
        "backbone_confidence": confidence,
        "backbone_ambiguous": ambiguous,
        "branch_count": branch_count,
        "backbone_n_atoms": len(path),
        "backbone_reason": f"{reason}; leaves={leaf_count}; diameter_coverage={coverage:.3f}; psmiles_two_stars={stars_ok}; terminal_stars={terminal_stars}",
        "backbone_atom_ids": path,
    }


def scan_lammpstrj_header(path: str | Path) -> dict[str, object]:
    path = Path(path)
    frames = 0
    timesteps: list[int] = []
    fields: list[str] = []
    atom_counts: set[int] = set()
    with path.open(errors="ignore") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if line.startswith("ITEM: TIMESTEP"):
                frames += 1
                ts = handle.readline().strip()
                if ts:
                    timesteps.append(int(float(ts)))
            elif line.startswith("ITEM: NUMBER OF ATOMS"):
                atom_counts.add(int(handle.readline().strip()))
            elif line.startswith("ITEM: ATOMS") and not fields:
                fields = line.strip().split()[2:]
    interval = float(np.median(np.diff(timesteps))) if len(timesteps) > 1 else math.nan
    return {"n_frames": frames, "timesteps": timesteps, "frame_interval_steps": interval, "fields": fields, "atom_counts": sorted(atom_counts)}


def iter_lammpstrj(path: str | Path, atom_ids: list[int] | None = None, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray, list[int]]]:
    wanted = set(atom_ids) if atom_ids is not None else None
    with Path(path).open(errors="ignore") as handle:
        emitted = 0
        while True:
            line = handle.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            timestep = int(float(handle.readline().strip()))
            handle.readline()
            n_atoms = int(handle.readline().strip())
            handle.readline()
            for _ in range(3):
                handle.readline()
            atom_header = handle.readline().strip().split()[2:]
            coord_names = [name for name in ("xu", "yu", "zu") if name in atom_header]
            if len(coord_names) != 3:
                coord_names = [name for name in ("x", "y", "z") if name in atom_header]
            if len(coord_names) != 3:
                raise ValueError(f"unsupported coordinate fields in {path}: {atom_header}")
            id_idx = atom_header.index("id")
            coord_idx = [atom_header.index(name) for name in coord_names]
            ids: list[int] = []
            xyz: list[list[float]] = []
            for _ in range(n_atoms):
                parts = handle.readline().split()
                atom_id = int(parts[id_idx])
                if wanted is None or atom_id in wanted:
                    ids.append(atom_id)
                    xyz.append([float(parts[i]) for i in coord_idx])
            order = np.argsort(ids)
            sorted_ids = [ids[i] for i in order]
            yield timestep, np.asarray(xyz, dtype=float)[order], sorted_ids
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                return
