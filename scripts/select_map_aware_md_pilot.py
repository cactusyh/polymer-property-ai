#!/usr/bin/env python3
"""Select a method-development map-aware MD pilot and expansion queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def parse_motif(psmiles: str) -> dict:
    result = {
        "rdkit_parse": False,
        "aromatic_atom_indices": [],
        "aromatic_rings": [],
        "conjugated_bonds": [],
        "sp2_atom_indices": [],
        "sp2_sp2_bonds": [],
        "hetero_adjacent_to_conjugation": [],
        "attachment_atom_indices": [],
        "candidate_backbone_atom_indices": [],
        "candidate_conjugated_linker_bonds": [],
        "conjugation_paths": [],
        "ring_to_ring_linker_bonds": [],
    }
    mol = Chem.MolFromSmiles(str(psmiles).replace("[∗]", "*"))
    if mol is None:
        return result
    result["rdkit_parse"] = True
    result["aromatic_atom_indices"] = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    result["attachment_atom_indices"] = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    result["sp2_atom_indices"] = [
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetHybridization() == Chem.HybridizationType.SP2 or a.GetIsAromatic()
    ]
    for bond in mol.GetBonds():
        pair = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        if bond.GetIsConjugated():
            result["conjugated_bonds"].append(pair)
            if all(i in result["sp2_atom_indices"] for i in pair):
                result["sp2_sp2_bonds"].append(pair)
        if bond.GetIsConjugated() and any(mol.GetAtomWithIdx(i).GetSymbol() not in ("C", "H", "*") for i in pair):
            result["hetero_adjacent_to_conjugation"].extend(
                i for i in pair if mol.GetAtomWithIdx(i).GetSymbol() not in ("C", "H", "*")
            )
    result["hetero_adjacent_to_conjugation"] = sorted(set(result["hetero_adjacent_to_conjugation"]))
    rings = [list(r) for r in mol.GetRingInfo().AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    result["aromatic_rings"] = rings
    result["candidate_backbone_atom_indices"] = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() not in ("H", "*")
    ]
    ring_atoms = [set(r) for r in rings]
    for bond in mol.GetBonds():
        if not bond.GetIsConjugated():
            continue
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if any(begin in r for r in ring_atoms) and any(end in r for r in ring_atoms) and not any(begin in r and end in r for r in ring_atoms):
            result["ring_to_ring_linker_bonds"].append([begin, end])

    conjugated_graph = {i: set() for i in result["sp2_atom_indices"]}
    for begin, end in result["sp2_sp2_bonds"]:
        conjugated_graph.setdefault(begin, set()).add(end)
        conjugated_graph.setdefault(end, set()).add(begin)
    seen = set()
    path_id = 0
    for start in sorted(conjugated_graph):
        if start in seen:
            continue
        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(conjugated_graph.get(node, ()))
        if len(component) >= 2:
            edges = [b for b in result["sp2_sp2_bonds"] if b[0] in component and b[1] in component]
            result["conjugation_paths"].append({"path_id": path_id, "atom_indices": sorted(component), "bond_indices": edges})
            path_id += 1
    return result


def greedy_clusters(smiles: list[str], threshold: float = 0.65) -> list[int]:
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        fps.append(AllChem.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol) if mol else None)
    clusters = []
    representatives = []
    for idx, fp in enumerate(fps):
        cluster = None
        if fp is not None:
            for cid, rep in enumerate(representatives):
                if DataStructs.TanimotoSimilarity(fp, rep) >= threshold:
                    cluster = cid
                    break
        if cluster is None:
            cluster = len(representatives)
            representatives.append(fp)
        clusters.append(cluster)
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--expansion-size", type=int, default=200)
    args = parser.parse_args()
    root = args.root
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    props = pd.read_parquet(processed / "electronic_property_registry.parquet")
    props = props[props["property_name"].eq("band_gap_chain")].copy()
    props = props[props["valid_psmiles"].fillna(False)].drop_duplicates("polymer_id")
    motifs = pd.read_parquet(processed / "electronic_motif_registry.parquet")
    frame = props.merge(motifs, on=["polymer_id", "canonical_psmiles"], how="left", suffixes=("", "_motif"))
    frame["band_gap_value"] = frame["target_value"]
    frame["raw_psmiles"] = frame["raw_psmiles"].fillna(frame["canonical_psmiles"])
    frame["chemical_cluster_id"] = greedy_clusters(frame["canonical_psmiles"].tolist())
    frame["motif_score"] = (
        frame[["motif_aromatic", "motif_conjugated_bond", "motif_sp2", "motif_hetero_adjacent", "motif_conjugated_linker"]]
        .fillna(False).astype(int).sum(axis=1)
    )
    frame["gap_bin"] = pd.qcut(frame["band_gap_value"], q=3, labels=["low", "medium", "high"], duplicates="drop")
    frame = frame.sort_values(["motif_score", "band_gap_value", "polymer_id"], ascending=[False, True, True]).reset_index(drop=True)

    selected = []
    used_clusters = set()
    for gap_bin in ["low", "medium", "high"]:
        subset = frame[frame["gap_bin"].astype(str).eq(gap_bin)]
        for row in subset.itertuples():
            if row.chemical_cluster_id not in used_clusters or len(selected) < args.pilot_size:
                selected.append(row.Index)
                used_clusters.add(row.chemical_cluster_id)
            if len(selected) >= args.pilot_size:
                break
        if len(selected) >= args.pilot_size:
            break
    if len(selected) < args.pilot_size:
        selected.extend(i for i in frame.index if i not in selected)
    pilot_idx = selected[:args.pilot_size]
    expansion_idx = [i for i in frame.index if i not in pilot_idx][:args.expansion_size]

    def output(indices: list[int], label: str) -> pd.DataFrame:
        out = frame.loc[indices].copy()
        out["selection_reason"] = out.apply(
            lambda r: f"METHOD_DEVELOPMENT_MD_PILOT; motif_score={int(r.motif_score)}; gap_bin={r.gap_bin}; cluster={int(r.chemical_cluster_id)}",
            axis=1,
        )
        out["priority_rank"] = np.arange(1, len(out) + 1)
        out["pilot_label"] = label
        columns = [
            "polymer_id", "canonical_psmiles", "raw_psmiles", "band_gap_value",
            "motif_aromatic", "motif_conjugated_bond", "motif_sp2", "motif_hetero_adjacent",
            "motif_conjugated_linker", "chemical_cluster_id", "selection_reason", "priority_rank", "pilot_label",
        ]
        for col in columns:
            if col not in out:
                out[col] = False if col.startswith("motif_") else None
        return out[columns]

    pilot = output(pilot_idx, "METHOD_DEVELOPMENT_MD_PILOT")
    expansion = output(expansion_idx, "FUTURE_EXPANSION_QUEUE")
    pilot.to_csv(processed / "map_aware_md_pilot_20.csv", index=False)
    expansion.to_csv(processed / "map_aware_md_expansion_200_candidates.csv", index=False)

    annotation_rows = []
    for row in pd.concat([pilot, expansion], ignore_index=True).drop_duplicates("polymer_id").itertuples():
        annotation = parse_motif(row.canonical_psmiles)
        annotation_rows.append({"polymer_id": row.polymer_id, "canonical_psmiles": row.canonical_psmiles, **annotation})
        system = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/map_aware_single_chain_md_pilot") / "systems" / row.polymer_id / "metadata"
        system.mkdir(parents=True, exist_ok=True)
        (system / "rdkit_motif_annotation.json").write_text(json.dumps(annotation_rows[-1], indent=2), encoding="utf-8")
    pd.DataFrame(annotation_rows).to_parquet(processed / "map_aware_rdkit_motif_annotations.parquet", index=False)
    print(f"pilot={len(pilot)} expansion={len(expansion)} annotations={len(annotation_rows)}")


if __name__ == "__main__":
    main()
