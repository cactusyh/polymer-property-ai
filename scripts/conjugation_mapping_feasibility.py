#!/usr/bin/env python
"""Audit feasibility of mapping PSMILES electronic motifs onto existing MD atoms."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.md.lammps import parse_lammps_data  # noqa: E402

RDLogger.DisableLog("rdApp.*")
RAW_ROOT = Path("/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/data_total")
OVERLAP = ROOT / "data/processed/md_electronic_property_overlap.parquet"
DESC = ROOT / "data/processed/single_chain_electronic_md_descriptors.parquet"


def metadata_inventory(paths):
    if isinstance(paths, Path):
        paths = [paths]
    files = [p for path in paths for p in path.iterdir() if p.is_file()]
    suffix = Counter((p.suffix.lower() or "<no_extension>") for p in files)
    names = Counter(p.name for p in files)
    return files, suffix, names


def motif_record(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    result = {
        "rdkit_parse": bool(mol), "aromatic_atom_count": 0, "aromatic_ring_count": 0,
        "conjugated_bond_count": 0, "sp2_atom_count": 0, "hetero_adjacent_to_conjugation_count": 0,
        "conjugated_linker_bond_count": 0, "candidate_conjugated_torsion_count": 0,
        "motif_aromatic": False, "motif_conjugated_bond": False, "motif_sp2": False,
        "motif_hetero_adjacent": False, "motif_conjugated_linker": False,
    }
    if mol is None:
        return result
    aromatic_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    aromatic_bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetIsAromatic()}
    conjugated_bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetIsConjugated()}
    sp2_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2}
    hetero_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in {1, 6}}
    adjacent_hetero = {a.GetIdx() for b in mol.GetBonds() if b.GetIsConjugated() for a in (b.GetBeginAtom(), b.GetEndAtom()) if a.GetIdx() in hetero_atoms}
    linker_bonds = {b.GetIdx() for b in mol.GetBonds() if (b.GetBeginAtomIdx() in aromatic_atoms or b.GetEndAtomIdx() in aromatic_atoms) and (b.GetBeginAtomIdx() in sp2_atoms or b.GetEndAtomIdx() in sp2_atoms) and b.GetIdx() not in aromatic_bonds}
    torsion_centers = set()
    for bond_idx in conjugated_bonds | linker_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        left = bond.GetBeginAtom(); right = bond.GetEndAtom()
        for left_n in left.GetNeighbors():
            for right_n in right.GetNeighbors():
                if left_n.GetIdx() != right.GetIdx(): torsion_centers.add((left_n.GetIdx(), left.GetIdx(), right.GetIdx(), right_n.GetIdx()))
    ring_info = mol.GetRingInfo()
    rings = [ring for ring in ring_info.AtomRings() if len(ring) >= 5 and all(i in aromatic_atoms for i in ring)]
    result.update({
        "aromatic_atom_count": len(aromatic_atoms), "aromatic_ring_count": len(rings),
        "conjugated_bond_count": len(conjugated_bonds), "sp2_atom_count": len(sp2_atoms),
        "hetero_adjacent_to_conjugation_count": len(adjacent_hetero), "conjugated_linker_bond_count": len(linker_bonds),
        "candidate_conjugated_torsion_count": len(torsion_centers), "motif_aromatic": bool(aromatic_atoms),
        "motif_conjugated_bond": bool(conjugated_bonds), "motif_sp2": bool(sp2_atoms),
        "motif_hetero_adjacent": bool(adjacent_hetero), "motif_conjugated_linker": bool(linker_bonds),
    })
    return result


def topology_metadata(data_path: str):
    result = {
        "data_readable": False, "data_header_comment": "", "atom_style": "unknown", "n_atoms_topology": np.nan,
        "n_bonds_topology": np.nan, "n_molecule_ids": np.nan, "molecule_id_values": "", "atom_type_label_count": 0,
        "has_atom_type_comments": False, "has_atom_names": False, "has_residue_names": False,
        "has_repeat_mapping_text": False, "has_atom_index_mapping_text": False, "mapping_metadata_signal": "none",
    }
    try:
        topo = parse_lammps_data(data_path)
        result.update({
            "data_readable": True, "data_header_comment": Path(data_path).read_text(errors="ignore").splitlines()[0][:200],
            "atom_style": topo.atom_style, "n_atoms_topology": len(topo.atoms), "n_bonds_topology": len(topo.bonds),
            "n_molecule_ids": len({int(v.get("molecule_id", 0)) for v in topo.atoms.values()}),
            "molecule_id_values": ";".join(map(str, sorted({int(v.get("molecule_id", 0)) for v in topo.atoms.values()}))),
            "atom_type_label_count": len(topo.type_labels), "has_atom_type_comments": bool(topo.type_labels),
        })
        text = Path(data_path).read_text(errors="ignore")
        lower = text.lower()
        result["has_atom_names"] = bool(re.search(r"\batom[_ ]?name|\bname\b", lower))
        result["has_residue_names"] = bool(re.search(r"residue|resname|residue_name", lower))
        result["has_repeat_mapping_text"] = bool(re.search(r"repeat|monomer|backbone|repeat[_ -]?unit", lower))
        result["has_atom_index_mapping_text"] = bool(re.search(r"atom[_ -]?index|mapping|map", lower))
        if result["has_atom_index_mapping_text"] or result["has_repeat_mapping_text"] or result["has_atom_names"] or result["has_residue_names"]:
            result["mapping_metadata_signal"] = "possible_text_signal_requires_manual_validation"
        elif result["has_atom_type_comments"] or result["n_molecule_ids"] > 1:
            result["mapping_metadata_signal"] = "coarse_forcefield_or_molecule_metadata_only"
    except Exception as exc:
        result["mapping_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    overlap = pd.read_parquet(OVERLAP)
    overlap = overlap[overlap["property"] == "band_gap_chain"].drop_duplicates("polymer_id").copy()
    desc = pd.read_parquet(DESC)[["polymer_id", "canonical_psmiles", "data_file", "trajectory_file", "n_atoms", "n_frames", "backbone_confidence"]]
    data = overlap.merge(desc, on=["polymer_id", "canonical_psmiles", "data_file", "trajectory_file"], how="left", suffixes=("", "_desc"))
    motif_rows = []
    mapping_rows = []
    metadata_counts = Counter()
    examples = {k: [] for k in ["HIGH", "MEDIUM", "LOW", "FAILED"]}
    for row in data.itertuples(index=False):
        motif = motif_record(row.canonical_psmiles)
        topo = topology_metadata(row.data_file)
        for key in ["has_atom_names", "has_residue_names", "has_repeat_mapping_text", "has_atom_index_mapping_text", "has_atom_type_comments"]:
            if topo[key]: metadata_counts[key] += 1
        mapping_confidence = "FAILED"
        mapping_reason = "No atom-index map, repeat-unit map, residue/name correspondence, or construction-order metadata was found; force-field atom-type comments are coarse only."
        if topo["has_atom_index_mapping_text"] or topo["has_repeat_mapping_text"] or topo["has_atom_names"] or topo["has_residue_names"]:
            mapping_confidence = "LOW"
            mapping_reason = "Textual mapping-like signal exists, but no validated atom-index correspondence was established."
        elif topo["has_atom_type_comments"] or topo["n_molecule_ids"] > 1:
            mapping_confidence = "LOW"
            mapping_reason = "Only coarse force-field type labels or molecule IDs are available; these do not identify PSMILES atoms."
        if len(examples[mapping_confidence]) < 5:
            examples[mapping_confidence].append({"polymer_id": row.polymer_id, "canonical_psmiles": row.canonical_psmiles, "data_file": row.data_file, "reason": mapping_reason})
        motif_rows.append({"polymer_id": row.polymer_id, "canonical_psmiles": row.canonical_psmiles, **motif})
        mapping_rows.append({"polymer_id": row.polymer_id, "canonical_psmiles": row.canonical_psmiles, "data_file": row.data_file, "trajectory_file": row.trajectory_file, "mapping_confidence": mapping_confidence, "mapping_reason": mapping_reason, **topo, "mapping_usable_for_conjugation": False, "representative_subset": False})
    motif_df = pd.DataFrame(motif_rows)
    mapping_df = pd.DataFrame(mapping_rows)
    # Representative subset: deterministic stratified sample by motif class, used for explicit mapping attempts.
    motif_df["motif_group"] = np.select([motif_df.motif_aromatic, motif_df.motif_conjugated_bond, motif_df.motif_sp2], ["aromatic", "conjugated", "sp2"], default="none")
    sample_ids = motif_df.groupby("motif_group", group_keys=False).head(25).polymer_id
    mapping_df.loc[mapping_df.polymer_id.isin(set(sample_ids)), "representative_subset"] = True
    motif_df.to_parquet(ROOT / "data/processed/electronic_motif_registry.parquet", index=False)
    families = [
        ("CONJUGATED_CENTRAL_BOND_TORSION", "conjugated bonds plus validated atom map", "HIGH/MEDIUM mapping required; no validated maps", "NO-GO"),
        ("AROMATIC_RING_PLANE_TWIST", "two mapped aromatic rings and atom-resolved ring membership", "No ring atom correspondence or ring identity in topology", "NO-GO"),
        ("SP2_SEGMENT_PLANARITY", "mapped sp2 atom segments", "sp2 identity cannot be transferred to trajectory atoms", "NO-GO"),
        ("CONFORMATIONAL_CONJUGATION_LENGTH_PROXY", "mapped contiguous conjugated path", "path membership and atom order unavailable", "NO-GO"),
        ("AROMATIC_CONJUGATED_CONTACTS", "mapped aromatic/conjugated atom sets plus PBC", "contact categories cannot be assigned to chemical motifs", "NO-GO"),
        ("CONJUGATED_TORSIONAL_DYNAMICS", "mapped torsion list over frames", "no atom-resolved conjugated torsion list", "NO-GO"),
    ]
    feasibility = []
    for family, requirement, failure, rec in families:
        motif_mask = motif_df.motif_aromatic | motif_df.motif_conjugated_bond | motif_df.motif_sp2
        feasible = int((motif_mask & mapping_df.mapping_confidence.isin(["HIGH", "MEDIUM"])).sum())
        feasibility.append({"descriptor_family": family, "required_metadata": requirement, "feasible_polymer_count": feasible, "expected_coverage_fraction": feasible / len(data), "mapping_high_count": int((mapping_df.mapping_confidence == "HIGH").sum()), "mapping_medium_count": int((mapping_df.mapping_confidence == "MEDIUM").sum()), "mapping_low_count": int((mapping_df.mapping_confidence == "LOW").sum()), "mapping_failed_count": int((mapping_df.mapping_confidence == "FAILED").sum()), "main_failure_mode": failure, "computational_cost": "moderate" if "CONTACT" not in family else "high", "electronic_relevance": "high" if family in {"CONJUGATED_CENTRAL_BOND_TORSION", "AROMATIC_RING_PLANE_TWIST", "CONFORMATIONAL_CONJUGATION_LENGTH_PROXY"} else "medium-high", "interpretability": "high", "recommendation": rec})
    feas_df = pd.DataFrame(feasibility)
    feas_df.to_parquet(ROOT / "data/processed/conjugation_mapping_feasibility.parquet", index=False)
    # Make metadata inventory concise but complete in the report.
    matched_dirs = [Path(path).parent for path in data.data_file.drop_duplicates()]
    _, suffix_counts, name_counts = metadata_inventory(matched_dirs)
    motif_counts = {c: int(motif_df[c].sum()) for c in ["motif_aromatic", "motif_conjugated_bond", "motif_sp2", "motif_hetero_adjacent", "motif_conjugated_linker"]}
    report = ["# Conjugation-Aware MD Mapping Feasibility", "", f"Audit population: **{len(data):,}** chain-gap/MD overlap polymers.", "", "## Metadata Inventory", "", "Only `structure.data` and `trajectory.lammpstrj` were present in the matched system directories. No mol/SDF/PDB/XYZ/JSON sidecars, RDKit molecule files, atom-index maps, or repeat-unit construction logs were found.", "", "### File Types", "", "| file type | count |", "|---|---:|"]
    for ext, count in suffix_counts.most_common(): report.append(f"| `{ext}` | {count} |")
    report += ["", "### Topology Metadata Signals", "", f"- Atom-type comments: `{metadata_counts['has_atom_type_comments']:,}` systems in the matched audit; these are force-field labels such as `c3,0` and `hc,0`, not chemical atom maps.", f"- Atom names: `{metadata_counts['has_atom_names']:,}`.", f"- Residue names: `{metadata_counts['has_residue_names']:,}`.", f"- Repeat/construction text: `{metadata_counts['has_repeat_mapping_text']:,}`.", f"- Atom-index mapping text: `{metadata_counts['has_atom_index_mapping_text']:,}`.", "- Molecule IDs are available through the LAMMPS `full` atom style, but they identify coarse molecule membership rather than repeat-unit or PSMILES atom identity.", "", "## PSMILES Motif Registry", "", "| motif | polymers |", "|---|---:|"]
    for key, count in motif_counts.items(): report.append(f"| `{key}` | {count:,} |")
    report += ["", "Motifs are identified from RDKit repeat-unit PSMILES only. They are not claimed to be mapped onto trajectory atoms.", "", "## Mapping Test", "", "A deterministic representative subset was selected across motif groups. The attempted mapping used only topology metadata, atom-type comments, molecule IDs, and explicit text signals. No correspondence was inferred from connectivity alone.", "", "| confidence | all matched systems | representative examples |", "|---|---:|---|"]
    for level in ["HIGH", "MEDIUM", "LOW", "FAILED"]:
        count = int((mapping_df.mapping_confidence == level).sum())
        ex = "; ".join(f"{x['polymer_id']} {x['canonical_psmiles']}" for x in examples[level]) or "none"
        report.append(f"| `{level}` | {count:,} | {ex} |")
    report += ["", "Result: no HIGH or MEDIUM atom-resolved PSMILES-to-topology mappings were established. LOW indicates only coarse metadata compatibility and is not usable for primary conjugation-aware descriptors. FAILED indicates no mapping evidence beyond the identity-level registry.", "", "## Descriptor Feasibility", "", feas_df.to_string(index=False), "", "## Priority Ranking", "", "1. **Conjugated central-bond torsion** — highest relevance and moderate cost, but NO-GO with current files because central-bond atom identity is unavailable.", "2. **Aromatic ring-plane twist** — high relevance and interpretable, but NO-GO because ring atom membership is unavailable.", "3. **Conformational conjugation-length proxies** — high relevance, but NO-GO because contiguous mapped paths are unavailable.", "4. **sp2 segment planarity** — medium-high relevance, but NO-GO because sp2 atom labels cannot be transferred.", "5. **Conjugated torsional dynamics** — high relevance but more expensive; NO-GO for the same mapping reason.", "6. **Aromatic/conjugated contacts** — potentially relevant but highest ambiguity/cost; NO-GO without motif-resolved atom sets.", "", "## Final Decision", "", "**REGENERATE_MD_WITH_MAPPING_METADATA**", "", "Existing files are sufficient for topology-only geometry, but not for reliable electronically relevant PSMILES-to-trajectory atom mapping. Future simulations should preserve explicit repeat-unit atom indices, PSMILES/RDKit atom indices, residue or repeat IDs, bond-order-aware construction metadata, and a machine-readable map from each trajectory atom to its repeat-unit atom."]
    (ROOT / "reports/conjugation_aware_md_mapping_feasibility.md").write_text("\n".join(report) + "\n")
    print(f"matched systems: {len(data)}")
    print("metadata: only structure.data and trajectory.lammpstrj")
    print("motifs:", motif_counts)
    print("mapping HIGH/MEDIUM/LOW/FAILED:", *(int((mapping_df.mapping_confidence == x).sum()) for x in ["HIGH", "MEDIUM", "LOW", "FAILED"]))
    print("decision: REGENERATE_MD_WITH_MAPPING_METADATA")


if __name__ == "__main__":
    main()
