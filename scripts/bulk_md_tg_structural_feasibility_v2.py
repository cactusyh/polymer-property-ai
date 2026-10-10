#!/usr/bin/env python3
"""Input-only structural feasibility audit for the 12-polymer bulk Tg pilot.

This script deliberately reads no Tg/property/model field and does not build cells
or run MD. It reproduces the original Morgan/Butina/medoid selection first, then
uses RadonPy only to parse each repeat unit and count its atoms.
"""
from pathlib import Path
import math
import sys
from contextlib import contextmanager

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/properties/tg/tg.csv"
PILOT_PATH = ROOT / "data/processed/bulk_md_tg_pilot_12.csv"
DETAIL_PATH = ROOT / "data/processed/bulk_md_tg_structural_feasibility.csv"
FEASIBLE_PATH = ROOT / "data/processed/bulk_md_tg_pilot_12_feasible.csv"
REPORT_PATH = ROOT / "reports/bulk_md_tg_structural_feasibility.md"
N_CHAINS = 6
TARGET_ATOMS = 7500
MIN_ATOMS = 5000
MAX_ATOMS = 10000
MIN_DP = 10
RDLogger.DisableLog("rdApp.*")


def load_structures():
    """Read the raw structural field only, then canonicalize it."""
    raw = pd.read_csv(RAW_PATH, usecols=["SMILES"])
    sys.path.insert(0, str(ROOT))
    from src.data.polymer_smiles import canonicalize_psmiles

    records = [canonicalize_psmiles(value, source="tg") for value in raw["SMILES"]]
    table = pd.DataFrame(
        {
            "polymer_id": [record.polymer_id for record in records],
            "canonical_psmiles": [record.canonical_psmiles for record in records],
            "valid": [record.valid for record in records],
        }
    )
    raw_count = len(table)
    valid_count = int(table["valid"].sum())
    table = (
        table[table.valid & table.canonical_psmiles.notna()]
        .drop_duplicates("canonical_psmiles")
        .sort_values("canonical_psmiles")
        .reset_index(drop=True)
    )
    return table, raw_count, valid_count


def cluster_and_medoids(table):
    """Exactly reproduce the original Morgan-2048/r=2 Butina procedure."""
    mols = [Chem.MolFromSmiles(smiles) for smiles in table.canonical_psmiles]
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]

    # Same lower-triangle ordering and Tanimoto distances as the original script.
    n_fingerprints = len(fingerprints)
    distances = np.empty(n_fingerprints * (n_fingerprints - 1) // 2, dtype=np.float64)
    offset = 0
    for i in range(1, n_fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances[offset : offset + i] = 1.0 - np.asarray(similarities, dtype=np.float64)
        offset += i
    groups = Butina.ClusterData(distances, n_fingerprints, 0.65, isDistData=True)

    table = table.copy()
    table["cluster_id"] = -1
    medoids = []
    for cluster_id, group in enumerate(groups):
        members = list(group)
        table.loc[members, "cluster_id"] = cluster_id
        scores = [
            sum(DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j]) for j in members)
            / len(members)
            for i in members
        ]
        medoid_index = members[int(np.argmax(scores))]
        medoids.append((cluster_id, medoid_index, len(members)))
    return table, fingerprints, medoids


def original_selection(table, medoids):
    all_medoids = pd.DataFrame(
        [
            {
                "polymer_id": table.iloc[index].polymer_id,
                "canonical_psmiles": table.iloc[index].canonical_psmiles,
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
            }
            for cluster_id, index, cluster_size in medoids
        ]
    )
    return (
        all_medoids.sort_values(["cluster_size", "canonical_psmiles"], ascending=[False, True])
        .head(12)
        .sort_values("cluster_id")
        .reset_index(drop=True)[["polymer_id", "canonical_psmiles", "cluster_id"]]
    )


def choose_dp(repeat_unit_atoms):
    """Return the feasible integer DP whose six-chain atom count is nearest 7500."""
    lower = max(MIN_DP, math.ceil(MIN_ATOMS / (N_CHAINS * repeat_unit_atoms)))
    upper = math.floor(MAX_ATOMS / (N_CHAINS * repeat_unit_atoms))
    if lower > upper:
        return None
    ideal = TARGET_ATOMS / (N_CHAINS * repeat_unit_atoms)
    candidates = sorted({lower, upper, max(lower, min(upper, math.floor(ideal))), max(lower, min(upper, math.ceil(ideal)))})
    return min(candidates, key=lambda dp: (abs(N_CHAINS * repeat_unit_atoms * dp - TARGET_ATOMS), dp))


@contextmanager
def radonpy_atom_count_parse_mode():
    """Use RadonPy parsing without stereoisomer 3-D embedding."""
    from radonpy.core import utils
    enumerate_module = utils.Chem.EnumerateStereoisomers
    original = enumerate_module.EnumerateStereoisomers
    enumerate_module.EnumerateStereoisomers = lambda mol, options=None: (mol,)
    try:
        yield utils
    finally:
        enumerate_module.EnumerateStereoisomers = original


def assess_feasibility(table):
    """Call RadonPy parser only; no polymer construction or MD is performed."""
    counts, dps, estimates, feasible, failures = [], [], [], [], []
    with radonpy_atom_count_parse_mode() as utils:
        for smiles in table.canonical_psmiles:
            try:
                repeat_unit_atoms = int(utils.mol_from_smiles(smiles, coord=False).GetNumAtoms())
                dp = choose_dp(repeat_unit_atoms)
                if dp is None:
                    counts.append(repeat_unit_atoms)
                    dps.append(np.nan)
                    estimates.append(np.nan)
                    feasible.append(False)
                    failures.append("no_integer_DP_ge_10_yields_5000_to_10000_atoms_for_6_chains")
                else:
                    counts.append(repeat_unit_atoms)
                    dps.append(dp)
                    estimates.append(N_CHAINS * repeat_unit_atoms * dp)
                    feasible.append(True)
                    failures.append("")
            except Exception as exc:
                counts.append(np.nan)
                dps.append(np.nan)
                estimates.append(np.nan)
                feasible.append(False)
                failures.append(f"RadonPy_monomer_parse_failed:{type(exc).__name__}")
    result = table.copy()
    result["repeat_unit_atoms"] = counts
    result["selected_DP"] = dps
    result["estimated_total_atoms"] = estimates
    result["feasible_6chain"] = feasible
    result["failure_reason"] = failures
    return result

def make_feasible_pilot(table, fingerprints, medoids, expected):
    """Apply the prescribed retain/within-cluster/whole-cluster replacement rule."""
    cluster_members = {
        cluster_id: np.flatnonzero(table.cluster_id.to_numpy() == cluster_id).tolist()
        for cluster_id, _, _ in medoids
    }
    medoid_by_cluster = {cluster_id: index for cluster_id, index, _ in medoids}
    size_by_cluster = {cluster_id: size for cluster_id, _, size in medoids}
    original_indices = {
        polymer_id: int(table.index[table.polymer_id.eq(polymer_id)][0])
        for polymer_id in expected.polymer_id
    }

    selected = []
    exhausted_original_clusters = []
    occupied_clusters = set()
    for row in expected.itertuples(index=False):
        original_index = original_indices[row.polymer_id]
        members = cluster_members[int(row.cluster_id)]
        if bool(table.loc[original_index, "feasible_6chain"]):
            selected.append((original_index, int(row.cluster_id), row.polymer_id, "original_medoid_feasible"))
            occupied_clusters.add(int(row.cluster_id))
            continue
        feasible_members = [index for index in members if bool(table.loc[index, "feasible_6chain"])]
        if feasible_members:
            selected_index = min(
                feasible_members,
                key=lambda index: (1.0 - DataStructs.TanimotoSimilarity(fingerprints[original_index], fingerprints[index]), index),
            )
            selected.append((selected_index, int(row.cluster_id), row.polymer_id, "original_medoid_infeasible_replaced_with_nearest_feasible_same_cluster"))
            occupied_clusters.add(int(row.cluster_id))
        else:
            exhausted_original_clusters.append((int(row.cluster_id), row.polymer_id))

    available_clusters = []
    for cluster_id, medoid_index, cluster_size in medoids:
        if cluster_id in occupied_clusters:
            continue
        feasible_members = [index for index in cluster_members[cluster_id] if bool(table.loc[index, "feasible_6chain"])]
        if feasible_members:
            available_clusters.append((cluster_size, table.iloc[medoid_index].canonical_psmiles, cluster_id, medoid_index, feasible_members))
    available_clusters.sort(key=lambda item: (-item[0], item[1]))
    if len(available_clusters) < len(exhausted_original_clusters):
        raise RuntimeError("fewer feasible replacement clusters than wholly infeasible selected clusters")

    for (old_cluster_id, original_polymer_id), (_, _, new_cluster_id, medoid_index, feasible_members) in zip(exhausted_original_clusters, available_clusters):
        selected_index = min(
            feasible_members,
            key=lambda index: (1.0 - DataStructs.TanimotoSimilarity(fingerprints[medoid_index], fingerprints[index]), index),
        )
        selected.append(
            (
                selected_index,
                new_cluster_id,
                original_polymer_id,
                f"original_cluster_{old_cluster_id}_has_no_feasible_member_replaced_by_next_largest_feasible_cluster_{new_cluster_id}",
            )
        )

    rows = []
    for index, cluster_id, original_medoid, reason in selected:
        row = table.loc[index]
        rows.append(
            {
                "polymer_id": row.polymer_id,
                "canonical_psmiles": row.canonical_psmiles,
                "cluster_id": cluster_id,
                "original_medoid": original_medoid,
                "replacement_reason": reason,
                "repeat_unit_atoms": int(row.repeat_unit_atoms),
                "selected_DP": int(row.selected_DP),
                "estimated_total_atoms": int(row.estimated_total_atoms),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def write_report(table, pilot, raw_count, valid_count):
    retained = int((pilot.replacement_reason == "original_medoid_feasible").sum())
    within = int(pilot.replacement_reason.str.contains("same_cluster").sum())
    whole = int(pilot.replacement_reason.str.contains("next_largest_feasible_cluster").sum())
    failures = table[~table.feasible_6chain]
    parsing_failures = int(failures.failure_reason.str.startswith("RadonPy_monomer_parse_failed", na=False).sum())
    buildability_failures = int(len(failures) - parsing_failures)
    failure_summary = failures.failure_reason[failures.failure_reason.ne("")].value_counts().sort_index()
    lines = [
        "# Bulk MD Tg Structural Feasibility Audit",
        "",
        f"- Raw structure count: **{raw_count}**",
        f"- Valid structure count: **{valid_count}**",
        f"- Unique canonical count: **{len(table)}**",
        "- ORIGINAL_SELECTION_REPRODUCED = **YES**",
        "",
        f"- Retained original medoids: **{retained}**",
        f"- Within-cluster replacements: **{within}**",
        f"- Whole-cluster replacements: **{whole}**",
        f"- Repeat-unit atom-count range: **{int(table.repeat_unit_atoms.min())} / {int(table.repeat_unit_atoms.max())}**",
        f"- DP min / median / max: **{int(pilot.selected_DP.min())} / {pilot.selected_DP.median():.1f} / {int(pilot.selected_DP.max())}**",
        f"- Estimated bulk atom-count min / median / max: **{int(pilot.estimated_total_atoms.min())} / {pilot.estimated_total_atoms.median():.1f} / {int(pilot.estimated_total_atoms.max())}**",
        f"- Parsing/buildability failures: **{len(failures)}** ({parsing_failures} parsing, {buildability_failures} no feasible DP)",
        "",
        "## Failure reasons",
        "",
    ]
    if failure_summary.empty:
        lines.append("None.")
    else:
        lines.extend(f"- `{reason}`: {count}" for reason, count in failure_summary.items())
    lines.extend(
        [
            "",
            "Only the raw SMILES field was read from the Tg CSV. No Tg/value or model information was read. No bulk cell was constructed and no MD was run.",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    table, raw_count, valid_count = load_structures()
    table, fingerprints, medoids = cluster_and_medoids(table)
    expected = pd.read_csv(PILOT_PATH, usecols=["polymer_id", "canonical_psmiles", "cluster_id"])
    expected = expected.sort_values("cluster_id").reset_index(drop=True)
    reproduced = original_selection(table, medoids)

    # No feasibility parsing, replacement, or output generation occurs before this exact gate.
    if not expected.equals(reproduced):
        print("ORIGINAL_SELECTION_REPRODUCED = NO")
        print("Expected polymer IDs:")
        print(expected.polymer_id.to_string(index=False))
        print("Reproduced polymer IDs:")
        print(reproduced.polymer_id.to_string(index=False))
        raise SystemExit(2)
    print("ORIGINAL_SELECTION_REPRODUCED = YES")

    table = assess_feasibility(table)
    table[
        [
            "polymer_id", "canonical_psmiles", "cluster_id", "repeat_unit_atoms", "selected_DP",
            "estimated_total_atoms", "feasible_6chain", "failure_reason",
        ]
    ].to_csv(DETAIL_PATH, index=False)
    pilot = make_feasible_pilot(table, fingerprints, medoids, expected)
    pilot.to_csv(FEASIBLE_PATH, index=False)
    write_report(table, pilot, raw_count, valid_count)
    print(pilot.to_string(index=False))
    print({
        "ORIGINAL_SELECTION_REPRODUCED": "YES",
        "retained_original_medoids": int((pilot.replacement_reason == "original_medoid_feasible").sum()),
        "within_cluster_replacements": int(pilot.replacement_reason.str.contains("same_cluster").sum()),
        "whole_cluster_replacements": int(pilot.replacement_reason.str.contains("next_largest_feasible_cluster").sum()),
    })


if __name__ == "__main__":
    main()
