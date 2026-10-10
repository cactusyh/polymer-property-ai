"""Versioned data contract and leakage audit for the existing polymer datasets.

This module never reads bulk holdout Tg values. The frozen bulk split contributes
identities only, so its members can be excluded before any model selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.polymer_smiles import stable_polymer_id
from src.phase6a import SCHEMA_VERSION

TASKS = (
    "Tg", "band_gap_chain", "band_gap_bulk", "electron_affinity",
    "ionization_energy", "atomization_energy",
)
MD_FEATURES = (
    ("log_Rg", "log_Rg"),
    ("Re/Rg", "Re_over_Rg"),
    ("q1", "q1"),
    ("q2", "q2"),
    ("anisotropy", "relative_shape_anisotropy"),
    ("contact_fraction", "short_range_contact_fraction"),
    ("torsion_entropy", "torsion_entropy"),
    ("log_torsion_transition_rate", "log_torsion_transition_rate"),
)
CONTRACT_COLUMNS = (
    "polymer_id", "raw_psmiles", "canonical_psmiles", "stereochemistry",
    "connection_count", "connection_positions", "structure_hash", "source_dataset",
    "source_sample_id", "source_provenance", "target_availability",
    "md_mapping_status", "md_mapping_confidence", "qc_status", "split_assignment",
)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def structure_hash(canonical: str) -> str:
    """Full hash of stereochemistry-aware canonical PSMILES and its two wildcards."""
    if not isinstance(canonical, str) or canonical.count("*") != 2:
        raise ValueError("A valid canonical PSMILES with two connections is required")
    return digest_bytes(canonical.encode())


def identity_fields(canonical: str) -> dict[str, object]:
    return {
        "polymer_id": stable_polymer_id(canonical),
        "structure_hash": structure_hash(canonical),
        "stereochemistry": "explicit" if any(x in canonical for x in ("@", "/", "\\")) else "unspecified",
        "connection_count": canonical.count("*"),
        "connection_positions": ";".join(str(i) for i, c in enumerate(canonical) if c == "*"),
    }


def validate_identity(frame: pd.DataFrame, source: str) -> None:
    valid = frame[frame.canonical_psmiles.notna()]
    bad = valid[valid.apply(lambda row: stable_polymer_id(row.canonical_psmiles) != row.polymer_id, axis=1)]
    if not bad.empty:
        raise ValueError(f"{source}: {len(bad)} canonical/polymer_id mismatches")
    if valid.canonical_psmiles.map(lambda s: s.count("*") != 2).any():
        raise ValueError(f"{source}: invalid connection-point count")
    if valid.groupby("polymer_id").canonical_psmiles.nunique().gt(1).any():
        raise ValueError(f"{source}: polymer_id collision")


def electronic_roles(splits: pd.DataFrame, protocol: str, fold: int = 0) -> pd.Series:
    """Reuse the recorded assignments; fold 0/1/2 are test/validation/train."""
    if protocol == "random":
        return splits.random_split
    if protocol != "ood" or fold not in (0, 1, 2):
        raise ValueError("protocol must be random or ood with fold 0, 1, or 2")
    test, validation = f"fold_{fold}", f"fold_{(fold + 1) % 3}"
    def role(value: str) -> str:
        if value in ("train", "validation", "test"):
            return value
        if value not in ("fold_0", "fold_1", "fold_2"):
            raise ValueError(f"Unknown official electronic OOD assignment: {value}")
        return "test" if value == test else "validation" if value == validation else "train"
    return splits.ood_split.map(role)


def split_sets(tg: pd.DataFrame, electronic: pd.DataFrame, splits: pd.DataFrame,
               protocol: str, fold: int = 0) -> dict[str, dict[str, set[str]]]:
    if protocol not in ("random", "ood"):
        raise ValueError(protocol)
    valid_tg = tg.loc[tg.valid & tg.canonical_psmiles.notna()]
    result = {"Tg": {role: set(valid_tg.loc[valid_tg[f"{protocol}_split"] == role, "polymer_id"])
                     for role in ("train", "validation", "test")}}
    joined = electronic.loc[electronic.valid_psmiles & electronic.target_value.notna()].merge(
        splits.loc[splits.seed == 13], on=["property_name", "canonical_psmiles", "sample_id"],
        validate="one_to_one", how="left", indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise ValueError("Electronic registry has missing official split assignments")
    joined = joined.assign(role=electronic_roles(joined, protocol, fold))
    for task in TASKS[1:]:
        part = joined.loc[joined.property_name == task]
        result[task] = {role: set(part.loc[part.role == role, "polymer_id"])
                        for role in ("train", "validation", "test")}
    for task, roles in result.items():
        if any(not roles[part] for part in ("train", "validation", "test")):
            raise ValueError(f"{task}: official split has an empty role")
        if (roles["train"] & roles["test"]) or (roles["train"] & roles["validation"]) or (roles["test"] & roles["validation"]):
            raise ValueError(f"{task}: exact identity leakage within official splits")
    return result


def md_mapping(md: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "polymer_id", "canonical_psmiles", "data_file", "trajectory_file",
               "mapping_verified", "mapping_method", "data_file_exists", "trajectory_file_exists"]
    joined = md.merge(mapping[columns], on="sample_id", validate="one_to_one", suffixes=("", "_map"), indicator=True)
    if len(joined) != len(md) or not joined._merge.eq("both").all():
        raise ValueError("MD mapping is incomplete or ambiguous")
    matches = {name: joined[name].eq(joined[f"{name}_map"])
               for name in ("polymer_id", "canonical_psmiles", "data_file", "trajectory_file")}
    verified = (joined.mapping_verified & joined.data_file_exists & joined.trajectory_file_exists
                & joined.mapping_method.eq("metadata_match") & pd.concat(matches.values(), axis=1).all(axis=1))
    joined["md_mapping_status"] = verified.map({True: "VERIFIED", False: "EXCLUDED_UNVERIFIED"})
    joined["md_mapping_confidence"] = verified.map({True: "high", False: "none"})
    return joined


def qc_mask(md: pd.DataFrame, source: str) -> pd.Series:
    return (md[f"{source}_valid"].fillna(False).astype(bool)
            & md[f"{source}_quality_class"].isin(["good", "borderline"])
            & pd.to_numeric(md[source], errors="coerce").notna()
            & pd.to_numeric(md[f"{source}_target_quality_weight"], errors="coerce").gt(0))


def md_coverage(md: pd.DataFrame) -> dict[str, int]:
    verified = md.loc[md.md_mapping_status == "VERIFIED"]
    return {name: int(qc_mask(verified, source).sum()) for name, source in MD_FEATURES}


def _fingerprint(canonical: str):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(canonical)
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol) if mol else None


def near_duplicate_count(train: set[str], evaluation: set[str], identities: dict[str, str],
                         threshold: float = 0.95) -> int:
    """Exact Morgan similarity screen for the labelled evaluation identities."""
    from rdkit import DataStructs
    base = [_fingerprint(identities[x]) for x in sorted(train) if x in identities]
    base = [x for x in base if x is not None]
    count = 0
    for polymer_id in sorted(evaluation):
        fp = _fingerprint(identities[polymer_id]) if polymer_id in identities else None
        if fp is not None and base and max(DataStructs.BulkTanimotoSimilarity(fp, base)) >= threshold:
            count += 1
    return count


def leakage_audit(roles: dict[str, dict[str, set[str]]], pi_ids: set[str], md_ids: set[str],
                  holdout: set[str], identities: dict[str, str], protocol: str,
                  near_duplicates: bool = True) -> dict[str, object]:
    blocked = holdout | set().union(*(x["validation"] | x["test"] for x in roles.values()))
    raw_train = set().union(*(x["train"] for x in roles.values()))
    safe_train = raw_train - blocked
    clean_pi = pi_ids - blocked
    production_pi = pi_ids - holdout
    if safe_train & blocked or clean_pi & blocked or production_pi & holdout:
        raise ValueError("PROHIBITED_OVERLAP: training/pretraining intersects evaluation or locked holdout")
    for task, split in roles.items():
        if any(split[a] & split[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"PROHIBITED_OVERLAP: {task} split leakage")
    evaluation = blocked - holdout
    task_ids = {task: set().union(*split.values()) for task, split in roles.items()}
    included = [task for task in TASKS if task in task_ids]
    cross = {f"{a}__{b}": len(task_ids[a] & task_ids[b])
             for i, a in enumerate(included) for b in included[i + 1:]}
    canonical_overlap = len({identities[x] for x in safe_train if x in identities}
                            & {identities[x] for x in blocked if x in identities})
    hash_overlap = len({structure_hash(identities[x]) for x in safe_train if x in identities}
                       & {structure_hash(identities[x]) for x in blocked if x in identities})
    if canonical_overlap or hash_overlap:
        raise ValueError("PROHIBITED_OVERLAP: canonical or structure-hash overlap")
    near = near_duplicate_count(safe_train, evaluation, identities) if near_duplicates else None
    return {
        "protocol": protocol, "status": "PASS", "identity_overlap_after_filter": len(safe_train & blocked),
        "canonical_overlap_after_filter": canonical_overlap,
        "structure_hash_overlap_after_filter": hash_overlap,
        "raw_pi1m_validation_test_overlap": len(pi_ids & evaluation),
        "clean_pi1m_validation_test_overlap": len(clean_pi & evaluation),
        "cross_task_identity_overlap": cross,
        "md_property_identity_overlap": {task: len(ids & md_ids) for task, ids in task_ids.items()},
        "locked_holdout_count": len(holdout), "locked_holdout_training_overlap": len(safe_train & holdout),
        "locked_holdout_clean_pretrain_overlap": len(clean_pi & holdout),
        "locked_holdout_production_pretrain_overlap": len(production_pi & holdout),
        "near_duplicate_evaluation_identities_at_0_95": near,
        "safe_supervised_train_ids": len(safe_train),
        "clean_transfer_pi1m_ids": len(clean_pi), "production_pi1m_ids": len(production_pi),
        "raw_training_conflicts_removed": len(raw_train & blocked),
    }


def source_contract(frame: pd.DataFrame, source: str, sample_col: str, target: pd.Series | str,
                    split: pd.Series, qc: pd.Series | str = "available",
                    mapping: pd.Series | str = "NOT_APPLICABLE",
                    confidence: pd.Series | str = "NOT_APPLICABLE",
                    provenance: pd.Series | str = "") -> pd.DataFrame:
    valid = frame.loc[frame.canonical_psmiles.notna()].copy()
    validate_identity(valid, source)
    out = valid[["polymer_id", "raw_psmiles", "canonical_psmiles"]].copy()
    canonical = out.canonical_psmiles
    out["stereochemistry"] = canonical.str.contains(r"@|/|\\", regex=True).map({True: "explicit", False: "unspecified"})
    out["connection_count"] = canonical.str.count(r"\*")
    out["connection_positions"] = canonical.map(lambda s: ";".join(str(i) for i, c in enumerate(s) if c == "*"))
    out["structure_hash"] = canonical.map(structure_hash)
    out["source_dataset"] = source
    out["source_sample_id"] = valid[sample_col].astype(str)
    out["source_provenance"] = provenance if isinstance(provenance, str) else provenance.loc[valid.index]
    out["target_availability"] = target if isinstance(target, str) else target.loc[valid.index]
    out["md_mapping_status"] = mapping if isinstance(mapping, str) else mapping.loc[valid.index]
    out["md_mapping_confidence"] = confidence if isinstance(confidence, str) else confidence.loc[valid.index]
    out["qc_status"] = qc if isinstance(qc, str) else qc.loc[valid.index]
    out["split_assignment"] = split.loc[valid.index].astype(str)
    return out[list(CONTRACT_COLUMNS)]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def build_manifest(root: Path, output: Path, near_duplicates: bool = True) -> dict[str, object]:
    """Read only real registries and write deterministic, machine-readable manifests."""
    data = root / "data/processed"
    paths = {
        "tg": data / "tg_registry.parquet",
        "electronic": data / "electronic_property_registry.parquet",
        "pi1m": data / "pi1m_no_tg.parquet",
        "md": data / "md_physics_full.parquet",
        "md_mapping": data / "md_psmiles_mapping.csv",
        "electronic_splits": root / "data/splits/electronic_property_splits.csv",
        "bulk_split": data / "bulk_md_tg_final_cohort_150_split.csv",
    }
    tg = pd.read_parquet(paths["tg"], columns=["tg_id", "raw_psmiles", "canonical_psmiles",
                                                   "polymer_id", "source", "valid", "random_split", "ood_split"])
    electronic = pd.read_parquet(paths["electronic"])
    pi = pd.read_parquet(paths["pi1m"], columns=["pi1m_id", "raw_psmiles", "canonical_psmiles", "polymer_id", "valid", "split"])
    md = md_mapping(pd.read_parquet(paths["md"]), pd.read_csv(paths["md_mapping"]))
    splits = pd.read_csv(paths["electronic_splits"])
    bulk = pd.read_csv(paths["bulk_split"], usecols=["polymer_id", "split"])
    holdout = set(bulk.loc[bulk.split == "LOCKED_HOLDOUT", "polymer_id"])
    if len(bulk) != 150 or len(holdout) != 30:
        raise ValueError("Frozen bulk cohort size or holdout split changed")
    for name, frame in (("tg", tg.loc[tg.valid]), ("electronic", electronic.loc[electronic.valid_psmiles]),
                        ("pi1m", pi.loc[pi.valid]), ("md", md)):
        validate_identity(frame, name)
    excluded_mapping = int(md.md_mapping_status.ne("VERIFIED").sum())
    roles = {name: split_sets(tg, electronic, splits, name) for name in ("random", "ood")}
    pi = pi.loc[pi.valid & pi.canonical_psmiles.notna()].copy()
    pi_ids = set(pi.polymer_id)
    md_ids = set(md.loc[md.md_mapping_status == "VERIFIED", "polymer_id"])
    identities = dict(zip(pd.concat([tg.polymer_id, electronic.polymer_id, md.polymer_id]),
                          pd.concat([tg.canonical_psmiles, electronic.canonical_psmiles, md.canonical_psmiles])))
    audits = {name: leakage_audit(role, pi_ids, md_ids, holdout, identities, name, near_duplicates)
              for name, role in roles.items()}
    coverage = md_coverage(md)
    output.mkdir(parents=True, exist_ok=True)
    split_records = []
    for protocol, task_roles in roles.items():
        blocked = holdout | set().union(*(r["validation"] | r["test"] for r in task_roles.values()))
        for policy, excluded in (("clean-transfer", blocked), ("production", holdout)):
            eligible = pi.loc[~pi.polymer_id.isin(excluded)].sort_values("pi1m_id")
            eligible.to_parquet(output / f"pi1m_{policy}_{protocol}.parquet", index=False)
            split_records.append({"protocol": protocol, "policy": policy, "rows": len(eligible),
                                  "excluded_identities_sha256": digest_bytes(canonical_json(sorted(excluded))),
                                  "unbiased_transfer": policy == "clean-transfer"})
    electronic_split = electronic.merge(
        splits.loc[splits.seed == 13, ["property_name", "canonical_psmiles", "sample_id", "random_split", "ood_split"]],
        on=["property_name", "canonical_psmiles", "sample_id"], how="left", validate="one_to_one",
    )
    if electronic_split.random_split.isna().any() or electronic_split.ood_split.isna().any():
        raise ValueError("Electronic split assignment is incomplete")
    md_feature_masks = {name: qc_mask(md, source).to_numpy(bool) for name, source in MD_FEATURES}
    md_availability = pd.Series([";".join(name for name, available in md_feature_masks.items()
                                          if available[i]) for i in range(len(md))], index=md.index)
    source_rows = [
        source_contract(tg.loc[tg.valid], "tg", "tg_id", "Tg",
                        tg.random_split.astype(str) + "|" + tg.ood_split.astype(str), provenance="data/raw/properties/tg/tg.csv"),
        source_contract(electronic_split.loc[electronic_split.valid_psmiles], "electronic", "sample_id",
                        electronic_split.property_name.astype(str),
                        electronic_split.random_split.astype(str) + "|" + electronic_split.ood_split.astype(str),
                        provenance=electronic_split.source_file),
        source_contract(md, "single_chain_md", "sample_id", md_availability,
                        pd.Series("unassigned", index=md.index), qc=md.overall_quality_flag,
                        mapping=md.md_mapping_status, confidence=md.md_mapping_confidence,
                        provenance=md.data_file),
        source_contract(pi, "pi1m_no_tg", "pi1m_id", "UNLABELED", pi.split,
                        provenance="data/processed/pi1m_no_tg.parquet"),
    ]
    contract = pd.concat(source_rows, ignore_index=True)
    contract.to_parquet(output / "identity_registry.parquet", index=False)
    verified_md = md.loc[md.md_mapping_status == "VERIFIED"]
    counts = {"tg_valid": int(tg.valid.sum()), "pi1m_valid": len(pi), "md_rows": len(md),
              "md_verified_mapping": len(verified_md), "md_excluded_mapping_mismatch": excluded_mapping,
              "md_at_least_one_usable_target": int(pd.concat([qc_mask(verified_md, source) for _, source in MD_FEATURES], axis=1).any(axis=1).sum()),
              "bulk_development": int(bulk.split.eq("DEVELOPMENT").sum()), "bulk_locked_holdout": len(holdout),
              "electronic": {task: int((electronic.property_name == task).sum()) for task in TASKS[1:]}}
    unit = {task: {"n_valid": counts["electronic"][task],
                   "raw_code": sorted(electronic.loc[electronic.property_name == task, "property_code"].dropna().astype(str).unique()),
                   "unit_metadata": sorted(electronic.loc[electronic.property_name == task, "target_unit"].dropna().astype(str).unique()),
                   "source": sorted(Path(p).name for p in electronic.loc[electronic.property_name == task, "source_file"].dropna().unique()),
                   "status": "EXPLORATORY_UNRESOLVED_UNITS" if task in ("electron_affinity", "ionization_energy", "atomization_energy") else "SEPARATE_TASK_UNRESOLVED_UNIT"}
            for task in TASKS[1:]}
    manifest = {"schema_version": SCHEMA_VERSION, "contract_columns": CONTRACT_COLUMNS,
                "input_sha256": {name: file_sha256(path) for name, path in paths.items()},
                "counts": counts, "md_feature_order": [name for name, _ in MD_FEATURES],
                "split_policy": split_records, "leakage_audit": audits, "md_coverage": coverage,
                "unit_audit": unit, "locked_holdout_policy": "identity exclusion only; no holdout Tg values read"}
    manifest_hash = digest_bytes(canonical_json(manifest))
    manifest["manifest_sha256"] = manifest_hash
    write_json(output / "manifest.v1.json", manifest)
    write_json(output / "leakage_audit.json", audits)
    write_json(output / "unit_audit.json", unit)
    write_json(output / "md_coverage.json", coverage)
    return manifest
