"""Real-data task runner; guarded so full training follows manifest and smoke gates."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.psmiles_tokenizer import tokenize_psmiles
from src.phase6a.benchmark import checkpoint_hash, regression_metrics, result_record
from src.phase6a.contract import (
    MD_FEATURES,
    TASKS,
    canonical_json,
    md_mapping,
    qc_mask,
    split_sets,
)
from src.phase6a.models import (
    MorganEncoder,
    PolymerEncoderAdapter,
    TrainOnlyNormalizer,
    UnifiedPolymerModel,
)

SPECIAL = ["<PAD>", "<MASK>", "<UNK>", "<CLS>", "<SEP>"]


def verified_manifest(path: Path) -> dict[str, object]:
    import hashlib
    value = json.loads(path.read_text())
    received = value.pop("manifest_sha256")
    expected = hashlib.sha256(canonical_json(value)).hexdigest()
    if received != expected:
        raise ValueError("Data manifest hash mismatch")
    root = path.resolve().parents[1]
    from src.phase6a.contract import file_sha256
    for source, recorded_hash in value["input_sha256"].items():
        locations = {
            "tg": "data/processed/tg_registry.parquet",
            "electronic": "data/processed/electronic_property_registry.parquet",
            "pi1m": "data/processed/pi1m_no_tg.parquet",
            "md": "data/processed/md_physics_full.parquet",
            "md_mapping": "data/processed/md_psmiles_mapping.csv",
            "electronic_splits": "data/splits/electronic_property_splits.csv",
            "bulk_split": "data/processed/bulk_md_tg_final_cohort_150_split.csv",
        }
        if file_sha256(root / locations[source]) != recorded_hash:
            raise ValueError(f"Frozen source changed since manifest build: {source}")
    value["manifest_sha256"] = received
    return value


def task_frame(root: Path, task: str, roles: dict[str, dict[str, set[str]]],
               holdout: set[str]) -> dict[str, pd.DataFrame]:
    if task not in TASKS:
        raise ValueError(task)
    if task == "Tg":
        data = pd.read_parquet(root / "data/processed/tg_registry.parquet",
                               columns=["polymer_id", "canonical_psmiles", "Tg", "valid"],
                               filters=[("polymer_id", "not in", sorted(holdout))])
        data = data.loc[data.valid & data.Tg.notna(), ["polymer_id", "canonical_psmiles", "Tg"]]
        data = data.rename(columns={"Tg": "target_value"})
    else:
        data = pd.read_parquet(root / "data/processed/electronic_property_registry.parquet")
        data = data.loc[data.property_name.eq(task) & data.valid_psmiles & data.target_value.notna(),
                        ["polymer_id", "canonical_psmiles", "target_value"]]
    if data.polymer_id.duplicated().any():
        raise ValueError(f"{task}: duplicate polymer identity in targets")
    blocked = holdout | set().union(*(r["validation"] | r["test"] for r in roles.values()))
    task_roles = roles[task]
    subsets = {
        "train": data.loc[data.polymer_id.isin(task_roles["train"] - blocked)].sort_values("polymer_id"),
        "validation": data.loc[data.polymer_id.isin(task_roles["validation"] - holdout)].sort_values("polymer_id"),
        "test": data.loc[data.polymer_id.isin(task_roles["test"] - holdout)].sort_values("polymer_id"),
    }
    if not len(subsets["train"]) or not len(subsets["test"]):
        raise ValueError(f"{task}: empty leakage-safe train or test partition")
    if set(subsets["train"].polymer_id) & blocked:
        raise ValueError("PROHIBITED_OVERLAP in supervised training")
    return subsets


def md_arrays(root: Path, frames: dict[str, pd.DataFrame]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], TrainOnlyNormalizer]:
    md = md_mapping(pd.read_parquet(root / "data/processed/md_physics_full.parquet"),
                    pd.read_csv(root / "data/processed/md_psmiles_mapping.csv"))
    md = md.loc[md.md_mapping_status == "VERIFIED"].set_index("polymer_id")
    feature_names = tuple(name for name, _ in MD_FEATURES)
    arrays = {}
    for split, frame in frames.items():
        values = np.zeros((len(frame), len(MD_FEATURES)), dtype=np.float64)
        mask = np.zeros_like(values, dtype=bool)
        for j, (_, source) in enumerate(MD_FEATURES):
            selected = md.reindex(frame.polymer_id)
            valid = qc_mask(selected, source).to_numpy(bool)
            raw = pd.to_numeric(selected[source], errors="coerce").to_numpy(float)
            values[:, j] = np.where(valid, raw, 0.0)
            mask[:, j] = valid
        arrays[split] = values, mask
    scaler = TrainOnlyNormalizer(feature_names).fit(*arrays["train"], partition="train")
    return {split: scaler.transform(*value) for split, value in arrays.items()}, scaler


def load_encoder(encoder: str, policy: str, protocol: str, manifest: dict[str, object],
                 train_psmiles: list[str], root: Path) -> torch.nn.Module:
    if encoder == "morgan":
        return MorganEncoder(128)
    if encoder == "random_psmiles":
        vocabulary = SPECIAL + sorted({token for text in train_psmiles for token in tokenize_psmiles(text)} - set(SPECIAL))
        return PolymerEncoderAdapter.random({token: i for i, token in enumerate(vocabulary)},
                                            max_sequence_length=192, layers=8, hidden_size=512,
                                            attention_heads=8, intermediate_size=2048)
    if encoder == "pi1m_mlm":
        if policy == "production":
            path = root / "checkpoints/B_MLM/best.pt"
            return PolymerEncoderAdapter.pretrained(path, policy=policy)
        match = next(x for x in manifest["split_policy"]
                     if x["protocol"] == protocol and x["policy"] == "clean-transfer")
        path = root / f"phase6a_runs/clean-transfer/pretrain_{protocol}/best.pt"
        return PolymerEncoderAdapter.pretrained(path, policy=policy,
                                                 excluded_ids_hash=match["excluded_identities_sha256"])
    raise ValueError(encoder)


def run_task(root: Path, *, manifest_path: Path, task: str, protocol: str,
             policy: str, encoder: str, mode: str, fusion: str,
             auxiliary_md_loss: bool, seed: int, epochs: int, batch_size: int,
             output: Path) -> list[dict[str, object]]:
    if not (root / "phase6a_artifacts/smoke/smoke.json").is_file():
        raise ValueError("CPU smoke gate has not passed")
    manifest = verified_manifest(manifest_path)
    if manifest["leakage_audit"][protocol]["status"] != "PASS":
        raise ValueError("Leakage audit has not passed")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tg = pd.read_parquet(root / "data/processed/tg_registry.parquet",
                         columns=["polymer_id", "canonical_psmiles", "valid", "random_split", "ood_split"])
    electronic = pd.read_parquet(root / "data/processed/electronic_property_registry.parquet")
    splits = pd.read_csv(root / "data/splits/electronic_property_splits.csv")
    roles = split_sets(tg, electronic, splits, protocol)
    bulk = pd.read_csv(root / "data/processed/bulk_md_tg_final_cohort_150_split.csv", usecols=["polymer_id", "split"])
    holdout = set(bulk.loc[bulk.split == "LOCKED_HOLDOUT", "polymer_id"])
    frames = task_frame(root, task, roles, holdout)
    md, md_scaler = md_arrays(root, frames)
    train_target = frames["train"].target_value.to_numpy(float)
    target_mean, target_std = float(train_target.mean()), float(train_target.std())
    if target_std < 1e-12:
        target_std = 1.0
    structure = load_encoder(encoder, policy, protocol, manifest,
                             frames["train"].canonical_psmiles.tolist(), root)
    model = UnifiedPolymerModel(structure, modality_dropout=0.1, fusion=fusion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train = frames["train"].reset_index(drop=True)
    values, masks = md["train"]
    n_train_with_md = int(masks.any(axis=1).sum())
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            ps = train.iloc[indices].canonical_psmiles.tolist()
            md_values = torch.as_tensor(values[indices], dtype=torch.float32)
            md_mask = torch.as_tensor(masks[indices], dtype=torch.bool)
            result = model(ps, mode, md_values, md_mask)
            y = torch.as_tensor((train.iloc[indices].target_value.to_numpy(float) - target_mean) / target_std,
                                dtype=torch.float32)
            loss = torch.mean((result["predictions"][task] - y) ** 2)
            if auxiliary_md_loss and md_mask.any():
                loss = loss + 0.1 * torch.mean((result["auxiliary_md"][md_mask] - md_values[md_mask]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    digest = checkpoint_hash(model)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "task": task, "encoder": encoder,
                "feature_order": md_scaler.features, "md_mean": md_scaler.mean,
                "md_std": md_scaler.std, "target_mean": target_mean, "target_std": target_std,
                "manifest_sha256": manifest["manifest_sha256"]}, output / "checkpoint.pt")
    records = []
    for split in ("validation", "test"):
        frame = frames[split].reset_index(drop=True)
        if frame.empty:
            continue
        values, masks = md[split]
        predictions = []
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                stop = start + batch_size
                result = model(frame.iloc[start:stop].canonical_psmiles.tolist(), mode,
                               torch.as_tensor(values[start:stop], dtype=torch.float32),
                               torch.as_tensor(masks[start:stop], dtype=torch.bool))
                predictions.extend(result["predictions"][task].cpu().numpy() * target_std + target_mean)
        metric = regression_metrics(frame.target_value.to_numpy(float), np.asarray(predictions))
        record = result_record(task=task, metrics=metric, seed=seed,
                               split=f"{protocol}_{split}", checkpoint_sha256=digest,
                               manifest_sha256=manifest["manifest_sha256"],
                               result_type="CLEAN_TRANSFER_BENCHMARK" if policy == "clean-transfer" else "PRODUCTION_RESULT",
                               configuration={"encoder": encoder, "mode": mode, "fusion": fusion,
                                              "auxiliary_md_loss": auxiliary_md_loss, "policy": policy,
                                              "n_valid_train": len(train), "n_train_with_observed_md": n_train_with_md})
        records.append(record)
    (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    return records
