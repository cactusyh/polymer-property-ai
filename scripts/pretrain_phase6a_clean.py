#!/usr/bin/env python
"""Explicit full-run launcher for a clean PI1M MLM checkpoint and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase6a.contract import canonical_json, split_sets
from src.phase6a.training import verified_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "ood"], required=True)
    parser.add_argument("--enable-full-training", action="store_true")
    args = parser.parse_args()
    if not args.enable_full_training:
        parser.error("Full MLM training requires the manifest and smoke gates")
    if not (ROOT / "phase6a_artifacts/smoke/smoke.json").is_file():
        raise ValueError("CPU smoke gate has not passed")
    manifest = verified_manifest(ROOT / "phase6a_artifacts/manifest.v1.json")
    policy = next(x for x in manifest["split_policy"]
                  if x["protocol"] == args.split and x["policy"] == "clean-transfer")
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet",
                         columns=["polymer_id", "canonical_psmiles", "valid", "random_split", "ood_split"])
    electronic = pd.read_parquet(ROOT / "data/processed/electronic_property_registry.parquet",
                                  columns=["property_name", "sample_id", "canonical_psmiles", "polymer_id",
                                           "valid_psmiles", "target_value"])
    splits = pd.read_csv(ROOT / "data/splits/electronic_property_splits.csv")
    roles = split_sets(tg, electronic, splits, args.split)
    bulk = pd.read_csv(ROOT / "data/processed/bulk_md_tg_final_cohort_150_split.csv", usecols=["polymer_id", "split"])
    excluded = set(bulk.loc[bulk.split == "LOCKED_HOLDOUT", "polymer_id"])
    excluded |= set().union(*(r["validation"] | r["test"] for r in roles.values()))
    if hashlib.sha256(canonical_json(sorted(excluded))).hexdigest() != policy["excluded_identities_sha256"]:
        raise ValueError("Clean-transfer exclusion identities changed")
    corpus = ROOT / f"phase6a_artifacts/pi1m_clean-transfer_{args.split}.parquet"
    ids = set(pd.read_parquet(corpus, columns=["polymer_id"]).polymer_id)
    if ids & excluded:
        raise ValueError("PROHIBITED_OVERLAP in clean pretraining corpus")
    output = ROOT / f"phase6a_runs/clean-transfer/pretrain_{args.split}"
    config = yaml.safe_load((ROOT / "configs/pretrain_pi1m.yaml").read_text())
    config["train_path"] = str(corpus)
    config["output_dir"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "pretrain_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    subprocess.run([sys.executable, str(ROOT / "scripts/pretrain_pi1m.py"),
                    "--config", str(config_path)], cwd=ROOT, check=True)
    checkpoint = output / "best.pt"
    sidecar = {"policy": "clean-transfer", "protocol": args.split,
               "manifest_sha256": manifest["manifest_sha256"],
               "excluded_identities_sha256": policy["excluded_identities_sha256"],
               "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
               "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    checkpoint.with_suffix(".provenance.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": sidecar["checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
