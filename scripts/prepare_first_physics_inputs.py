#!/usr/bin/env python
"""Export read-only CSV views for the first GPU experiment.

The source parquet files, split assignments, tokenizer, and MD preprocessing are
not changed. This adapter exists because the H800 environment used for training
does not include pyarrow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/first_physics_gpu_inputs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pi = pd.read_parquet(ROOT / "data/processed/pi1m_no_tg.parquet")
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet")
    md = pd.read_parquet(ROOT / "data/processed/md_physics_full.parquet")
    manifest = pd.read_csv(ROOT / "data/splits/md_pretrain_ood_seed13.csv")

    pi.loc[pi.valid & pi.split.eq("train"), ["raw_psmiles", "valid", "split"]].to_csv(
        out / "pi1m_valid_train.csv", index=False
    )
    pi.loc[pi.valid & pi.split.eq("validation"), ["raw_psmiles", "valid", "split"]].to_csv(
        out / "pi1m_valid_validation.csv", index=False
    )

    tg_cols = ["tg_id", "canonical_psmiles", "Tg", "valid", "ood_split"]
    tg.loc[tg.valid & tg.canonical_psmiles.notna(), tg_cols].to_csv(
        out / "tg_valid.csv", index=False
    )

    targets = ["log_Rg", "q1", "q2", "relative_shape_anisotropy"]
    md_cols = ["sample_id", "canonical_psmiles"]
    for target in targets:
        md_cols += [target, f"{target}_quality_class"]
    md_pool = md[md.sample_id.isin(set(manifest.sample_id))].copy()
    md_pool[md_cols].to_csv(out / "md_d0_ood_pool.csv", index=False)

    valtest = set(tg.loc[tg.ood_split.isin(["validation", "test"]), "canonical_psmiles"].dropna())
    intersection = sorted(set(md_pool.canonical_psmiles.dropna()) & valtest)
    if intersection:
        raise RuntimeError(f"strict-inductive MD/Tg OOD intersection is non-empty: {intersection[:5]}")

    pd.DataFrame(
        [
            {
                "pi1m_no_tg_rows": len(pi),
                "pi1m_valid_rows": int(pi.valid.sum()),
                "pi1m_valid_train_rows": int((pi.valid & pi.split.eq("train")).sum()),
                "pi1m_valid_validation_rows": int((pi.valid & pi.split.eq("validation")).sum()),
                "pi1m_invalid_rows": int((~pi.valid).sum()),
                "md_ood_pool_rows": len(md_pool),
                "md_intersection_tg_validation": int(md_pool.canonical_psmiles.isin(set(tg.loc[tg.ood_split.eq("validation"), "canonical_psmiles"].dropna())).sum()),
                "md_intersection_tg_test": int(md_pool.canonical_psmiles.isin(set(tg.loc[tg.ood_split.eq("test"), "canonical_psmiles"].dropna())).sum()),
            }
        ]
    ).to_csv(out / "input_audit.csv", index=False)
    print(pd.read_csv(out / "input_audit.csv").to_string(index=False))


if __name__ == "__main__":
    main()
