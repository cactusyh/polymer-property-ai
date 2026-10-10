#!/usr/bin/env python
"""Export immutable diagnostic views for the CUDA environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/physics_diagnostic_inputs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pi = pd.read_parquet(ROOT / "data/processed/pi1m_no_tg.parquet")
    tg = pd.read_parquet(ROOT / "data/processed/tg_registry.parquet")
    md = pd.read_parquet(ROOT / "data/processed/md_physics_full.parquet")

    pi.loc[pi.valid & pi.split.eq("train"), ["raw_psmiles", "valid", "split"]].to_csv(out / "pi_train.csv", index=False)
    pi.loc[pi.valid & pi.split.eq("validation"), ["raw_psmiles", "valid", "split"]].to_csv(out / "pi_validation.csv", index=False)
    tg.loc[tg.valid & tg.canonical_psmiles.notna(), ["tg_id", "canonical_psmiles", "Tg", "valid", "ood_split"]].to_csv(out / "tg.csv", index=False)
    md.to_csv(out / "md.csv", index=False)

    audit = {
        "pi1m_no_tg_rows": len(pi),
        "pi1m_invalid_rows": int((~pi.valid).sum()),
        "pi_valid_rows": int(pi.valid.sum()),
        "pi_valid_train_rows": int((pi.valid & pi.split.eq("train")).sum()),
        "pi_valid_validation_rows": int((pi.valid & pi.split.eq("validation")).sum()),
        "tg_valid_rows": int(tg.valid.sum()),
        "md_rows": len(md),
    }
    pd.DataFrame([audit]).to_csv(out / "audit.csv", index=False)
    print(pd.DataFrame([audit]).to_string(index=False))


if __name__ == "__main__":
    main()
