#!/usr/bin/env python3
"""Build and run one map-aware production system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from generate_map_aware_single_chain_md import build_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--production-steps", type=int, default=100000)
    parser.add_argument("--dump-every", type=int, default=500)
    parser.add_argument("--equil-steps", type=int, default=1000)
    parser.add_argument("--target-atoms", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--box-padding", type=float, default=15.0)
    parser.add_argument("--allow-short-chain-diagnostic", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--lammps", default="/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/miniforge3/envs/radonpy/bin/lmp")
    parser.add_argument("--lammps-wrapper", default=None)
    args = parser.parse_args()
    row = pd.read_csv(args.row_csv).iloc[0].to_dict()
    status = build_system(row, args)
    output = args.output_root / "logs" / "single_system_status" / f"{row['polymer_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
