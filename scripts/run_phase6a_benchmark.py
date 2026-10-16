#!/usr/bin/env python
"""Explicitly gated single-configuration scientific benchmark launcher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase6a.training import run_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", choices=["random", "ood"], required=True)
    parser.add_argument("--policy", choices=["clean-transfer", "production"], required=True)
    parser.add_argument("--encoder", choices=["morgan", "random_psmiles", "pi1m_mlm"], required=True)
    parser.add_argument("--mode", choices=["structure_only", "chain_md_observed", "chain_md_predicted"], required=True)
    parser.add_argument("--fusion", choices=["gated", "concat"], default="gated")
    parser.add_argument("--auxiliary-md-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--enable-full-training", action="store_true")
    args = parser.parse_args()
    if not args.enable_full_training:
        parser.error("Full training is gated; first pass the manifest, leakage audit, and CPU smoke checks")
    output = (ROOT / "phase6a_runs" / args.policy / args.split / args.task / args.encoder /
              args.mode / args.fusion / f"aux_{int(args.auxiliary_md_loss)}" / f"seed_{args.seed}")
    results = run_task(ROOT, manifest_path=ROOT / "phase6a_artifacts/manifest.v1.json",
                       task=args.task, protocol=args.split, policy=args.policy,
                       encoder=args.encoder, mode=args.mode, fusion=args.fusion,
                       auxiliary_md_loss=args.auxiliary_md_loss, seed=args.seed,
                       epochs=args.epochs, batch_size=args.batch_size, output=output)
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
