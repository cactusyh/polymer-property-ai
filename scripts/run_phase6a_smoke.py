#!/usr/bin/env python
"""Run deterministic CPU software checks; fixture values are not scientific data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase6a.benchmark import matrix, smoke_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("phase6a_artifacts/manifest.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("phase6a_artifacts/smoke"))
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    manifest = json.loads(manifest_path.read_text())
    result = smoke_train(ROOT / args.output, manifest["manifest_sha256"])
    (ROOT / args.output / "benchmark_matrix.json").write_text(json.dumps(matrix(), indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
