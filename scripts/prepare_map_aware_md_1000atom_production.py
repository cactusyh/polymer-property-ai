#!/usr/bin/env python3
"""Apply the validated 100 ps production protocol to prepared long-chain inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/map_aware_md_1000atom_chain_manifest.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/inspire/ssd/project/sais-suiren-foundation-model-prediction/public/yh/radonpy/"
            "map_aware_single_chain_md_1000atom_200"
        ),
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 200 or manifest.polymer_id.nunique() != 200:
        raise SystemExit("expected exactly 200 unique manifest systems")

    updated = []
    for polymer_id in manifest.polymer_id.astype(str):
        system = args.output_root / "systems" / polymer_id
        input_path = system / "lammps" / "in.map_aware"
        metadata_path = system / "metadata" / "system_metadata.json"
        if not input_path.exists() or not metadata_path.exists():
            raise SystemExit(f"prepared system input missing: {polymer_id}")
        text = input_path.read_text(encoding="utf-8")
        if "run 5000" not in text:
            raise SystemExit(f"unexpected production input for {polymer_id}")
        input_path.write_text(text.replace("run 5000", "run 100000"), encoding="utf-8")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "production_steps": 100000,
                "production_time_ps": 100.0,
                "dump_every_steps": 500,
                "expected_frame_count": 201,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        updated.append(polymer_id)

    print(json.dumps({"prepared_systems_updated": len(updated), "production_steps": 100000, "dump_every_steps": 500}, indent=2))


if __name__ == "__main__":
    main()
