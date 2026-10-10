#!/usr/bin/env bash
set -euo pipefail

FINAL_ROOT="$1"
RESULTS="$FINAL_ROOT/logs/gpu_queue_results.csv"
WRAPPER="$(cd "$(dirname "$0")" && pwd)/run_lammps_kokkos_gpu.sh"
CANARY_IDS=(
  poly_0382ff574025b3c6
  poly_dddca9e52ab732dc
  poly_087c04b9d9cdf6d7
  poly_d2ce25761366483b
  poly_5b2e2fc8f3913f8b
)

mkdir -p "$FINAL_ROOT/logs/gpu_workers"
if [[ ! -f "$RESULTS" ]]; then
  printf 'polymer_id,returncode,elapsed_seconds\n' > "$RESULTS"
fi

is_canary() {
  local polymer_id="$1"
  for canary_id in "${CANARY_IDS[@]}"; do
    [[ "$polymer_id" == "$canary_id" ]] && return 0
  done
  return 1
}

was_successful() {
  local polymer_id="$1"
  awk -F, -v id="$polymer_id" '$1 == id && $2 == 0 { found = 1 } END { exit !found }' "$RESULTS"
}

for system_dir in "$FINAL_ROOT"/systems/poly_*; do
  [[ -d "$system_dir" ]] || continue
  polymer_id="${system_dir##*/}"
  is_canary "$polymer_id" && continue
  was_successful "$polymer_id" && continue

  log_path="$FINAL_ROOT/logs/gpu_workers/${polymer_id}.log"
  started=$(date +%s)
  printf 'START %s %s\n' "$polymer_id" "$(date -Is)" > "$log_path"
  set +e
  (
    cd "$system_dir/lammps" &&
    CUDA_VISIBLE_DEVICES=0 "$WRAPPER" in.map_aware
  ) >> "$log_path" 2>&1
  returncode=$?
  set -e
  finished=$(date +%s)
  elapsed=$((finished - started))
  printf 'END %s returncode=%s elapsed_seconds=%s %s\n' "$polymer_id" "$returncode" "$elapsed" "$(date -Is)" >> "$log_path"
  printf '%s,%s,%s\n' "$polymer_id" "$returncode" "$elapsed" >> "$RESULTS"
  printf '{"polymer_id":"%s","returncode":%s,"elapsed_seconds":%s}\n' "$polymer_id" "$returncode" "$elapsed"
done

python - "$FINAL_ROOT" "$RESULTS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = Path(sys.argv[2])
rows = []
for line in results.read_text().splitlines()[1:]:
    polymer_id, returncode, elapsed = line.split(",")
    rows.append((polymer_id, int(returncode), float(elapsed)))

elapsed_sum = sum(item[2] for item in rows)
summary = {
    "concurrency": 1,
    "systems_attempted": len(rows),
    "systems_md_success": sum(item[1] == 0 for item in rows),
    "systems_failed": sum(item[1] != 0 for item in rows),
    "mean_worker_seconds": elapsed_sum / len(rows) if rows else None,
}
(root / "logs/gpu_queue_concurrency_1_final_summary.json").write_text(json.dumps(summary, indent=2))
PY
