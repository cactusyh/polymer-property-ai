# Fixed 150-system build failure audit and rescue

## Scope and frozen cohort

- Original identities: 150 unique polymers; DEVELOPMENT 120, LOCKED_HOLDOUT 30; unchanged.
- Structural comparison uses only the fixed cohort manifest and build status. No Tg values were read.
- No 500/400/300 K MD or Tg-model training was started by this rescue workflow.

## Finalized original 600 K queue

- All 97/97 original build-PASS systems have a terminal 600 K registry row.
- 600K_COMPLETE: 94/97; 600K_PLATEAU_NOT_REACHED: 3/97.
- DEVELOPMENT: 78/81 complete; LOCKED_HOLDOUT: 16/16 complete.
- 600 K failure table: `results/bulk_md_tg_150_600K_failure_table.csv` (3 rows).

## Original build failures

- Original build coverage: 97/150; unresolved: 53/150 (27 time-limited builds, 26 observed non-timeout errors).
- BUILD_TIMEOUT_30MIN: 27; GAFF2_mod assignment: 22; RadonPy recursion: 2; polymerize_rw retry exhaustion: 2.
- Original non-timeout runtimes were not recorded. Timeout runtimes are known only to have reached the 1800 s alarm.
- Per-polymer original error, traceback, chemistry, diagnostic fields and rescue outcome: `results/bulk_md_tg_150_build_failure_audit.csv`.

## GAFF2_mod assignment diagnosis

- Diagnostic replays attempted: 22/22 (20 assignment failures reproduced; 2 diagnostic timeouts).
- The frozen GAFF2_mod particle database contains no Si, Se, Ge, or Li atom types. The azide [N-][N+]#N motif also leaves a two-coordinate SP2 N without an assignment rule. These cases are classified `GAFF2_UNSUPPORTED_CHEMISTRY`; no parameter or atom type was invented.
- Two full-chain diagnostic replays reached their 2700 s diagnostic limit before force-field assignment; their original GAFF2_mod failure and absent Si/Ge particle types remain documented. This diagnostic timeout is not a build rescue result.
- No implementation/mapping bug, parameter-lookup bug, or existing omitted GAFF2_mod parameter was demonstrated. Bond, angle, dihedral and improper assignment cannot be evaluated after atom typing fails; those fields are left empty rather than guessed.
- GAFF2_mod is invoked without charge assignment; Gasteiger charges are assigned only after its successful return. These original failures therefore are not charge-assignment failures.
- Atom typing, bond, angle, dihedral and improper diagnostics, shared signatures, and logs are linked per polymer in the audit CSV.

## Structural build-failure bias

- PASS versus unresolved builds after rescue; FAIL includes time-limited builds and does not imply unsupported chemistry.
| Structural feature | PASS median | FAIL median | standardized mean difference (FAIL minus PASS) |
|---|---:|---:|---:|
| repeat_unit_atoms | 40.000 | 35.000 | -0.039 |
| aromatic_fraction | 0.375 | 0.000 | -0.598 |
| heteroatom_fraction | 0.326 | 0.400 | +0.264 |
| ring_count | 2.000 | 1.000 | -0.295 |
| selected_chain_count | 5.000 | 5.000 | +0.042 |
| selected_DP | 46.000 | 50.000 | +0.117 |
| estimated_total_atoms | 9000.000 | 9000.000 | +0.017 |

| Major chemical motif | PASS n/107 | FAIL n/43 | failure minus pass prevalence |
|---|---:|---:|---:|
| amide | 20/107 | 7/43 | -0.024 |
| ester | 22/107 | 7/43 | -0.043 |
| imide | 4/107 | 5/43 | +0.079 |
| aromatic_n | 20/107 | 2/43 | -0.140 |
| sulfur | 20/107 | 3/43 | -0.117 |
| fluorine | 14/107 | 5/43 | -0.015 |
| chlorine | 6/107 | 3/43 | +0.014 |
| bromine | 1/107 | 3/43 | +0.060 |
| nitro | 4/107 | 1/43 | -0.014 |
| nitrile | 8/107 | 1/43 | -0.052 |
| ether | 50/107 | 12/43 | -0.188 |
| alkene | 23/107 | 11/43 | +0.041 |
| hetero_ring | 45/107 | 16/43 | -0.048 |
| silicon | 0/107 | 18/43 | +0.419 |
| selenium | 0/107 | 2/43 | +0.047 |
| germanium | 0/107 | 1/43 | +0.023 |
| lithium | 0/107 | 1/43 | +0.023 |
| azide | 0/107 | 1/43 | +0.023 |

| Family | cohort members | current failures | failure rate |
|---|---:|---:|---:|
| silicon | 18 | 18 | 100.0% |
| selenium | 2 | 2 | 100.0% |
| germanium | 1 | 1 | 100.0% |
| lithium | 1 | 1 | 100.0% |
| imide | 9 | 5 | 55.6% |
| aromatic_n | 22 | 2 | 9.1% |

| Split | PASS | FAIL | failure rate |
|---|---:|---:|---:|
| DEVELOPMENT | 90/120 | 30/120 | 25.0% |
| LOCKED_HOLDOUT | 17/30 | 13/30 | 43.3% |

- The 150 Morgan chemical cluster IDs are each represented once in this cohort. Within-cluster failure rates therefore cannot be estimated; cluster ID remains in each audit row.
- Descriptive bias classification: **BUILD_FAILURE_BIAS_HIGH**. Thresholds: HIGH if any absolute standardized difference ≥0.8, motif prevalence gap ≥0.30, or split failure-rate gap ≥0.25; MODERATE if ≥0.5, ≥0.20, or ≥0.15; otherwise LOW.

## Standardized rescue tiers

1. BUILD_TIMEOUT_30MIN: replay the exact original build function and seed with only the per-polymer alarm changed from 1800 to 7200 s.
2. polymerize_rw exhaustion: retain monomer, DP, chain count, GAFF2_mod and export QC; use the validated single-chain robust, relaxed-overlap and extended-initialization random-walk settings at the original DP, with a uniform 1200 s cap per remaining tier. Never accept a shorter diagnostic chain. For export recursion, use an iterative equivalent of RadonPy molecule-ID traversal.
3. GAFF2_mod: inspect atom typing and bond/angle/dihedral/improper assignment against the frozen parameter database. Repair only demonstrated implementation or lookup bugs; do not invent or hand-fit parameters. Genuinely unsupported chemistry is GAFF2_UNSUPPORTED_CHEMISTRY.

- First construction-tier attempts recorded: 4/4; remaining full-DP walk tiers recorded: 2/2; iterative export-recursion repairs recorded: 2/2.

## 120-minute timeout retry

- Original 30-minute timeout systems: 27; attempted: 27/27.
- Rescued: 9; still timed out at 120 minutes: 18.
- Median successful retry build time: 932.0 s; maximum: 2380.0 s.
- Detailed retry records: `results/bulk_md_tg_150_timeout_retry.csv`.

- Coverage from timeout retries alone: 106/150. The final total below also includes the separately recorded export-recursion rescue.
- Timeout-only coverage by split: DEVELOPMENT: 89/120; LOCKED_HOLDOUT: 17/30.
- Timeouts remain unresolved construction outcomes, never unsupported-chemistry diagnoses.

## Build coverage after rescue

- Total: 107/150; DEVELOPMENT: 90/120; LOCKED_HOLDOUT: 17/30.
- Coverage gate: **BUILD_COVERAGE_INSUFFICIENT**.

| Final rescue/audit class | systems |
|---|---:|
| BUILD_PASS_AFTER_EXPORT_REPAIR | 1 |
| BUILD_PASS_AFTER_TIMEOUT_RETRY | 9 |
| BUILD_TIMEOUT_120MIN | 19 |
| GAFF2_UNSUPPORTED_CHEMISTRY | 22 |
| POLYMERIZE_RW_TIERS_EXHAUSTED | 2 |

## Stop condition

- Rescue records are reconciled; successful rescued structure.data files passed atom-count, molecule-ID, finite-coordinate and topology-reference QC.
- The final BUILD_TIMEOUT_120MIN count includes the 18 original timeout retries and one separate export-recursion rescue that reached its wall-time limit.
- Both POLYMERIZE_RW_TIERS_EXHAUSTED outcomes exhausted the allowed time in each fallback tier; they do not establish unsupported chemistry.
- Stopped after build rescue and coverage audit. No MD was started for rescued systems; no lower-temperature MD or Tg-model training was launched during finalization.
