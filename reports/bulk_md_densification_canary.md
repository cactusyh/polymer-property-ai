# Bulk MD Densification Canary

Classification: **BULK_DENSIFICATION_CANARY = PARTIAL**.

## Scope

Only the pre-existing `bulk_md_canary_01` was used.  No RadonPy structure was regenerated, no 12-system pilot was started, and no lower-temperature trajectory was run.

## Protocol and outcomes

| Stage | Ensemble / target | Duration | Density outcome | Status |
|---|---|---:|---:|---|
| A | energy minimization | 1,000 iterations | 0.1084 g/cm3 | complete |
| B | 600 K NVT relaxation | 20 ps | 0.1084 g/cm3 | complete |
| C | 600 K, 1 atm isotropic NPT densification | 50 ps | 0.1084 -> 0.5211 g/cm3 | complete |
| D | 600 K, 1 atm isotropic NPT short production | 20 ps | 0.5211 -> 0.6860 g/cm3 | complete, not plateaued |

The initial box was 87.89 A per side.  At the final production frame it was 47.52 A per side.  The final ten sampled production records have density 0.6754 +/- 0.0088 g/cm3 and temperature 600.4 +/- 5.2 K.  Potential energy remained finite (final-ten mean 10,752.6 kcal/mol).

## Stability and artifacts

- GPU runtime: NVIDIA H800, Kokkos 4.7.1 CUDA Serial, single GPU/single rank.
- Performance: 330.0 steps/s in densification and 294.0 steps/s in production.
- No NaN/Inf, lost atoms, PPPM out-of-range, CUDA error, KOKKOS error, or LAMMPS fatal error occurred.
- All stages reported zero dangerous neighbor builds.
- The production trajectory has 21 complete frames (71,000 through 91,000 steps) and every frame has 6,444 atoms.
- `bulk_600K.restart` is available.

## Decision

The exported data and GPU bulk workflow are numerically stable, and densification is smooth.  However, density increased materially during the full short production segment rather than fluctuating around a plateau.  The canary is therefore **PARTIAL**, not PASS.  A future continuation must resume from `bulk_600K.restart` and demonstrate a density/box plateau at 600 K before any lower-temperature sequence or 12-polymer expansion.
