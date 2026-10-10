# Map-Aware Single-Chain MD Generation Protocol

Date: 2026-09-14
Status: `METHOD_DEVELOPMENT_MD_PILOT`

## Evidence from old data

Inspection of old `structure.data` and `trajectory.lammpstrj` files shows:

- RadonPy-generated LAMMPS data files.
- GAFF-like atom-type comments, including `c3`, `ca`, `c`, `hc`, `ha`, `o`,
  and related labels.
- `atom_style full`-compatible topology with atoms, bonds, angles, dihedrals,
  and impropers.
- `units real` convention inferred from the RadonPy/Amber-style coefficients.
- Trajectories contain `ITEM: ATOMS id type xu yu zu`.
- Atom IDs are present and stable in the observed trajectories.
- Coordinates are unwrapped (`xu`, `yu`, `zu`) rather than wrapped `x`, `y`,
  `z` coordinates.
- The old files do not contain repeat-unit or PSMILES atom provenance.

The old raw inventory and feasibility reports did not expose a complete,
machine-readable equilibration log. Therefore, exact old run lengths and
temperature-ramp details cannot be established from the preserved artifacts.

## Pilot protocol

The pilot matches the recoverable old design where possible:

1. Parse the PSMILES repeat unit with RDKit and preserve source atom indices.
2. Build a single chain with RadonPy `polymerize_rw`.
3. Assign GAFF2-modified force-field terms with RadonPy `GAFF2_mod`.
4. Assign Gasteiger charges; systems that require a zero-charge fallback are
   flagged rather than silently treated as equivalent.
5. Write a LAMMPS data file with a padded orthorhombic box.
6. Run conjugate-gradient minimization.
7. Initialize at 300 K.
8. Run short 300 K NVT equilibration and production stages for technical
   mapping/descriptor validation.
9. Dump `id type xu yu zu`, sorted by atom ID.

Default pilot settings in `scripts/generate_map_aware_single_chain_md.py`:

| Setting | Value |
|---|---:|
| target atoms | 300 |
| timestep | 1 fs |
| temperature | 300 K |
| ensemble | NVT |
| equilibration | 1,000 steps |
| production | 5,000 steps |
| dump interval | 500 steps |
| coordinate convention | unwrapped `xu yu zu` |

This is intentionally a short technical pilot, not a production-quality
sampling campaign.

## Mapping protocol

Each generated atom carries source provenance before LAMMPS conversion:

`RDKit repeat-unit atom index -> repeat index -> chain index -> LAMMPS atom ID -> trajectory atom ID`

The pilot writes sidecar metadata under the new output root and validates
continuous atom IDs, atom-count consistency, finite charges, and trajectory
ID stability. A failed assertion marks the system failed.

