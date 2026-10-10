# Conjugation-Aware MD Mapping Feasibility

Audit population: **1,593** chain-gap/MD overlap polymers.

## Metadata Inventory

Only `structure.data` and `trajectory.lammpstrj` were present in the matched system directories. No mol/SDF/PDB/XYZ/JSON sidecars, RDKit molecule files, atom-index maps, or repeat-unit construction logs were found.

### File Types

| file type | count |
|---|---:|
| `.data` | 1593 |
| `.lammpstrj` | 1593 |

### Topology Metadata Signals

- Atom-type comments: `1,593` systems in the matched audit; these are force-field labels such as `c3,0` and `hc,0`, not chemical atom maps.
- Atom names: `0`.
- Residue names: `0`.
- Repeat/construction text: `0`.
- Atom-index mapping text: `0`.
- Molecule IDs are available through the LAMMPS `full` atom style, but they identify coarse molecule membership rather than repeat-unit or PSMILES atom identity.

## PSMILES Motif Registry

| motif | polymers |
|---|---:|
| `motif_aromatic` | 886 |
| `motif_conjugated_bond` | 1,397 |
| `motif_sp2` | 1,446 |
| `motif_hetero_adjacent` | 1,362 |
| `motif_conjugated_linker` | 886 |

Motifs are identified from RDKit repeat-unit PSMILES only. They are not claimed to be mapped onto trajectory atoms.

## Mapping Test

A deterministic representative subset was selected across motif groups. The attempted mapping used only topology metadata, atom-type comments, molecule IDs, and explicit text signals. No correspondence was inferred from connectivity alone.

| confidence | all matched systems | representative examples |
|---|---:|---|
| `HIGH` | 0 | none |
| `MEDIUM` | 0 | none |
| `LOW` | 1,593 | poly_f36d57fccdfc9661 *C*; poly_6e545ca8b11c4c21 *CC(*)C; poly_e55e25032750ba4e *CC(*)CC; poly_bd578e3eb80fd2c5 *CC(*)CCC; poly_ce2b8672b0f0f7d7 *CC(*)CC(C)C |
| `FAILED` | 0 | none |

Result: no HIGH or MEDIUM atom-resolved PSMILES-to-topology mappings were established. LOW indicates only coarse metadata compatibility and is not usable for primary conjugation-aware descriptors. FAILED indicates no mapping evidence beyond the identity-level registry.

## Descriptor Feasibility

                      descriptor_family                                           required_metadata  feasible_polymer_count  expected_coverage_fraction  mapping_high_count  mapping_medium_count  mapping_low_count  mapping_failed_count                                        main_failure_mode computational_cost electronic_relevance interpretability recommendation
        CONJUGATED_CENTRAL_BOND_TORSION                    conjugated bonds plus validated atom map                       0                         0.0                   0                     0               1593                     0          HIGH/MEDIUM mapping required; no validated maps           moderate                 high             high          NO-GO
              AROMATIC_RING_PLANE_TWIST two mapped aromatic rings and atom-resolved ring membership                       0                         0.0                   0                     0               1593                     0 No ring atom correspondence or ring identity in topology           moderate                 high             high          NO-GO
                  SP2_SEGMENT_PLANARITY                                    mapped sp2 atom segments                       0                         0.0                   0                     0               1593                     0   sp2 identity cannot be transferred to trajectory atoms           moderate          medium-high             high          NO-GO
CONFORMATIONAL_CONJUGATION_LENGTH_PROXY                           mapped contiguous conjugated path                       0                         0.0                   0                     0               1593                     0               path membership and atom order unavailable           moderate                 high             high          NO-GO
           AROMATIC_CONJUGATED_CONTACTS               mapped aromatic/conjugated atom sets plus PBC                       0                         0.0                   0                     0               1593                     0 contact categories cannot be assigned to chemical motifs               high          medium-high             high          NO-GO
          CONJUGATED_TORSIONAL_DYNAMICS                             mapped torsion list over frames                       0                         0.0                   0                     0               1593                     0                 no atom-resolved conjugated torsion list           moderate          medium-high             high          NO-GO

## Priority Ranking

1. **Conjugated central-bond torsion** — highest relevance and moderate cost, but NO-GO with current files because central-bond atom identity is unavailable.
2. **Aromatic ring-plane twist** — high relevance and interpretable, but NO-GO because ring atom membership is unavailable.
3. **Conformational conjugation-length proxies** — high relevance, but NO-GO because contiguous mapped paths are unavailable.
4. **sp2 segment planarity** — medium-high relevance, but NO-GO because sp2 atom labels cannot be transferred.
5. **Conjugated torsional dynamics** — high relevance but more expensive; NO-GO for the same mapping reason.
6. **Aromatic/conjugated contacts** — potentially relevant but highest ambiguity/cost; NO-GO without motif-resolved atom sets.

## Final Decision

**REGENERATE_MD_WITH_MAPPING_METADATA**

Existing files are sufficient for topology-only geometry, but not for reliable electronically relevant PSMILES-to-trajectory atom mapping. Future simulations should preserve explicit repeat-unit atom indices, PSMILES/RDKit atom indices, residue or repeat IDs, bond-order-aware construction metadata, and a machine-readable map from each trajectory atom to its repeat-unit atom.
