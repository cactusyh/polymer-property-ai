# Map-Aware Single-Chain MD Pilot

Date: 2026-09-14  
Label: `METHOD_DEVELOPMENT_MD_PILOT`

## Result
The pipeline preserved explicit RDKit repeat-unit to chain to LAMMPS to trajectory atom IDs for 15 of 20 selected systems.

## Findings
- Structures generated: 15/20.
- Force-field successes: 15/20.
- MD successes: 15/20.
- Valid atom mappings: 15/20.
- Descriptor successes: 15/20 attempted systems.
- Main failure mode: RadonPyError: Reached maximum number of retrying polymerize_rw..

Mapped conjugated torsions, aromatic ring-plane twist, heuristic hard/soft conjugation-length proxies, and minimum-image aromatic/conjugated contact descriptors are computable on successful systems. These are exploratory descriptors; the soft/hard lengths are not quantum conjugation lengths.

The pilot is not ready to scale to 200: mapping success is 75%, below the 80% threshold. RadonPy random-walk chain construction failed for five selected chemistries. Dynamic torsion descriptors and full sp2 segment planarity remain incomplete in this technical extractor.

No old raw trajectories were modified, no existing descriptors were overwritten, no property models were trained, and no 200-system production run was launched.

## Final classification
**PARTIAL_READY_NEEDS_FIXES**
