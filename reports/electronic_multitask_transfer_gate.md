# Electronic Multitask Transfer Gate

## Decision: PROPERTY_DEPENDENT_TRANSFER

- Grouped-CV B>A positive properties: `5/5`; conservative reproducible B>A properties: `2/5`.
- Grouped-CV C>B positive properties: `2/5`; conservative reproducible C>B properties: `0/5`.
- Periodicity classification: **PROPERTY_DEPENDENT**.
- Current foundation encoder recommendation: **MODEL B (PI1M MLM)** unless the grouped-CV periodicity threshold is met.

## Recommendation

- **OPTION 1:** train `PI1M_CLEAN_ALL_DOWNSTREAM -> B_CLEAN` as the main foundation encoder.
- Train `C_CLEAN` only as an explicitly secondary comparison if periodicity is reproducibly positive across at least three grouped tasks; otherwise do not invest further in periodicity optimization.
- Recommended bulk MD priority: **MEDIUM PRIORITY**. The cross-property language-transfer gate does not by itself prove a bulk-specific information gap; condensed-phase physics remains a later targeted pilot.

## Strict-inductive status

- `ALL_DOWNSTREAM_POLYMERS`: `9,283` unique canonical polymers.
- Overlap with valid PI1M canonical polymers: `335`.
- Valid unique PI1M canonical polymers: `988,775`; planned removal: `0.0339%`.
- No PI1M retraining was run.

## Stopping point

EA, IE, and atomization end-to-end repeated grouped/random CV, statistics, OOF predictions, cross-property integration, and overlap planning are complete. No PI1M retraining, MD physics model, or bulk MD was started.
