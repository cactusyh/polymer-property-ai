# Map-Aware MD Repair Attempts

The five original failures were rerun in an isolated repair root. Original successful pilot systems were not overwritten.

| polymer_id | strategy | attempt | construction_success | md_success | mapping_success | descriptor_success | elapsed_seconds | failure_reason |
|---|---|---:|---:|---:|---:|---:|---:|---|
| poly_14935f771a8f2f1e | original | 1 | False | False | False | False | 0.595 | RadonPyError: Reached maximum number of retrying polymerize_rw. |
| poly_14935f771a8f2f1e | robust_random_walk | 2 | True | True | True | True | 0.974 |  |
| poly_6b48dbdab6fc1fe5 | original | 1 | True | True | True | True | 0.825 |  |
| poly_6e95c85143be1ba0 | original | 1 | False | False | False | False | 1.679 | RadonPyError: Reached maximum number of retrying polymerize_rw. |
| poly_6e95c85143be1ba0 | robust_random_walk | 2 | True | True | True | True | 2.432 |  |
| poly_cdf56834602957d8 | original | 1 | False | False | False | False | 1.214 | RadonPyError: Reached maximum number of retrying polymerize_rw. |
| poly_cdf56834602957d8 | robust_random_walk | 2 | True | True | True | True | 1.753 |  |
| poly_e4ca8901a2d48a49 | original | 1 | True | True | True | True | 0.291 |  |

All five repaired systems reached successful full workflows using the tiered construction pipeline.