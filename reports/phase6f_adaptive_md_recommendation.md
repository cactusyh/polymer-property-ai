# Phase 6F adaptive MD recommendation

{
  "schema_version": "phase6f.adaptive_md_decision.v1",
  "decision_case": "A",
  "bulk_residual_signal": {
    "paired_delta": {
      "ci95_high": 1.051339306371593,
      "ci95_low": -0.1980060771901918,
      "delta_rmse_mean": 0.41383576937116795,
      "delta_rmse_median": 0.41835531879063126,
      "resamples": 1000,
      "seed": 13,
      "unit": "chemical_group"
    },
    "final_metrics": {
      "MAE": 128.8870949498508,
      "R2": -1.0559708354031194,
      "RMSE": 151.78484029862375,
      "n": 224
    }
  },
  "launch_executed": false,
  "rationale": "Residual signal is absent or uncertain; propose a small diagnostic pilot and improved physical descriptors before any larger batch.",
  "pilot_proposal": {
    "cohort_size": "30-50 only if weak positive signal reproduces",
    "replicas": "two independent replicas for a subset",
    "temperatures": "chosen around structure-model Tg estimates without experimental labels",
    "features": [
      "density and volume response",
      "segmental mobility",
      "torsional transition rates",
      "explicit convergence estimates"
    ]
  }
}
