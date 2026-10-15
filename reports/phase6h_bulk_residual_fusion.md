# Phase 6H bulk residual fusion

{
  "schema_version": "phase6h.bulk_residual_fusion.v1",
  "source_hashes": {
    "snapshot": "004930fb6f3eef9d402176b44aba3f3e21a7e7f21ee3c649269cf83e2e350e11",
    "folds": "5d5ff6f4645a9492fb2eb649a7a40f939d24d273ecc9c7a7a35bce83c23d2550"
  },
  "implementation": "Phase 6H recomputed fold-local predictions with corrected Ridge; no prior predictions are silently reused.",
  "bulk_n": 224,
  "metrics": {
    "structure": {
      "RMSE": 103.91490907399634,
      "MAE": 80.11566935392268,
      "R2": 0.03635765100369237,
      "Spearman": 0.46032778133675,
      "N": 224,
      "coverage": 224
    },
    "direct": {
      "RMSE": 91.40833428997901,
      "MAE": 69.92411973332466,
      "R2": 0.2543556289263166,
      "Spearman": 0.5424727738629085,
      "N": 224,
      "coverage": 224,
      "paired_delta": {
        "delta_rmse_mean": -12.495368014661123,
        "delta_rmse_median": -12.482926608146897,
        "ci95_low": -16.31123490696615,
        "ci95_high": -8.961213922039544,
        "resamples": 1000,
        "unit": "chemical_group",
        "seed": 13
      }
    },
    "residual": {
      "RMSE": 103.80152391531158,
      "MAE": 79.91228363242685,
      "R2": 0.03845943084352288,
      "Spearman": 0.4659918855434551,
      "N": 224,
      "coverage": 224,
      "paired_delta": {
        "delta_rmse_mean": -0.12088590119794088,
        "delta_rmse_median": -0.12214634290337756,
        "ci95_low": -0.4997721825069167,
        "ci95_high": 0.23951495086536168,
        "resamples": 1000,
        "unit": "chemical_group",
        "seed": 13
      }
    }
  },
  "bulk_md_launch": false
}
