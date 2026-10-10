# Final multitemperature production QC

{
  "cohort": 150,
  "600K_reused_valid": 49,
  "600K_newly_completed": 99,
  "600K_total_complete": 148,
  "600K_DEVELOPMENT": 119,
  "600K_LOCKED_HOLDOUT": 29,
  "600K_terminal": 150,
  "600K_gate": "GO",
  "all_four_complete": 148,
  "DEVELOPMENT_complete": 119,
  "LOCKED_HOLDOUT_complete": 29,
  "failure_categories": {
    "600K_PLATEAU_NOT_REACHED": 2,
    "NOT_RUN_UPSTREAM": 6
  },
  "systematic_numerical_flags": {
    "600": [],
    "500": [],
    "400": [],
    "300": []
  },
  "new_GPU_hours": 198.8356172889471,
  "reused_historical_GPU_hours": 25.81242525047726,
  "final_status": "FINAL_MULTITEMP_PRODUCTION_PASS",
  "updated_unix": 1791562128.7076037,
  "by_temperature": {
    "600": {
      "complete": 148,
      "plateau_density_range": [
        0.252122611,
        1.61827731
      ],
      "equilibration_ps_distribution": {
        "300": 64,
        "500": 5,
        "200": 53,
        "400": 26
      }
    },
    "500": {
      "complete": 148,
      "plateau_density_range": [
        0.2997141628,
        1.7087753190000001
      ],
      "equilibration_ps_distribution": {
        "100": 138,
        "200": 10
      }
    },
    "400": {
      "complete": 148,
      "plateau_density_range": [
        0.323651134,
        1.738690737
      ],
      "equilibration_ps_distribution": {
        "100": 147,
        "200": 1
      }
    },
    "300": {
      "complete": 148,
      "plateau_density_range": [
        0.3155890637,
        1.805297719
      ],
      "equilibration_ps_distribution": {
        "100": 148
      }
    }
  }
}

No Tg values or prediction results are read. Descriptor schema and cohort/split are checksum-frozen. No descriptors are extracted and no models are trained.
