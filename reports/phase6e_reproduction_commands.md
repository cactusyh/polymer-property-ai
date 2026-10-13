# Phase 6E reproduction

```bash
conda run -n md_tg_ml python scripts/phase6e_pipeline.py --all
conda run -n md_tg_ml python -m pytest tests/test_phase6e_pipeline.py
```

The mace-lammps environment is not modified. Raw trajectories and checkpoints remain outside Git.
