#!/usr/bin/env python
from pathlib import Path
Path('reports/representation_analysis_plan.md').write_text('# Representation Analysis Plan\n\nRun after checkpoints B/C/D/E exist: extract pooled embeddings, compute UMAP, target correlations, nearest-neighbor retrieval, and CKA.\n')
print('wrote reports/representation_analysis_plan.md')
