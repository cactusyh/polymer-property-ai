# Normalized Gyration Eigenvalue QC Audit

Finite q triplets: 7,613/7,614
Max |q1+q2+q3-1| among finite triplets: 4.441e-16

Eigenvalue ordering is descending before normalization. The different validity percentages are caused by per-component relative block-error QC: `q1` has the largest mean magnitude and therefore lower relative error, while smaller `q2/q3` components fail relative convergence more often.

This is statistically explainable, not a normalization bug. For model training, `q1` and `q2` are retained and `q3` is excluded as dependent because `q3=1-q1-q2`. Backbone-dependent targets are masked unless `backbone_confidence` is `high` or `medium`.
