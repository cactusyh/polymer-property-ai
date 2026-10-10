# Periodicity Validation

Held-out repeat-number variants were compared against different-polymer negatives. This does not infer periodicity learning from MLM validation loss.

| experiment | model | positive_cosine_mean | negative_cosine_mean | contrastive_validation_loss | Recall@1 | Recall@10 | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| periodicity_validation | B_MLM | 0.9203 | 0.6710 | 4.6360 | 0.2720 | 0.6470 | 1000 |
| periodicity_validation | C_PERIODIC | 0.9391 | 0.0045 | 0.7356 | 0.4980 | 1.0000 | 1000 |

MODEL C shows repeat invariance only if positive-vs-negative separation and retrieval metrics improve over MODEL B.
