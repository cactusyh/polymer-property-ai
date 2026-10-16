# Phase 6AD resume determinism

The corrected trainer uses gradient accumulation of 16 microbatches per optimizer update. An uninterrupted two-update run and a one-update plus fresh-process resume run matched model, MLM head, optimizer, RNG, masked-input, logits, and prediction hashes exactly for 512 samples.
