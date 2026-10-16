# Phase 6Z resume determinism

`RESUME_DETERMINISM_PASS`. Four uninterrupted steps and two-step plus fresh-process resume matched all captured model, MLM head, optimizer, scaler, global RNG, sampler, row, mask, logits, and prediction hashes. The workerless path explicitly disables DataLoader worker RNG state and uses a deterministic per-epoch loader seed.
