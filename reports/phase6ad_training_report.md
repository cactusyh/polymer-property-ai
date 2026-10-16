# Phase 6AD training correction

The previous implementation stepped the optimizer on every microbatch despite `gradient_accumulation_steps: 16`. The correction scales each microbatch loss, steps only at accumulation boundaries, handles final partial accumulation, and records optimizer updates separately from microbatch and row counters.
