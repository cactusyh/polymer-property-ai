# Tg Benchmark and Strict-Inductive Manifest Report

Tg rows: 7,174
Valid Tg rows: 7,170
PI1M rows after removing all Tg canonical polymers: 995,612

## Split Counts
- `random`: {'train': 5766, 'validation': 706, 'test': 702}
- `ood`: {'train': 5568, 'test': 889, 'validation': 717}

## Strict-Inductive MD Checks
- `random` seed 13: MD pool rows=6,330, intersection with Tg validation/test=0
- `ood` seed 13: MD pool rows=6,171, intersection with Tg validation/test=0
