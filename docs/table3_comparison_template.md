# Table 3: paper versus Jetson

Reference: MemFLoRA (58).pdf, Table 3, page 5 (2026-09-30 version).

Each paper row is followed by its Jetson row. Entries are peak / saved state in decimal MB; optimizer is already included in peak.

| Backbone | Method | Source | B=1 peak / saved [MB] | B=8 peak / saved [MB] | B=32 peak / saved [MB] | B=64 peak / saved [MB] | Optimizer [MB] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T-ResNet | Full FT | Paper | 9.02 / 0.90 | 13.84 / 7.09 | 35.08 / 28.33 | 63.40 / 56.65 | 4.49 |
| T-ResNet | Full FT | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| T-ResNet | LoRA-C | Paper | 5.51 / 3.10 | 11.53 / 9.12 | 32.20 / 29.79 | 59.76 / 57.35 | 0.10 |
| T-ResNet | LoRA-C | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| T-ResNet | LoRA-Edge | Paper | 2.93 / 0.60 | 7.05 / 4.72 | 21.19 / 18.86 | 40.04 / 37.71 | 0.02 |
| T-ResNet | LoRA-Edge | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| T-ResNet | MemFLoRA | Paper | 2.44 / 0.02 | 2.50 / 0.11 | 2.81 / 0.42 | 3.23 / 0.83 | 0.08 |
| T-ResNet | MemFLoRA | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| T-ResNet | MemFLoRA-SG | Paper | 2.50 / 0.02 | 2.78 / 0.11 | 3.74 / 0.42 | 5.01 / 0.83 | 0.10 |
| T-ResNet | MemFLoRA-SG | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| MobileNetV2 | Full FT | Paper | 43.20 / 16.12 | 109.00 / 81.91 | 346.93 / 319.85 | 666.64 / 639.56 | 17.97 |
| MobileNetV2 | Full FT | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| MobileNetV2 | LoRA-C | Paper | 27.17 / 17.61 | 90.00 / 80.44 | 305.40 / 295.84 | 592.61 / 583.05 | 0.30 |
| MobileNetV2 | LoRA-C | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| MobileNetV2 | LoRA-Edge | Paper | 17.09 / 7.66 | 69.77 / 60.34 | 250.39 / 240.96 | 491.22 / 481.79 | 0.16 |
| MobileNetV2 | LoRA-Edge | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| MobileNetV2 | MemFLoRA | Paper | 9.63 / 0.20 | 10.54 / 1.11 | 13.66 / 4.23 | 17.83 / 8.40 | 0.16 |
| MobileNetV2 | MemFLoRA | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |
| MobileNetV2 | MemFLoRA-SG | Paper | 9.91 / 0.20 | 11.73 / 1.11 | 17.97 / 4.23 | 26.30 / 8.40 | 0.16 |
| MobileNetV2 | MemFLoRA-SG | Jetson | PENDING | PENDING | PENDING | PENDING | INCOMPLETE |

Paper values are transcribed exactly, including the known MobileNetV2 Full FT saved-state cells (16.12 and 81.91) and SG optimizer cell (0.16). They are references, not targets or validated corrections.

CSV deltas are Jetson minus the PRINTED, rounded paper value, not differences from the original unrounded experiment. The paper uses historical reference-counting/single-buffer estimates; Jetson uses distinct concurrent backing allocations. Differences are not automatically hardware effects.

Paper rank is 2. Missing, failed, rank-mismatched or unverified LoRA-Edge cells have no numerical comparison. Partial sweeps leave the other cells PENDING. Optimizer is marked INCOMPLETE until all four batches of a row are verified.

All 40 cells are retained in paper order even for a subset run. Non-paper batch sizes are available in table3.md and measurements.csv, not this comparison.
