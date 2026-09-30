# Reproduce Table 3 on Jetson

After transferring/committing and pulling the updated code, run from your Jetson checkout:

```bash
cd ~/Desktop/MemFLoRA
~/venvs/memflora/bin/python tools/jetson_profile.py
```

The profiler first checks CUDA/PyTorch and shared imports in an isolated process.
An environment failure stops before any cases or output directory are created.
It never installs packages or silently switches Python environments.

One serial sweep runs 40 isolated cases: T-ResNet and MobileNetV2, five methods,
and batches 1, 8, 32, 64, rank 2. Each performs two warm-up updates and one measured
complete update. No accuracy evaluation, dataset download, pretraining, or reverse
sweep is run. Existing output directories are never overwritten.

## Compare one cell at a time

Each new timestamped `runs/jetson_table3_*/` contains:

- `table3.md`, `table3.csv`, `table3.tex`: new measurements in Table 3's layout.
- **`table3_comparison.md`**: the same grid with paired **Paper** and **Jetson** rows
  for every method, B=1/8/32/64, and optimizer state.
- **`table3_comparison.csv`**: 40 keyed comparisons, unrounded measured bytes,
  printed paper values, and signed MB differences (Jetson minus printed paper).
- `measurements.csv`, `results.json`, `reductions.csv`: raw metrics, validation
  checks, actual module-class counts, provenance, and same-backbone/batch reductions.
- `cases/*.json`, `*.log`, `*.trace.json.gz`: individual results and evidence.

Exports are refreshed after each case. PENDING/FAILED cells are never zero-filled.
Partial sweeps keep the full reference grid; their unmeasured cells stay PENDING.
Comparison requires rank 2. Other ranks and non-paper batches remain available in
the ordinary measurement files, without misleading reference deltas.

The reference is the printed **MemFLoRA (58).pdf, Table 3, page 5**, supplied on
2026-09-30. It deliberately retains the original MobileNetV2 Full FT saved cells
16.12 and 81.91, and SG optimizer 0.16. Those known paper errors are NOT measurement
targets. Differences against printed values include rounding; the paper's older
reference-counting and single-buffer estimates differ from the new metric.

See [comparison template](table3_comparison_template.md) for the paired layout.
No old or predicted Jetson numbers are substituted into a new run.

## Corrected LoRA-Edge selection

The adapter factory now passes `optimized=True` for `lora_edge_optimized`.
Previously that method accidentally built ordinary `LoRAEdgeConv2d`, which retains
different backward state. The original run ending `223913_130962` therefore has
eight LoRA-Edge cases that must not be treated as the intended optimized baseline.

The profiler now verifies every LoRA-Edge wrapper is exactly
`LoRAEdgeConv2dOptimized` before warm-up. Empty, ordinary, V2, or mixed wrapper
sets are rejected. Actual module-class counts and the verification check are
recorded in each successful result. A legacy LoRA-Edge result lacking this check
is marked UNVERIFIED IMPLEMENTATION by the comparison exporter.

## What the numbers mean

All units are decimal MB = 1,000,000 bytes. The peak is the maximum concurrent
sum of distinct requested backing allocations attributed to model parameters and
buffers, materialized ordinary/projection Adam state, parameter gradients,
saved-backward storage, SG captured gradients, and adapted-site replay inputs.
Aliased references count once; separate allocations count separately.
Replay-input overlap is measured, not replaced by a one-buffer estimate.

The value after the slash is distinct saved-backward storage at the end of
forward, including masks/constants/materialized weights saved by autograd.
It is not peak minus model minus optimizer. Optimizer bytes are already included
in peak; SG includes both optimizers. Core peak excluding replay-only allocations
is also available, but independent maxima must not be added or subtracted as if
they happened at the same time.

Inputs/labels, application checkpoint copies, general intermediates/workspaces,
allocator cache, and host optimizer scalars are outside training-state scope.
Requested/allocated/reserved CUDA peaks are separate runtime metrics, not additive
pools, whole-process RAM, or physical on-chip SRAM. Counter agreement checks totals;
it is not an independent proof of every semantic attribution.

The workload is synthetic FP32 Opportunity-shaped input (97 channels, 60 samples,
17 classes), random initial weights, no AdaBN calibration. Full FT/LoRA retain their
train-mode BN policy; MemFLoRA/SG use their eval-mode BN policy. SG runs capture,
backward, replay with unchanged parameters, and both optimizer updates. The CUDA
bitpacking extension is required unless `--allow-bitpack-fallback` is explicit.

Instrumentation can alter scheduling. This is not a latency benchmark or an
accuracy reproduction. Library versions, source hashes, allocator settings, and
backend flags are recorded with the results.

## Lightweight checks and options

List the cases without importing PyTorch or running a model:

```bash
python tools/jetson_profile.py --dry-run
```

Run local accounting/export tests (standard library only):

```bash
python -m unittest discover -s tests -p test_jetson_profile.py -v
```

Verify adapter selection, saved-input behavior, and forward/gradient equivalence
with tiny CPU models (requires PyTorch, no CUDA):

```bash
python -m unittest discover -s tests -p test_lora_edge_injection.py -v
```

For a future targeted measurement, use
`--models tresnet --methods lora_edge --batches 1`.
Optional `--checkpoint-copy` retains application-style copies outside training
state. `--save-snapshots` and `--trace-stacks` provide additional diagnostics.
