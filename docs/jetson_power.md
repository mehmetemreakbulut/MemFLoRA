# Small Jetson power/time benchmark

From the Jetson checkout:

```bash
~/venvs/memflora/bin/python tools/jetson_power.py
```

Defaults: both backbones, all five Table 3 methods, rank 2, batch 64,
three independent process repeats per configuration. Each repeat uses a 5-second
warm-up, 5-second idle settling period, 10-second idle baseline, and at least
30 seconds of sustained complete training updates. Allow roughly 25 minutes of
these windows in total, plus model setup, extension compilation, and other overhead.
No pretraining, dataset download, accuracy evaluation, or reverse-order sweep.

For a small Full FT / MemFLoRA-only experiment:

```bash
~/venvs/memflora/bin/python tools/jetson_power.py --methods full memflora
```

The output is just progress lines, `summary.csv` (one row per configuration,
repeat means and sample standard deviations), `results.json` and per-repeat JSON/logs.
No HTML report. Existing output directories are never overwritten. `--dry-run`
lists cases without CUDA or telemetry. A host lock prevents duplicate copies of
this script. Other GPU workloads must be stopped separately by the user.

Columns: `idle_w`, `total_w`, `above_idle_w`, `ms_per_update`,
`total_j_per_update`, `above_idle_j_per_update`, plus `_sd` columns.

**Baseline subtraction:** idle means the same warmed, model-loaded process with
no updates, not the OS-only idle state and not the rated TDP/power-mode setting.
`above_idle_w = total_w - idle_w`; `above_idle_j_per_update =
(integrated_active_energy - idle_w * active_seconds) / updates`.
Both raw and subtracted results are retained. Negative differences are flagged,
not clamped. A baseline subtraction is an estimate, not perfect isolation of
training-only energy; clocks and temperatures can change with load.

The sampler reads the current `VDD_IN` value from tegrastats every 200 ms, uses
monotonic receipt timestamps, and integrates with trapezoids and interpolated
window boundaries. Missing rails, insufficient samples, uncovered boundaries,
or large telemetry gaps fail the case rather than yielding a zero. Raw lines
retain temperature/clock/utilization context. This is module-input telemetry,
not wall-plug or GPU-only power. Sampling and sensor averaging limit precision.

`ms_per_update` is synchronized block wall time divided by completed updates,
including host work and synchronization every 10 updates; it is an amortized
update time, not individual-step latency percentiles. Initial imports, compilation,
warm-up, idle windows, and file exports are outside this active window. The
workload uses synthetic FP32 Opportunity-shaped inputs already on the GPU,
random weights, and no input pipeline/checkpoint copies. SG includes replay and
both optimizer updates. Existing method/BN policies and verified optimized
LoRA-Edge are reused; CUDA bitpacking is required for MemFLoRA/SG. FP32 storage
does not imply TF32 is disabled; backend flags are recorded.

Memory tracing and autograd save instrumentation are not enabled. The existing
memory profilers are unchanged. Device/software/source hashes and read-only
power-mode/clock queries are recorded. This script does not alter clocks, power
modes, cooling, or other processes. Review raw telemetry for thermal drift or
throttling; results are not automatically certified steady-state. Failed repeats
leave their summary row incomplete; they are not silently excluded from averages.

Tests (standard library only):

```bash
python3 -m unittest discover -s tests -p test_jetson_power.py -v
```
