# Reproduction scripts

There are 21 maintained experiment scripts and one smoke test. They share shell
setup in `common.sh`; experiment arguments preserve their recorded values and
ordering. Historical outputs remain under `results/`.

## Running

```bash
./scripts/00_smoke_test.sh          # minutes: checks the pipeline works
./scripts/01_main_accuracy_opportunity_t_resnet.sh
./scripts/run_all.sh main           # a whole group
make all                            # all 21 paper experiments
```

Every script writes to `$RESULTS_ROOT` (default `runs/`), never to `results/`,
so a new run can be compared against the shipped reference rather than replacing
it.

| Variable | Default | Meaning |
|---|---|---|
| `PYTHON` | `python3` | interpreter to use |
| `RESULTS_ROOT` | `runs` | where new output is written |

## The experiments

| # | Script | Supports |
|---|---|---|
| 00 | `00_smoke_test.sh` | nothing — pipeline check only |
| 01–06 | `*_main_accuracy_*` | main accuracy table (3 datasets × 2 backbones) |
| 07–08 | `*_convergence_*` | adaptation-step curves |
| 09–12 | `*_memory_profiling_*` | peak SRAM, performance counters, batch-size scaling |
| 14–17 | `*_ablations_*` | initialization, geometry, output scale / bottleneck-BN, AdaBN |
| 20–22 | `*_tinytl_comparison_*` | TinyTL-style accuracy and memory comparison |
| 23–24 | `*_time_spec_*` | adaptation loop time, threshold-reaching steps |

Groups for `run_all.sh`: `main`, `convergence`, `memory`, `ablations`, `tinytl`,
`time_spec`, `all`.

## Before you run anything

1. **Install** — `make install` (dependencies are declared in `pyproject.toml`).
2. **Get the data** — datasets are not redistributable and are not in this
   repository. `docs/DATA.md` says where to obtain Opportunity, RealDisp and
   RealWorld and where to put them.
3. **Check it works** — `make smoke`.

## Cost

These are full paper runs: up to 40 seeds × 3 ranks × 8 methods, each with 20
pretraining epochs. Experiments 01–08 are the expensive ones and need a GPU.
The profiling and ablation runs (09–24) are much smaller.

Reference outputs are committed under `results/` for comparison. `run_all.sh all`
runs the 21 paper scripts; run the smoke test separately. `make smoke` writes
under `$(RESULTS_ROOT)/smoke`, while the smoke script alone defaults to `runs/smoke`.
