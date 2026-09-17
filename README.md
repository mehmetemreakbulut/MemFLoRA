# MemFLoRA

Memory-efficient adapter-based domain adaptation for on-device human activity
recognition.

This repository contains the code and the reference results for the MemFLoRA
paper. Every experiment in the paper has a script in `scripts/`, and the output
that script produced is committed under `results/` — so the tables and figures can
be checked without re-running anything, and a re-run can be diffed against the
original.

## What is here

```text
scripts/     21 experiment scripts, plus a smoke test
results/     the output those scripts produced, grouped by paper section
experiments/ the benchmark runner
src/         adapters, backbones, dataset loaders, memory/performance profiling
tests/       unit tests (no dataset needed)
docs/        DATA.md — how to get the datasets
             REPRODUCIBILITY.md — experiment index and how to re-run
```

## Quickstart

```bash
make install                # pinned deps + editable install
make test                   # unit tests, no data required
# put the datasets in place — see docs/DATA.md
make smoke                  # few-minute end-to-end check
./scripts/01_main_accuracy_opportunity_t_resnet.sh
```

New runs go to `$RESULTS_ROOT` (default `runs/`) and never overwrite `results/`.

## The method

MemFLoRA adapts a frozen convolutional backbone to a new target domain by
inserting low-rank adapters whose backward pass does not require the full-precision
input activations that dominate training-time SRAM. Two variants are reported:

| Paper name | Code name |
|---|---|
| MemFLoRA | `bnpa_fa_postbn_bnr_off_scaled` |
| MemFLoRA-SG | `bnpa_q3_sg` |

Baselines: full fine-tuning, BN-tuning, bias-tuning, LoRA-C, LoRA-Edge, TinyTL
lite-residual, and the unadapted source model (`zero_shot`).

## Headline numbers

Opportunity + T-ResNet, leave-one-subject-out, 40 seeds, 50 adaptation steps —
from `results/main_accuracy/opportunity_t_resnet/minimal_benchmark_summary.csv`:

| Method | Rank | Macro-F1 | ± stderr |
|---|---|---|---|
| Full fine-tuning | — | 0.8435 | 0.0012 |
| MemFLoRA-SG | 8 | 0.8092 | 0.0024 |
| **MemFLoRA** | **8** | **0.8058** | **0.0023** |
| LoRA-C | 8 | 0.7962 | 0.0026 |
| LoRA-Edge | 8 | 0.7641 | 0.0030 |
| BN-tuning | — | 0.6766 | 0.0045 |
| Bias-tuning | — | 0.6502 | 0.0050 |
| Source model, no adaptation | — | 0.5126 | 0.0039 |

Peak training SRAM for the same backbone, batch size 64 — from
`results/memory_profiling/opportunity_t_resnet/full_sram_summary.csv`:

| Method | Rank | Peak SRAM |
|---|---|---|
| Full fine-tuning | — | 60.5 MiB |
| LoRA-C | 2 | 57.0 MiB |
| BN-tuning | — | 54.8 MiB |
| LoRA-Edge | 2 | 38.2 MiB |
| **MemFLoRA** | **2** | **3.1 MiB** |

MemFLoRA reaches within ~4 macro-F1 points of full fine-tuning at rank 8 while
training in roughly a twentieth of the peak SRAM at rank 2. These are the
Opportunity/T-ResNet numbers; RealDisp and RealWorld results, both backbones, are
in `results/main_accuracy/`.

## Reproducing

`docs/REPRODUCIBILITY.md` has the full index: which script produces which results
folder, what each supports in the paper, what the shipped files contain, and what
should and should not reproduce exactly on different hardware.

Dependencies are declared in `pyproject.toml`; `requirements-lock.txt` preserves
the original complete environment. RealDisp always uses the NumPy parser, so
installing pandas cannot change preprocessing. Datasets are not included; see
`docs/DATA.md` for their expected layout.

New results use compact CSV schemas: 20 core trial fields, with SG counters and
time-to-threshold fields included only when relevant. `target_domain` identifies
the subject, location, or scenario. Historical reference files retain their
original columns. Profiling is opt-in; the profiling scripts enable it explicitly.

## Citation

A citation entry will be added on publication.

## License

MIT — see `LICENSE`.
