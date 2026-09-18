# MemFLoRA

Memory-Floor LoRA: low-rank adapters for fine-tuning CNNs on small devices,
where the thing you run out of is memory, not parameters.

Cutting the  number of trainable weights shrinks the gradients and the optimizer state, but for
a convolutional network those were never the big cost. The big cost is the
activations that autograd keeps around for the backward pass, and a normal LoRA
adapter still keeps every one of them at full width.

MemFLoRA is built around that problem instead. The rule it follows is simple: no
trainable part of the backward pass is allowed to need a full-width activation.
To get there it

- freezes the down-projection and only trains the up-projection, so the only
  thing saved for the adapter is a rank-`r` bottleneck,
- runs the backbone's BatchNorm in eval mode, which turns it into a fixed affine
  map with nothing to save,
- stores the ReLU mask as packed bits instead of a float tensor, and
- scales the adapter output by the frozen BatchNorm scale, so it lives on the same
  scale as the backbone it's added to.

There's also an optional variant, MemFLoRA-SG, that recovers the gradient for the
frozen projection by replaying the forward pass, so the projection can learn too
without keeping full-width activations around.

The code was written for sensor-based human activity recognition, but nothing in
the adapter itself is specific to that.

## Method names

The method names in the code are internal ones that stuck around from
development. The two you probably care about:

| Name | In the code |
|---|---|
| MemFLoRA | `bnpa_fa_postbn_bnr_off_scaled` |
| MemFLoRA-SG | `bnpa_q3_sg` |

The baselines are `zero_shot` (no adaptation), `full` (full fine-tuning),
`bn_tuning`, `bias_tuning`, `lora_c`, `lora_edge_optimized`, and the
`tinytl_lite_residual_bias*` family. The full list lives in `src/methods.py`.

## Setup

You'll need Python 3.11 or newer.

```bash
pip install -r requirements.txt     # or: make install
```

That installs the repo in editable mode along with pinned versions of torch,
torchvision, numpy, scikit-learn and pillow, plus pytest and black. If you prefer
conda, `conda env create -f environment.yml` does the same thing.

`requirements-lock.txt` is a full `pip freeze` of the machine the code was
developed on. You don't need it for normal use, and it pins CUDA packages that
will only install on Linux.

## Data

The datasets aren't included, so you'll have to download them yourself. The code
supports three public HAR datasets: Opportunity, RealDisp and RealWorld (HAR). By
default it looks for them under `data/<dataset>`, laid out like this:

```text
data/
├── opportunity/
│   └── OpportunityUCIDataset/dataset/S1-ADL1.dat ...
├── realdisp/
│   └── subject1_ideal.log, subject1_self.log ...
└── realworld/
    └── realworld2016_dataset/proband1/data/acc_walking_csv.zip ...
```

The loaders search these folders fairly loosely, so small differences in nesting
are usually fine. If your data lives somewhere else, pass `--data-root`.

## Running things

Start with the smoke test. It trains for one epoch on a single Opportunity subject
and finishes in a few minutes, which is enough to tell you the data, the model
and the adapters are all wired up correctly. Don't read anything into its
accuracy.

```bash
make smoke
```

Everything goes through one entry point, `experiments/minimal_methods_benchmark.py`.
Here's a realistic run: MemFLoRA at rank 4 on Opportunity with a T-ResNet, every
subject held out in turn, five seeds, 50 adaptation steps.

```bash
python experiments/minimal_methods_benchmark.py \
  --dataset opportunity --backbone t_resnet_official \
  --method bnpa_fa_postbn_bnr_off_scaled \
  --rank 4 --target-subjects all --seed 1 2 3 4 5 \
  --window-size 60 --window-stride 30 \
  --label-column ml_both_arms --window-label-rule majority \
  --pretrain-epochs 20 --steps-adapt 50 --batch-size 64 \
  --reuse-first-source-model true --adapt-eval-every-steps 50 \
  --adapter-layers all --adabn-calibration-mode ema_no_reset \
  --adabn-calib-batches 1 --proj-init random --bnpa-bottleneck-bn on \
  --adabn-stat-source target --train_mode_adaBN off --adapt-lr 1e-3 \
  --results-root runs
```

Most arguments accept several values and the runner sweeps over all of them, so
`--method zero_shot full bnpa_fa_postbn_bnr_off_scaled --rank 2 4 8` runs every
combination. Each run gets its own dated folder under `--results-root`, with the
exact command, the resolved config, a CSV row per trial and a summary CSV.

A few things that tripped me up and might trip you up too:

- **`--adabn-calib-batches` defaults to `all`.** If you want AdaBN calibrated on a
  single batch of target data, pass `1` explicitly.
- **`--reuse-first-source-model true` saves a lot of time.** The source model is
  pretrained once per target and reused across seeds. Setting it to `false`
  retrains from scratch for every seed, which is more independent but much slower.
- **CPU works, it's just slow.** A 20-epoch Opportunity pretrain takes a few
  minutes on a CPU. Big sweeps really want a GPU.
- **MobileNetV2 wants `--adapter-layers pointwise_only`.** The MemFLoRA adapter
  only supports ungrouped convolutions, and MobileNetV2 is full of depthwise ones.

For memory numbers, add `--profile-full-sram true`, and for multiply-accumulate
counts add `--profile-performance true`. Both write their own CSVs next to the
trial results.

The `scripts/` folder has ready-made sweeps grouped by what they measure:
accuracy, convergence over adaptation steps, memory profiling, design ablations, a
TinyTL comparison and adaptation time. You can run a single script or a whole
group:

```bash
./scripts/15_ablations_geometry.sh
./scripts/run_all.sh ablations
```

Output goes to `$RESULTS_ROOT`, which defaults to `runs/`. Some of these sweeps
are large and take days on a CPU, so read a script before you start it.

## Jetson memory report and CUDA packing

`tools/jetson_memory.py` compares MemFLoRA and full fine-tuning in separate
processes using random weights and Opportunity-shaped synthetic inputs. It does
not reproduce the paper's accuracy experiment. AdaBN calibration is off by
default; `--use-adabn` enables one calibration batch.

```bash
python tools/jetson_memory.py --model mobilenetv2 --rank 2 --batch 64 \
  --bitpack-backend cuda
```

The report is served at `http://127.0.0.1:8000`; Ctrl+C saves `report.html` beside
`data.json` under `runs/jetson_memory_mobilenetv2_r2/`. Over SSH, forward port 8000.
Use `--out` to keep comparisons in separate directories.

The optional CUDA extension packs eight gates per byte and fuses ReLU/ReLU6 with
packing for contiguous adapted and frozen blocks. This avoids a full-size boolean
mask in those blocks. The torch fallback also avoids the previous int64 packing
reduction. Backward uses the existing unpacker and gradient formulas.

Building the extension requires a CUDA-enabled PyTorch compatible with your
JetPack installation, the CUDA toolkit (`nvcc`), a C++ compiler, and `ninja`
(`pip install ninja`). Keep your Jetson-compatible PyTorch installation. The
extension builds on first use and is cached by PyTorch; the profiler prepares it
before warm-up. Compilation defaults to one job to limit RAM use; `MAX_JOBS` can
override this. Copy the changed `src/` files, including `src/utils/csrc/`, to the
Jetson together with the profiler.

- `--bitpack-backend cuda` requires the compiled kernel and fails if it cannot load.
- `--bitpack-backend auto` (default) attempts compilation and warns before falling
  back to torch if the build fails.
- `--bitpack-backend torch` uses only torch operations, with no compilation.

For the main benchmark, select the same behavior with the environment variable
`MEMFLORA_BITPACK_BACKEND=cuda`, `auto`, or `torch`.

The report separates maximum requested CUDA memory, maximum reserved memory,
maximum training state, and saved state at the end of forward. Its category table
is a snapshot at the overall requested peak: a zero mask row can mean the peak
occurred before masks were created or after they were released. Full fine-tuning
does not use MemFLoRA's packed masks. Linux RSS, private pages, and PSS are separate
views and must not be added to CUDA memory or used to infer library overhead by
subtracting CUDA bytes. Source hashes, software versions, and kernel backend
status are included in the report.

`--trace-stacks --save-snapshot` records allocation stacks and saves PyTorch
allocator snapshots for investigation; stack recording adds overhead.
`--reverse-order` measures full fine-tuning first. With calibration enabled,
`--empty-cache-after-calibration` releases unused cached CUDA blocks after
calibration; it does not free live tensors or guarantee a lower training peak.

## What's where

```text
src/
  adapters/     the MemFLoRA blocks and their hand-written backward passes,
                plus LoRA-C, LoRA-Edge, TinyTL and the frozen minimal blocks
  models/       T-ResNet and MobileNetV2, and the code that injects adapters
  data/         loaders for the three datasets and the train/test splits
  profiling/    saved-tensor SRAM accounting and MAC counting
  utils/        AdaBN helpers, bit packing, CSV writing
  methods.py    method names and which settings each one uses
  train.py      AdaBN calibration
experiments/    the benchmark runner: CLI, training loop, reporting
scripts/        shell wrappers around the runner
tests/          unit tests, no datasets needed
```

If you want to understand the method, start with
`src/adapters/bnpa_conv_bn_act_2d.py`. The forward and backward of the fused
Conv–BN–ReLU block are written out by hand there, and that's where the memory
savings actually happen.

## Tests and formatting

```bash
make test
black --check src experiments tests
```

The tests don't need any data. They check bit packing, compare the BNPA backward
against a plain autograd reference, and exercise adapter injection and the CSV
output. Code is formatted with black at the default 88 columns.

Compiled-kernel tests are opt-in and require CUDA and the build tools above:

```bash
MEMFLORA_TEST_CUDA_KERNEL=1 pytest tests/test_bitpack.py tests/test_fused_packing_blocks.py
```

These check byte order, tail padding, activation boundaries, non-default CUDA
streams, and block outputs/gradients. `tests/test_jetson_memory.py` checks profiler
accounting with synthetic traces and mocked Linux counters.

## License

MIT. See `LICENSE`.
