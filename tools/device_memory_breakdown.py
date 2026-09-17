"""Measure where memory goes when MemFLoRA adapts T-ResNet on Opportunity.

Written for a Jetson, where the CPU and GPU share one memory pool, but it also
runs on an ordinary CUDA machine or on a CPU. Run each rank in its own process so
the second run does not inherit the first run's CUDA context, loaded kernels or
allocator cache:

    python tools/device_memory_breakdown.py --rank 2
    python tools/device_memory_breakdown.py --rank 4

Nothing is pretrained. How much memory an adaptation step needs depends on
tensor shapes, not on the values of the weights, so random weights give the same
numbers as a trained source model and save twenty epochs of pretraining.

Stages run in the order a real run pays for them: interpreter, PyTorch, CUDA
context, dataset, model, calibration, then training steps. Each component is the
growth measured across the stage that created it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2
METHOD = "bnpa_fa_postbn_bnr_off_scaled"  # MemFLoRA
SEED = 1

# The paper-style Opportunity + T-ResNet settings, minus the sweep dimensions.
# fmt: off
BENCHMARK_ARGV = [
    "--dataset", "opportunity",
    "--backbone", "t_resnet_official",
    "--t-resnet-feature-maps", "64",
    "--method", METHOD,
    "--target-subjects", "all",
    "--window-size", "60",
    "--window-stride", "30",
    "--label-column", "ml_both_arms",
    "--window-label-rule", "majority",
    "--adapter-layers", "all",
    "--adabn-calibration-mode", "ema_no_reset",
    "--adabn-calib-batches", "1",
    "--proj-init", "random",
    "--bnpa-bottleneck-bn", "on",
    "--adabn-stat-source", "target",
    "--train_mode_adaBN", "off",
    "--adapt-lr", "1e-3",
]
# fmt: on
GROUPS = (
    "Process and libraries",
    "Host data",
    "Live between steps",
    "Extra at the peak of a step",
    "Allocator cache (reserved, not used)",
    "Evaluation",
    "Benchmark runner only",
)


def process_rss() -> int | None:
    """Resident memory of this process, in bytes."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def system_used() -> int | None:
    """Memory in use across the whole system, MemTotal minus MemAvailable."""
    try:
        fields = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                fields[key] = int(value.split()[0]) * 1024
        return fields["MemTotal"] - fields["MemAvailable"]
    except (OSError, KeyError, ValueError):
        return None


def is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").exists()


class Recorder:
    """Snapshots of process, system and CUDA allocator memory at each stage."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.torch = None
        self.device = None

    @property
    def cuda(self) -> bool:
        return self.device is not None and self.device.type == "cuda"

    def snapshot(self, stage: str) -> dict:
        row = {"stage": stage, "rss": process_rss(), "system_used": system_used()}
        if self.cuda:
            torch = self.torch
            torch.cuda.synchronize(self.device)
            row["cuda_allocated"] = torch.cuda.memory_allocated(self.device)
            row["cuda_reserved"] = torch.cuda.memory_reserved(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            row["device_used"] = total - free
        self.rows.append(row)
        return row


def delta(before: dict, after: dict, key: str) -> int | None:
    if before.get(key) is None or after.get(key) is None:
        return None
    return after[key] - before[key]


def outside_allocator(before: dict, after: dict) -> int | None:
    """Memory growth that the PyTorch caching allocator did not hand out.

    Once CUDA is up, the driver's own view of used device memory is compared with
    the allocator's reserved pool. Before that, system memory is the only signal;
    on a Jetson the allocator's pool is carved out of it, so it is subtracted.
    """
    reserved = delta(before, after, "cuda_reserved") or 0
    device_grown = delta(before, after, "device_used")
    if device_grown is not None:
        return device_grown - reserved
    grown = delta(before, after, "system_used")
    if grown is None:
        return None
    return grown - reserved if is_jetson() else grown


def storage_bytes(tensors) -> int:
    """Physical bytes behind `tensors`, counting each shared storage once."""
    seen, total = set(), 0
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-root", default="data/opportunity")
    parser.add_argument("--target-subject", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2,
        help="steps before measuring, so Adam state and kernel selection exist",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="let cuDNN search algorithms; this can grow the workspace a lot",
    )
    parser.add_argument("--json", type=Path, help="also write raw bytes here")
    return parser.parse_args()


def main() -> int:
    cli = parse_cli()
    rec = Recorder()
    start = rec.snapshot("python interpreter")

    import torch  # imported here so the interpreter is measured on its own

    rec.torch = torch
    after_torch = rec.snapshot("import torch")

    sys.path.insert(0, str(ROOT))
    from experiments.benchmark_adabn import calibrate_minimal_adabn_if_needed
    from experiments.benchmark_cli import apply_dataset_defaults, parse_args
    from experiments.benchmark_common import prepare_batch_inputs
    from experiments.minimal_methods_benchmark import (
        build_backbone,
        configure_method,
    )
    from src.data.benchmark import (
        build_minimal_split,
        load_minimal_dataset_arrays,
        make_loaders,
    )
    from src.profiling.sram import SavedTensorProfiler
    from src.train import freeze_bn_eval

    after_imports = rec.snapshot("import project")

    if cli.device == "cuda" and not torch.cuda.is_available():
        print(
            "CUDA was requested but torch.cuda.is_available() is False. Check that "
            "this PyTorch build has CUDA support, or pass --device cpu.",
            file=sys.stderr,
        )
        return 2
    device = torch.device(cli.device)
    rec.device = device
    torch.manual_seed(SEED)
    if rec.cuda:
        torch.backends.cudnn.benchmark = cli.cudnn_benchmark
        torch.cuda.init()
        torch.zeros(1, device=device)
    after_context = rec.snapshot("cuda context")

    argv = BENCHMARK_ARGV + [
        "--data-root", cli.data_root,
        "--rank", str(cli.rank),
        "--batch-size", str(cli.batch_size),
    ]  # fmt: skip
    args = parse_args(argv)
    apply_dataset_defaults(args)
    args.batch_size = cli.batch_size
    args.seed = SEED
    trial = argparse.Namespace(
        **{
            **vars(args),
            "method": METHOD,
            "rank": cli.rank,
            "adabn_calib_batches": 1,
            "adapt_lr": 1e-3,
            "steps_adapt": 50,
            "target_domain": str(cli.target_subject),
        }
    )

    with tempfile.TemporaryDirectory() as scratch:
        arrays = load_minimal_dataset_arrays(args, Path(scratch))
    after_dataset = rec.snapshot("dataset arrays")

    split = build_minimal_split(args, arrays, cli.target_subject, SEED)
    loaders = make_loaders(split, cli.batch_size, SEED)
    after_split = rec.snapshot("split and loaders")

    model = build_backbone(trial).to(device)
    configure_method(model, trial)
    model.to(device)
    after_model = rec.snapshot("model with adapters")

    calibrate_minimal_adabn_if_needed(
        model, METHOD, loaders["Shift_train"], trial, device
    )
    after_calibration = rec.snapshot("adabn calibration")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=trial.adapt_lr, weight_decay=trial.adapt_weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    batches = iter(loaders["Shift_train"])

    def next_batch():
        nonlocal batches
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(loaders["Shift_train"])
            x, y = next(batches)
        x = prepare_batch_inputs(x.to(device), args.backbone)
        return x, y.to(device)

    def step(x, y):
        model.train()
        freeze_bn_eval(model)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # The runner keeps a copy of the best weights for checkpoint selection.
    best_state = deepcopy(model.state_dict())

    for _ in range(max(1, cli.warmup_steps)):
        step(*next_batch())
    after_warmup = rec.snapshot("warm-up steps")

    # Peak of one ordinary step, measured without any hooks attached.
    x, y = next_batch()
    optimizer.zero_grad(set_to_none=True)
    peak = before_step = reserved_peak = None
    if rec.cuda:
        torch.cuda.synchronize(device)
        before_step = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    freeze_bn_eval(model)
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    if rec.cuda:
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        reserved_peak = torch.cuda.max_memory_reserved(device)
    after_step = rec.snapshot("measured step")
    gradient_bytes = storage_bytes(p.grad for p in trainable)

    # A second step counts what autograd saves for backward, storage by storage.
    profiler = SavedTensorProfiler(model)
    seen: set = set()

    def pack(tensor):
        storage = tensor.untyped_storage() if tensor.numel() else None
        key = None if storage is None else (str(tensor.device), storage.data_ptr())
        if key is not None and key not in seen:
            seen.add(key)
            kind, _ = profiler.add(tensor)
            profiler.bytes_by_kind[kind] += storage.nbytes() - tensor.numel() * (
                tensor.element_size()
            )
        return tensor

    x2, y2 = next_batch()
    model.train()
    freeze_bn_eval(model)
    optimizer.zero_grad(set_to_none=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        with profiler.module_contexts(model):
            loss = criterion(model(x2), y2)
    loss.backward()
    optimizer.step()
    saved = profiler.bytes_by_kind

    # One evaluation batch, as the runner does between adaptation steps.
    eval_overhead = None
    model.eval()
    with torch.no_grad():
        if rec.cuda:
            torch.cuda.synchronize(device)
            eval_before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        model(x)
        if rec.cuda:
            torch.cuda.synchronize(device)
            eval_overhead = torch.cuda.max_memory_allocated(device) - eval_before
    model.train()
    freeze_bn_eval(model)
    final = rec.snapshot("end")

    weights = storage_bytes([*model.parameters(), *model.buffers()])
    optimizer_state = storage_bytes(
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    checkpoint = storage_bytes(best_state.values())
    batch = storage_bytes([x, y])
    saved_float = saved["activation"] + saved["constant"]
    saved_bits = saved["bitmask"]

    remainder = None
    if peak is not None:
        remainder = peak - before_step - saved_float - saved_bits - gradient_bytes

    source = build_backbone(trial)
    source_bytes = storage_bytes([*source.parameters(), *source.buffers()])

    components = [
        ("Process and libraries", None),
        ("Python interpreter", start["rss"]),
        ("PyTorch import", delta(start, after_torch, "rss")),
        ("Project imports", delta(after_torch, after_imports, "rss")),
        (
            "CUDA context and driver",
            outside_allocator(after_imports, after_context) if rec.cuda else None,
        ),
        (
            "cuDNN / cuBLAS handles and kernels",
            None
            if not rec.cuda
            else (outside_allocator(after_model, after_calibration) or 0)
            + (outside_allocator(after_calibration, after_warmup) or 0),
        ),
        ("Host data", None),
        ("Dataset arrays, all subjects", delta(after_context, after_dataset, "rss")),
        ("Split tensors and loaders", delta(after_dataset, after_split, "rss")),
        ("Live between steps", None),
        ("Weights and buffers", weights),
        ("Adam optimizer state", optimizer_state),
        ("Gradients", gradient_bytes),
        ("Best-checkpoint copy", checkpoint),
        ("Input batch", batch),
        ("Extra at the peak of a step", None),
        ("Saved activations", saved_float),
        ("Saved ReLU bitmasks", saved_bits),
        ("Workspace and intermediates (remainder)", remainder),
        ("Allocator cache (reserved, not used)", None),
        ("At peak", None if peak is None else reserved_peak - peak),
        ("Evaluation", None),
        ("One no-grad eval batch", eval_overhead),
        ("Benchmark runner only", None),
        ("Source model and its state copy", 2 * source_bytes),
    ]

    print_report(cli, rec, torch, components, before_step, peak, start, final)

    if cli.json:
        cli.json.write_text(
            json.dumps(
                {
                    "rank": cli.rank,
                    "batch_size": cli.batch_size,
                    "device": str(device),
                    "jetson": is_jetson(),
                    "components_bytes": {
                        name: value for name, value in components if name not in GROUPS
                    },
                    "peak_allocated_bytes": peak,
                    "stages": rec.rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


def print_report(cli, rec, torch, components, before_step, peak, start, final):
    def mib(value):
        return "n/a" if value is None else f"{value / MIB:10.2f} MiB"

    print(f"\nMemFLoRA, T-ResNet, Opportunity  rank={cli.rank}  batch={cli.batch_size}")
    print(f"device={rec.device}  jetson={is_jetson()}  torch={torch.__version__}")
    if rec.cuda:
        print(
            f"gpu={torch.cuda.get_device_name(rec.device)}  "
            f"cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}  "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark}"
        )
    for name, value in components:
        if name in GROUPS:
            print(f"\n{name}")
        else:
            print(f"  {name:<44}{mib(value)}")

    print("\nTotals")
    if peak is not None:
        print(f"  {'Allocator before the measured step':<44}{mib(before_step)}")
        print(f"  {'Allocator peak during the step':<44}{mib(peak)}")
    print(f"  {'Process resident memory at the end':<44}{mib(final['rss'])}")
    print(
        f"  {'System memory used by this run':<44}"
        f"{mib(delta(start, final, 'system_used'))}"
    )
    print(
        "\nSystem-wide numbers include anything else running at the time. Close "
        "other programs, run each rank in a fresh process, and repeat a few times."
    )


if __name__ == "__main__":
    sys.exit(main())
