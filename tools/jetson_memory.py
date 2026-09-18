"""Profile the memory of MemFLoRA and full fine-tuning on a Jetson.

    python tools/jetson_memory.py --model tresnet --rank 2
    python tools/jetson_memory.py --model mobilenetv2 --rank 2

Each method runs in a fresh process. The CUDA allocator trace reports requested
and reserved peaks separately, plus the training-state maximum and the category
breakdown at the overall requested peak. The report calls requested bytes memory
"used" and shows reserved memory separately. Whole-device RAM is estimated from
Linux MemTotal - MemAvailable and includes the OS and other programs; it is never
added to the CUDA totals.

The report is served at http://127.0.0.1:8000 and refreshes itself every second.
Over SSH, open a tunnel first: ssh -L 8000:localhost:8000 <jetson>. Press Ctrl+C
when done; a standalone copy is saved as report.html. Numbers are in MB (10^6 B).

The step uses random weights and random windows of the Opportunity shape (97
channels x 60 steps). This is a synthetic memory experiment, not a paper accuracy
run. AdaBN calibration is skipped unless --use-adabn is given. The optional CUDA
packing extension is prepared before warm-up; its first use may compile it.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"MemFLoRA": "bnpa_fa_postbn_bnr_off_scaled", "Full FT": "full"}
MODELS = {  # --model: display name, runner arguments
    "tresnet": (
        "T-ResNet",
        "--backbone t_resnet_official --t-resnet-feature-maps 64"
        " --adapter-layers all",
    ),
    "mobilenetv2": (
        "MobileNetV2",
        "--backbone mobilenet_v2 --mobilenet-v2-pretrained false --width-mult 1.0"
        " --adapter-layers pointwise_only",
    ),
}
# The paper's Opportunity settings.
COMMON_ARGS = (
    "--dataset opportunity --window-size 60 --window-stride 30"
    " --adabn-calibration-mode ema_no_reset --bnpa-bottleneck-bn on"
    " --adabn-stat-source target --train_mode_adaBN off"
)
# Category snapshots at the overall requested-byte peak, not separate maxima.
PEAK_BLOCKS = [
    [
        "A. Training state",
        [
            "Model state",
            "Optimizer state",
            "Parameter gradients",
            "Saved activations",
            "Saved ReLU bitmasks",
        ],
    ],
    [
        "B. Other working memory",
        [
            "Temporaries and workspace",
            "Input batch",
            "Best-checkpoint copy",
            "Other allocations kept between steps",
        ],
    ],
    ["C. Cache and allocation overhead", ["Allocator cache and rounding"]],
]
MARK = "@jetson_memory "  # marks the lines a measuring child sends back
LOCK = threading.Lock()


def device_memory() -> dict:
    """Estimated whole-device RAM use, not memory attributable to this process.

    MemAvailable accounts for memory Linux can reclaim for new applications:
    https://docs.kernel.org/filesystems/proc.html#meminfo
    """
    try:
        with open("/proc/meminfo") as f:
            info = {
                line.split(":")[0]: int(line.split()[1]) * 1024
                for line in f
                if line.startswith(("MemTotal:", "MemAvailable:"))
            }
        return {
            "device_ram_used": info["MemTotal"] - info["MemAvailable"],
            "device_ram_total": info["MemTotal"],
        }
    except (OSError, KeyError, ValueError):
        return {"device_ram_used": None, "device_ram_total": None}


class Monitor(threading.Thread):
    """Samples whole-device RAM, GPU, CPU and power every 0.5 s."""

    def __init__(self, samples: list) -> None:
        super().__init__(daemon=True)
        self.samples, self.process = samples, None

    def run(self) -> None:
        if shutil.which("tegrastats"):
            command = ["tegrastats", "--interval", "500"]
            if shutil.which("stdbuf"):  # otherwise the pipe buffers whole blocks
                command = ["stdbuf", "-oL", *command]
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
            for line in self.process.stdout:
                self.add(parse_tegrastats(line))
        else:
            previous = cpu_times()
            while True:
                time.sleep(0.5)
                current = cpu_times()
                loads = [
                    100 * (1 - (b[3] - a[3]) / max(1, sum(b) - sum(a)))
                    for a, b in zip(previous, current)
                ]
                previous = current
                self.add({"cpu": sum(loads) / len(loads), "cpu_max": max(loads)})

    def add(self, sample: dict) -> None:
        used = device_memory()["device_ram_used"]
        sample["device_ram"] = used / 1e6 if used is not None else None
        with LOCK:
            self.samples.append({"t": time.time(), **sample})


def parse_tegrastats(line: str) -> dict:
    cpu = re.search(r"CPU \[([^\]]*)\]", line)
    cores = [int(c) for c in re.findall(r"(\d+)%@", cpu.group(1))] if cpu else []
    gpu = re.search(r"GR3D_FREQ (\d+)%", line)
    power = re.search(r"VDD_IN (\d+)mW", line)
    return {
        "gpu": int(gpu.group(1)) if gpu else None,
        "cpu": sum(cores) / len(cores) if cores else None,
        "cpu_max": max(cores) if cores else None,
        "power": int(power.group(1)) / 1000 if power else None,
    }


def cpu_times() -> list[list[int]]:
    with open("/proc/stat") as f:
        return [
            [int(v) for v in line.split()[1:8]]
            for line in f
            if line.startswith("cpu") and line[3].isdigit()
        ]


def replay(trace: list, before: dict, reserved: int, kind) -> tuple:
    """Replay a step's allocator trace.

    `before` maps each block allocated when the trace began to its requested size,
    and `kind(i)` is 1, 2 or 3 for the saved tensor, gradient or temporary that
    event i allocates (0 is a block from before the step). Returns the requested
    bytes of each kind and the reserved bytes after every event (the first entry
    is the start), the entry of the peak, and the blocks live at the peak, as
    address -> (index of the alloc event, or -1; size).
    """

    def run(events):
        live = {address: (-1, size) for address, size in before.items()}
        totals = [sum(before.values()), 0, 0, 0, reserved]
        series = [totals.copy()]
        for i, event in enumerate(events):
            action, address = event["action"], event.get("addr")
            if action == "alloc":
                live[address] = (i, event["size"])
                totals[kind(i)] += event["size"]
            elif action in ("free_requested", "free_completed") and address in live:
                index, size = live.pop(address)
                totals[0 if index < 0 else kind(index)] -= size
            elif action in ("segment_alloc", "segment_map"):
                totals[4] += event["size"]
            elif action in ("segment_free", "segment_unmap"):
                totals[4] -= event["size"]
            series.append(totals.copy())
        return live, series

    series = run(trace)[1]
    peak_at = max(range(len(series)), key=lambda k: sum(series[k][:4]))
    return series, peak_at, run(trace[:peak_at])[0]


def attribute(trace, before, reserved, phases, saved, grads, owners):
    """Attribute every block live at the step's peak to a row of the table.

    `phases` holds trace-event offsets at the end of the forward and backward
    passes, `saved` maps the storage of each tensor
    autograd saved to its row, `grads` holds the gradients' storages and `owners`
    maps the tensors kept between steps to their row. Returns peak rows,
    independent maxima and forward-end metrics, and the step's timeline.
    Saved-state bytes exclude pre-existing model/optimizer/input storage.
    """
    allocs = [(i, e) for i, e in enumerate(trace) if e["action"] == "alloc"]
    # A saved tensor is the last block allocated at its address in the forward
    # pass; a gradient is the last block allocated at its address at all.
    forward = {event["addr"]: i for i, event in allocs if i < phases[0]}
    saved_at = {forward[a]: row for a, row in saved.items() if a in forward}
    last = {event["addr"]: i for i, event in allocs}
    grads_at = {last[a] for a in grads if a in last}

    def kind(i):
        return 1 if i in saved_at else 2 if i in grads_at else 3

    series, peak_at, live = replay(trace, before, reserved, kind)
    rows = dict.fromkeys(owners.values(), 0)
    for address, (index, size) in live.items():
        if index < 0:
            row = owners.get(address, "Other allocations kept between steps")
        elif index in saved_at:
            row = saved_at[index]
        elif index in grads_at:
            row = "Parameter gradients"
        else:
            row = "Temporaries and workspace"
        rows[row] = rows.get(row, 0) + size
    peak, held = sum(series[peak_at][:4]), series[peak_at][4]
    rows["Allocator cache and rounding"] = held - peak
    # Model and Adam storage is stable after warm-up. Track its lifetime too,
    # so a released pre-existing block cannot inflate the independent maximum.
    state_base = sum(
        size
        for address, size in before.items()
        if owners.get(address) in ("Model state", "Optimizer state")
    )
    live_base = dict(before)
    training_peak = state_base + series[0][1] + series[0][2]
    for i, event in enumerate(trace):
        if event["action"] in ("free_requested", "free_completed"):
            address = event.get("addr")
            size = live_base.pop(address, 0)
            if owners.get(address) in ("Model state", "Optimizer state"):
                state_base -= size
        training_peak = max(
            training_peak, state_base + series[i + 1][1] + series[i + 1][2]
        )
    forward_live = {}
    for i, event in enumerate(trace[: phases[0]]):
        if event["action"] == "alloc":
            forward_live[event["addr"]] = i
        elif event["action"] in ("free_requested", "free_completed"):
            forward_live.pop(event.get("addr"), None)
    forward_bytes = {"Saved activations": 0, "Saved ReLU bitmasks": 0}
    for i in forward_live.values():
        if i in saved_at:
            forward_bytes[saved_at[i]] += trace[i]["size"]
    metrics = {
        "requested_peak": peak,
        "reserved_at_requested_peak": held,
        "reserved_peak": max(point[4] for point in series),
        "training_state_peak": training_peak,
        "forward_saved_activations": forward_bytes["Saved activations"],
        "forward_saved_bitmasks": forward_bytes["Saved ReLU bitmasks"],
    }
    # About 300 points, each the highest of its stretch, so the peak survives.
    stretch = max(1, len(series) // 300)
    points = [
        max(
            ([k, *series[k]] for k in range(s, min(s + stretch, len(series)))),
            key=lambda p: sum(p[1:5]),
        )
        for s in range(0, len(series), stretch)
    ]
    # Keep exact boundaries and both peaks even when downsampling the plot.
    exact = {
        0,
        len(trace),
        *phases,
        peak_at,
        max(range(len(series)), key=lambda i: series[i][4]),
    }
    points = sorted({p[0]: p for p in points + [[i, *series[i]] for i in exact]}.values())
    timeline = {"points": points, "ends": phases, "peak": peak_at, "events": len(trace)}
    return rows, metrics, timeline


def emit(**message) -> None:
    print(MARK + json.dumps(message), flush=True)


def provenance(cli, method, torch, device):
    """Record enough context to distinguish builds and measurement policies."""
    from src.utils._bitpack_cuda import backend_status

    def read_optional(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    commit = None
    if shutil.which("git"):
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    sources = [
        Path(__file__).resolve(),
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "src").rglob("*.cu")),
        *sorted((ROOT / "src").rglob("*.cpp")),
        *sorted((ROOT / "experiments").rglob("*.py")),
        ROOT / "pyproject.toml",
    ]
    return {
        "method": method,
        "command": sys.argv,
        "python": platform.python_version(),
        "kernel": platform.release(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "jetpack_l4t": read_optional("/etc/nv_tegra_release"),
        "git_commit": commit.stdout.strip() if commit and commit.returncode == 0 else None,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "bitpack": backend_status(),
        "adabn": cli.use_adabn,
        "empty_cache_after_calibration": cli.empty_cache_after_calibration,
        "trace_stacks": cli.trace_stacks,
        "synthetic_inputs": True,
        "device_ram_measure": "MemTotal - MemAvailable; whole device, includes other programs",
    }


def measure(cli, method: str) -> None:
    """Measure one method. Runs in a child process and emits what it finds."""
    torch = None
    device = None

    def stage(name: str) -> dict:
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        row = {"name": name, "t": time.time(), **device_memory()}
        if device is not None and device.type == "cuda":
            row["reserved"] = torch.cuda.memory_reserved()
        emit(stage=row)
        return row

    initial = stage("start")
    import torch

    stage("import torch")
    sys.path.insert(0, str(ROOT))
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.benchmark_adabn import calibrate_minimal_adabn_if_needed
    from experiments.benchmark_cli import apply_dataset_defaults, parse_args
    from experiments.benchmark_common import method_forces_bn_eval
    from experiments.benchmark_common import prepare_batch_inputs
    from experiments.minimal_methods_benchmark import build_backbone, configure_method
    from src.train import freeze_bn_eval

    if cli.model == "mobilenetv2":
        import torchvision  # noqa: F401  the model imports it later; count it here

    stage("import project")
    if cli.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda", torch.cuda.current_device())
        torch.zeros(1, device=device).add_(1).item()  # fails early on a wrong build
    torch.manual_seed(1)
    stage("cuda context")
    if device.type == "cuda" and method != "full":
        from src.utils._bitpack_cuda import cuda_extension

        cuda_extension()  # compile/load outside the measured training step
    stage("packing backend setup")

    argv = [*COMMON_ARGS.split(), *MODELS[cli.model][1].split(), "--method", method]
    args = parse_args(argv + ["--rank", str(cli.rank)])
    apply_dataset_defaults(args)
    args = argparse.Namespace(
        **{
            **vars(args),
            "rank": cli.rank,
            "method": method,
            "batch_size": cli.batch,
            "adabn_calib_batches": 1 if cli.use_adabn else 0,
            "adapt_lr": 1e-3,
        }
    )
    windows = torch.randn(cli.batch, args.expected_input_channels, args.window_size)
    labels = torch.randint(args.expected_num_classes, (cli.batch,))
    loader = DataLoader(TensorDataset(windows, labels), batch_size=cli.batch)

    model = build_backbone(args).to(device)
    configure_method(model, args)
    model.to(device)
    stage("model")
    calibrate_minimal_adabn_if_needed(model, method, loader, args, device)
    stage("calibration")
    if cli.empty_cache_after_calibration and device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        stage("release unused calibration cache")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=args.adapt_weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoint = deepcopy(model.state_dict())  # the runner keeps the best weights
    batch_x, y = (tensor.to(device) for tensor in next(iter(loader)))
    x = prepare_batch_inputs(batch_x, args.backbone)
    bn_eval = method_forces_bn_eval(method)  # MemFLoRA freezes BN, full FT does not

    def start_step():
        model.train()
        if bn_eval:
            freeze_bn_eval(model)
        optimizer.zero_grad(set_to_none=True)  # frees the last step's gradients

    for _ in range(2):  # Adam's state, cuDNN and cuBLAS set themselves up here
        start_step()
        loss_fn(model(x), y).backward()
        optimizer.step()
    stage("warm-up")

    # The measured step: the same, with every allocation of it recorded.
    cuda = device.type == "cuda"
    saved = {}  # storage address -> (row, bytes) of each tensor autograd saves

    def pack(tensor):
        if tensor.numel() and tensor.device == device:
            storage = tensor.untyped_storage()
            bits = tensor.dtype in (torch.uint8, torch.bool)
            row = "Saved ReLU bitmasks" if bits else "Saved activations"
            saved[storage.data_ptr()] = (row, storage.nbytes())
        return tensor

    def trace_boundary() -> int:
        # Exact trace offset, including frees after the phase's final allocation.
        if not cuda:
            return 0
        return len(torch.cuda.memory._snapshot()["device_traces"][device.index])

    start_step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.memory._record_memory_history(
            "all",
            context="all" if cli.trace_stacks else None,
            max_entries=1_000_000,
        )
        before = {
            block["address"]: block["requested_size"]
            for segment in torch.cuda.memory._snapshot()["segments"]
            if segment["device"] == device.index
            for block in segment["blocks"]
            if block["state"] == "active_allocated"
        }
        reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = loss_fn(model(x), y)
    phases = [trace_boundary()]
    loss.backward()
    phases.append(trace_boundary())
    optimizer.step()
    s_done = stage("measured step")

    # Everything that exists between steps, by storage address.
    owners, sizes = {}, {}
    adam = [v for s in optimizer.state.values() for v in s.values()]
    kept = {
        "Model state": [*model.parameters(), *model.buffers()],
        "Optimizer state": adam,
        "Best-checkpoint copy": list(checkpoint.values()),
        "Input batch": [x, y],
    }
    for row, tensors in kept.items():
        for tensor in tensors:
            if tensor.device == device and tensor.numel():
                storage = tensor.untyped_storage()
                owners.setdefault(storage.data_ptr(), row)
                sizes[storage.data_ptr()] = storage.nbytes()
    grads = {p.grad.untyped_storage().data_ptr() for p in params if p.grad is not None}
    rows = {row: 0 for _, names in PEAK_BLOCKS for row in names}
    unmeasured = set()  # rows only the allocator trace can measure

    if cuda:
        snapshot = torch.cuda.memory._snapshot()
        trace = snapshot["device_traces"][device.index]
        torch.cuda.memory._record_memory_history(None)
        kinds = {address: row for address, (row, _) in saved.items()}
        found, metrics, timeline = attribute(
            trace, before, reserved, phases, kinds, grads, owners
        )
        rows.update(found)
        counters = torch.cuda.memory_stats()
        checks = {
            "requested_peak": (
                metrics["requested_peak"] == counters["requested_bytes.all.peak"]
            ),
            "reserved_peak": (
                metrics["reserved_peak"] == counters["reserved_bytes.all.peak"]
            ),
        }
        verified = all(checks.values())
        if cli.save_snapshot:
            import pickle

            out = cli.out or ROOT / "runs" / f"jetson_memory_{cli.model}_r{cli.rank}"
            out.mkdir(parents=True, exist_ok=True)
            with (out / f"{method}_snapshot.pickle").open("wb") as f:
                pickle.dump(snapshot, f)
    else:  # no allocator on the CPU: count every saved tensor and gradient
        for address, row in owners.items():
            rows[row] += sizes[address]
        for address, (row, size) in saved.items():
            if address not in owners:
                rows[row] += size
        rows["Parameter gradients"] = sum(
            p.grad.nbytes for p in params if p.grad is not None
        )
        metrics = {
            "requested_peak": None,
            "reserved_peak": None,
            "reserved_at_requested_peak": None,
            "training_state_peak": None,
            "forward_saved_activations": rows["Saved activations"],
            "forward_saved_bitmasks": rows["Saved ReLU bitmasks"],
        }
        # CPU bookkeeping is not a live-memory trace; never label it a peak.
        unmeasured = set(rows)
        verified = timeline = None
        checks = {}

    groups = [
        [block, [[row, None if row in unmeasured else rows[row]] for row in names]]
        for block, names in PEAK_BLOCKS
    ]
    for group in groups:  # each block carries its total
        values = [value for _, value in group[1] if value is not None]
        group.insert(1, sum(values) if values else None)
    totals = [
        [
            "A + B = GPU memory used",
            groups[0][1] + groups[1][1] if cuda else None,
        ],
        [
            "A + B + C = GPU memory reserved",
            sum(group[1] for group in groups) if cuda else None,
        ],
    ]
    summary = [
        ["Peak GPU memory used", metrics["requested_peak"]],
        ["Training state at that moment", groups[0][1]],
        ["Other working memory at that moment", groups[1][1]],
        ["GPU memory reserved at that moment", metrics["reserved_at_requested_peak"]],
        ["Peak GPU memory reserved", metrics["reserved_peak"]],
        ["Peak training state", metrics["training_state_peak"]],
        [
            "Saved for backward",
            metrics["forward_saved_activations"] + metrics["forward_saved_bitmasks"],
        ],
        ["Whole-device RAM in use", s_done["device_ram_used"]],
    ]
    metrics.update(
        device_ram_before=initial["device_ram_used"],
        device_ram_after=s_done["device_ram_used"],
        device_ram_total=s_done["device_ram_total"],
    )
    emit(
        device=str(device),
        verified=verified,
        groups=groups,
        totals=totals,
        summary=summary,
        timeline=timeline,
        metrics=metrics,
        checks=checks,
        metadata=provenance(cli, method, torch, device),
    )


def run_child(label: str, method: str, report: dict) -> None:
    """Measure one method in a fresh process, so it pays for its own libraries."""
    command = [sys.executable, __file__, *sys.argv[1:], "--child", method]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    assert child.stdout  # piped above
    for line in child.stdout:
        if not line.startswith(MARK):
            print(line, end="")
            continue
        message = json.loads(line[len(MARK) :])
        with LOCK:
            if "stage" in message:
                report["stages"].append({**message["stage"], "run": label})
            else:
                report["results"][label] = message
        if message.get("verified") is False:
            print(f"{label}: the allocator trace does not match PyTorch's peak")
    code = child.wait()
    if code:
        print(f"{label}: the measurement failed, see the error above")


def print_summary(report: dict) -> None:
    results = list(report["results"].values())
    if not results:
        return

    def line(name, values):
        cells = ["n/a" if v is None else f"{v / 1e6:.2f} MB" for v in values]
        print(f"{name:<58}" + "".join(f"{cell:>14}" for cell in cells))

    print(f"{'':<58}" + "".join(f"{label:>14}" for label in report["results"]))
    for i, (name, _) in enumerate(results[0]["summary"]):
        line(name, [r["summary"][i][1] for r in results])
    for g, (group, _, rows) in enumerate(results[0]["groups"]):
        print()
        line(f"{group} (total)", [r["groups"][g][1] for r in results])
        for i, (name, _) in enumerate(rows):
            line(f"  {name}", [r["groups"][g][2][i][1] for r in results])
    print()
    for i, (name, _) in enumerate(results[0]["totals"]):
        line(f"{name} (same moment)", [r["totals"][i][1] for r in results])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str.lower, choices=MODELS, default="tresnet")
    parser.add_argument("--rank", type=int, default=2, help="MemFLoRA's rank")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--use-adabn", action="store_true", help="calibrate BN on one batch first"
    )
    parser.add_argument(
        "--bitpack-backend",
        choices=("auto", "cuda", "torch"),
        default=os.environ.get("MEMFLORA_BITPACK_BACKEND", "auto"),
        help="auto builds CUDA packing if available; cuda requires it; torch disables it",
    )
    parser.add_argument(
        "--empty-cache-after-calibration",
        action="store_true",
        help="release unused CUDA cache after calibration; does not free live tensors",
    )
    parser.add_argument(
        "--trace-stacks",
        action="store_true",
        help="record allocation stacks (adds profiler overhead)",
    )
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="save each CUDA trace for PyTorch memory_viz",
    )
    parser.add_argument("--reverse-order", action="store_true", help="measure Full FT first")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, help="default: runs/jetson_memory_*")
    parser.add_argument("--cpu", action="store_true", help="measure on the CPU")
    parser.add_argument("--child", help=argparse.SUPPRESS)  # the method to measure
    cli = parser.parse_args()
    if cli.rank <= 0 or cli.batch <= 0:
        parser.error("--rank and --batch must be positive")
    os.environ["MEMFLORA_BITPACK_BACKEND"] = cli.bitpack_backend
    sys.stdout.reconfigure(line_buffering=True)
    if cli.child:
        measure(cli, cli.child)
        return 0

    out = cli.out or ROOT / "runs" / f"jetson_memory_{cli.model}_r{cli.rank}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "title": f"{MODELS[cli.model][0]}: MemFLoRA (rank {cli.rank}) vs full"
        f" fine-tuning, batch {cli.batch}, AdaBN {'on' if cli.use_adabn else 'off'}",
        "t0": time.time(),
        "runs": list(reversed(RUNS)) if cli.reverse_order else list(RUNS),
        "samples": [],
        "stages": [],
        "results": {},
        "done": False,
    }
    (out / "index.html").write_text(PAGE.replace("__DATA__", "null"))

    def save() -> None:
        with LOCK:
            text = json.dumps(report)
            # The periodic saver and shutdown path share this temporary file.
            (out / "data.json.tmp").write_text(text)
            os.replace(out / "data.json.tmp", out / "data.json")

    def keep_saving() -> None:
        while True:
            save()
            time.sleep(1)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(out))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", cli.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monitor = Monitor(report["samples"])
    monitor.start()
    threading.Thread(target=keep_saving, daemon=True).start()

    url = f"http://127.0.0.1:{cli.port}"
    print(f"live report: {url}")
    if os.environ.get("DISPLAY") or sys.platform == "darwin":
        webbrowser.open(url)
    else:
        tunnel = f"ssh -L {cli.port}:localhost:{cli.port} <jetson>"
        print(f"over SSH, run this on your laptop first: {tunnel}")

    for label in report["runs"]:
        time.sleep(3)  # a short gap between the runs on the charts
        run_child(label, RUNS[label], report)
    with LOCK:
        report["done"] = True
    print()
    print_summary(report)
    print("\nstill sampling; press Ctrl+C to stop and save report.html")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    if monitor.process:
        monitor.process.terminate()
    save()
    with LOCK:
        page = PAGE.replace("__DATA__", json.dumps(report).replace("</", "<\\/"))
    (out / "report.html").write_text(page)
    print(f"\nsaved {out / 'report.html'}")
    return 0


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MemFLoRA memory</title>
<style>
:root { --bg: #f9f9f7; --card: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --mute: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --line: rgba(11,11,11,.1);
  --s1: #2a78d6; --s2: #eb6834; --warn: #b3261e;
  --k0: #b9b8b0; --k1: #7c5cc4; --k2: #2f9e6e; --k3: #d4a72c; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d0d0d; --card: #1a1a19;
  --ink: #fff; --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
  --line: rgba(255,255,255,.1); --s1: #3987e5; --s2: #d95926; --warn: #f2b8b5;
  --k0: #57564f; --k1: #9b80e0; --k2: #3fb983; --k3: #e2b84a; } }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, sans-serif; padding: 24px 16px; }
main { max-width: 1000px; margin: auto; }
h1 { font-size: 19px; margin: 0; } h2 { font-size: 15px; margin: 0 0 4px; }
.sub, .label { color: var(--ink2); font-size: 13px; font-weight: 400; }
.warn { color: var(--warn); font-weight: 600; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px; margin: 16px 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; margin-bottom: 10px; }
.pair { display: flex; justify-content: space-between; gap: 8px; font-size: 16px;
  font-weight: 600; font-variant-numeric: tabular-nums; }
.note { color: var(--ink2); font-size: 12px; font-weight: 400; line-height: 1.35; }
.card > .note { margin-top: 8px; }
.section { margin-top: 22px; }
.equation { padding: 8px 0; font-weight: 600; }
.table-scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
td { padding: 3px 8px; vertical-align: top; }
tr.group td { font-weight: 600; padding-top: 12px; border-bottom: 1px solid var(--line); }
tr.total td { padding-top: 10px; padding-bottom: 10px;
  font-weight: 700; border-top: 2px solid var(--axis); }
td.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums;
  width: 110px; }
td.bar { width: 26%; } .fill { height: 6px; margin: 2px 0; border-radius: 0 3px 3px 0; }
.key { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 6px; }
.legend { margin-left: 14px; }
svg { display: block; width: 100%; }
.tick { fill: var(--mute); font-size: 11px; font-variant-numeric: tabular-nums; }
.stage { fill: var(--ink2); font-size: 11px; }
.run { fill: var(--ink); font-size: 12px; font-weight: 600; }
#tip { position: fixed; pointer-events: none; background: var(--card);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px; font-size: 12px; }
</style>
<main>
  <h1 id="title">MemFLoRA memory</h1>
  <div class="sub" id="status">waiting for data…</div>
  <div class="note">One training step after warm-up, using random inputs and weights.
    GPU figures cover PyTorch-managed memory. All values are in MB.</div>
  <div id="tiles"></div>
  <div class="card" id="breakdown" hidden>
    <h2>What adds up at peak GPU use</h2>
    <div class="note">All rows below refer to the same moment for each method:
      when its GPU memory use is highest. Each A, B or C subtotal adds the rows below it.</div>
    <div class="equation">A + B = used &nbsp; &middot; &nbsp; A + B + C = reserved</div>
    <div class="table-scroll"><table id="table"></table></div>
    <div class="note">This is the reserved memory at that moment.
      The peak reserved during the whole step can be higher.</div>
  </div>
  <div id="steps"></div>
  <div id="charts"></div>
  <details class="card"><summary>Run configuration and build information</summary>
    <pre id="metadata" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre>
  </details>
  <div class="sub">Raw samples: <a href="data.json">data.json</a></div>
</main>
<div id="tip" hidden></div>
<script>
const EMBEDDED = __DATA__;
const $ = (id) => document.getElementById(id);
const MB = (b) => (b == null ? "n/a" : (b / 1e6).toFixed(2) + " MB");
const NS = "http://www.w3.org/2000/svg";
const CHARTS = [
  ["Whole-device RAM in use", "MB", [["RAM in use", "device_ram"]]],
  ["GPU load", "%", [["GPU", "gpu"]], 100],
  ["CPU load", "%", [["Average of all cores", "cpu"], ["Busiest core", "cpu_max"]], 100],
  ["Input power", "W", [["VDD_IN", "power"]]],
];
const KINDS = ["Already held before the step", "Saved for backward", "Weight gradients", "Working buffers"];
const SECTIONS = [
  ["At peak GPU use", "These four boxes describe the same moment for each method.", [
    "Peak GPU memory used", "Training state at that moment",
    "Other working memory at that moment", "GPU memory reserved at that moment",
  ]],
  ["Training footprint", "These values can occur at different moments; do not add them together.", [
    "Peak GPU memory reserved", "Peak training state", "Saved for backward",
  ]],
  ["Whole device", "A separate view of the device, including the OS and other programs.", [
    "Whole-device RAM in use",
  ]],
];
const BOX_NOTES = {
  "Peak GPU memory used": "The most memory occupied by PyTorch tensors and working buffers. " +
    "Training state (A) + other working memory (B). Unused cache is excluded.",
  "Training state at that moment": "A: model weights, optimizer data, weight gradients, " +
    "and saved activations and masks. This is part of the used total beside it.",
  "Other working memory at that moment": "B: temporary buffers, inputs, a saved model copy " +
    "and other allocations. Added to A, this gives peak GPU memory used.",
  "GPU memory reserved at that moment": "Memory PyTorch holds from the GPU: used memory " +
    "(A + B), plus cache and allocation overhead (C). Some is available for reuse.",
  "Peak GPU memory reserved": "The most memory PyTorch held from the GPU at any point " +
    "during the step, including its cache. It can peak at a different moment than used memory.",
  "Peak training state": "The largest combined A total during the step. This measures " +
    "training state only; running the step also needs working memory.",
  "Saved for backward": "Activations and masks kept after the forward pass to calculate " +
    "gradients. This is part of training state, not an extra amount to add to it.",
  "Whole-device RAM in use": "Estimated RAM use after the training step finishes. " +
    "Includes the OS and other programs, so it depends on what else is running. " +
    "This is not a peak or a per-method cost; do not add it to the GPU figures.",
};
const LABELS = {
  "Model state": "Model weights and buffers",
  "Optimizer state": "Optimizer data",
  "Parameter gradients": "Weight gradients",
  "Saved activations": "Saved activations and small constants",
  "Saved ReLU bitmasks": "Saved activation masks",
  "Temporaries and workspace": "Temporary working buffers",
  "Input batch": "Input data",
  "Best-checkpoint copy": "Saved model copy",
  "Other allocations kept between steps": "Other allocations held before the step",
  "Allocator cache and rounding": "Cache and allocation overhead",
};
const NOTES = {  // shown under each block and row of the table
  "Model state": "All model weights, including frozen weights, and stored model statistics.",
  "Optimizer state": "Information Adam keeps to update trainable weights.",
  "Parameter gradients": "Gradients calculated so far, used to update trainable weights.",
  "Saved activations": "Intermediate values still kept for calculating gradients.",
  "Saved ReLU bitmasks": "Packed ReLU/ReLU6 gates. Zero can mean they are not created yet " +
    "or have already been released. Full FT does not use MemFLoRA's packed masks.",
  "Temporaries and workspace": "Short-lived buffers used while calculating outputs and gradients.",
  "Input batch": "The current batch of inputs and labels.",
  "Best-checkpoint copy": "A second copy of the weights, kept to restore the best training step.",
  "Other allocations kept between steps": "Other memory already held before this step; " +
    "its exact owners have not been identified.",
  "Allocator cache and rounding": "Reserved memory beyond the live allocations: " +
    "reusable cache, unused space inside allocated blocks, and memory awaiting release.",
};
let data = null, hoverT = null;

function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text != null) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function keyed(color, text, tag = "span", cls) {  // text behind a colour key
  const n = node(tag, null, cls), key = node("span", null, "key");
  key.style.background = color;
  n.append(key, text);
  return n;
}
function svgNode(parent, tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  parent.append(n);
  return n;
}
function niceTop(high) {  // a round axis maximum and its step
  const raw = high / 5, magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((v) => v >= raw);
  return [Math.ceil(high / step) * step, step];
}
function yAxis(svg, top, step, y, left, right) {
  for (let v = 0; v <= top + 1e-9; v += step) {
    svgNode(svg, "line", { x1: left, x2: right, y1: y(v), y2: y(v),
      stroke: v ? "var(--grid)" : "var(--axis)" });
    svgNode(svg, "text", { x: left - 6, y: y(v) + 4, "text-anchor": "end", class: "tick" })
      .textContent = Math.round(v);
  }
}
function stageAt(t) {  // a stage is logged when it ends, so it spans from the previous one
  const s = data.stages.find((row) => row.t >= t);
  return !s || s.name === "start" ? "idle" : `${s.run} · ${s.name}`;
}

function render() {
  const d = data, results = d.runs.map((run) => d.results[run]);
  const first = results.find(Boolean);
  const running = d.stages.length ? d.stages[d.stages.length - 1].run : "";
  $("title").textContent = d.title;
  $("status").replaceChildren((d.done ? "Measurement finished" : `Measuring ${running}…`) +
    (EMBEDDED ? "" : " · live, refreshes every second") + (first ? ` · ${first.device}` : ""));
  const checks = results.filter(Boolean).map((r) => r.verified);
  if (checks.includes(false)) {
    $("status").append(node("span", " · memory totals failed the cross-check; " +
      "treat these figures as unreliable", "warn"));
  } else if (checks.length && checks.every((v) => v === true)) {
    $("status").append(" · peak totals checked against PyTorch");
  }
  if (first) renderTables(d, results, first);
  $("metadata").textContent = JSON.stringify(Object.fromEntries(
    d.runs.map((run) => [run, d.results[run]?.metadata || null])), null, 2);
  $("steps").replaceChildren();
  drawSteps(d);
  $("charts").replaceChildren();
  for (const chart of CHARTS) drawChart(...chart);
}

function renderTables(d, results, first) {
  const cell = (r, value) => (r ? MB(value) : d.done ? "failed" : "…");
  const color = (k) => `var(--s${k + 1})`;
  const summaries = results.map((r) => new Map(r?.summary || []));
  $("tiles").replaceChildren(...SECTIONS.map(([heading, explanation, labels]) => {
    const section = node("section", null, "section");
    section.append(node("h2", heading), node("div", explanation, "note"));
    const grid = node("div", null, "tiles");
    labels.forEach((label) => {
      const card = node("div", null, "card");
      card.append(node("div", label, "label"));
      d.runs.forEach((run, k) => {
        const pair = node("div", null, "pair");
        pair.append(keyed(color(k), run), node("span", cell(results[k], summaries[k].get(label))));
        card.append(pair);
        const m = results[k]?.metrics;
        if (m && label === "Saved for backward") {
          card.append(node("div", `${MB(m.forward_saved_activations)} activations + ` +
            `${MB(m.forward_saved_bitmasks)} masks`, "note"));
        }
        if (m && label === "Whole-device RAM in use") {
          card.append(node("div", `Before the run: ${MB(m.device_ram_before)} · ` +
            `Usable device RAM: ${MB(m.device_ram_total)}`, "note"));
        }
      });
      card.append(node("div", BOX_NOTES[label], "note"));
      grid.append(card);
    });
    section.append(grid);
    return section;
  }));
  const head = node("tr", null, "group");
  head.append(node("td"), ...d.runs.map((run, k) => keyed(color(k), run, "td", "num")),
    node("td"));
  const rows = [head];
  const named = (text, suffix) => {  // a name cell with its explanation underneath
    const n = node("td", LABELS[text] || text);
    if (suffix) n.append(node("span", suffix, "label"));
    if (NOTES[text]) n.append(node("div", NOTES[text], "note"));
    return n;
  };
  const addTotal = (i) => {
    const [label] = first.totals[i];
    const total = node("tr", null, "total");
    total.append(named(label),
      ...results.map((r) => node("td", cell(r, r?.totals[i][1]), "num")), node("td"));
    rows.push(total);
  };
  first.groups.forEach(([group, , items], g) => {
    const total = node("tr", null, "group");
    total.append(named(group, " · total"),
      ...results.map((r) => node("td", cell(r, r?.groups[g][1]), "num")), node("td"));
    rows.push(total);
    const values = items.map((_, i) => results.map((r) => r?.groups[g][2][i][1]));
    const largest = Math.max(1, ...values.flat().map((v) => v || 0));  // bars scale per block
    items.forEach(([label], i) => {
      const row = node("tr"), bar = node("td", null, "bar");
      values[i].forEach((value, k) => {
        const fill = node("div", null, "fill");
        fill.style.width = value > 0 ? `max(1px, ${(100 * value) / largest}%)` : "0";
        fill.style.background = color(k);
        bar.append(fill);
      });
      row.append(named(label),
        ...values[i].map((value, k) => node("td", cell(results[k], value), "num")), bar);
      rows.push(row);
    });
    if (g === 1) addTotal(0);
    if (g === 2) addTotal(1);
  });
  $("breakdown").hidden = false;
  $("table").replaceChildren(...rows);
}

function drawSteps(d) {  // the allocator during each run's measured step, one scale
  const runs = d.runs.filter((run) => d.results[run]?.timeline);
  if (!runs.length) return;
  const high = Math.max(...runs.flatMap((run) =>
    d.results[run].timeline.points.map((p) => Math.max(p[5], p[1] + p[2] + p[3] + p[4]))));
  const [top, step] = niceTop(Math.max(1, high / 1e6) * 1.05);
  for (const run of runs) {
    const { points, ends, peak, events } = d.results[run].timeline;
    const card = node("div", null, "card");
    const heading = node("h2", `${run}: GPU memory during the training step (MB)`);
    KINDS.forEach((name, i) => heading.append(keyed(`var(--k${i})`, name, "span", "label legend")));
    heading.append(keyed("var(--ink2)", "Held by PyTorch (reserved)", "span", "label legend"));
    card.append(heading);
    card.append(node("div", "The colored areas add up to memory used. The dashed line " +
      "also includes cache and allocation overhead. Left to right follows memory operations, not elapsed time.", "note"));
    $("steps").append(card);
    const W = card.clientWidth - 28, H = 170, L = 46, R = 10, T = 18, B = 8;
    const svg = svgNode(card, "svg", { viewBox: `0 0 ${W} ${T + H + B}` });
    const x = (k) => L + (k / Math.max(1, events)) * (W - L - R);
    const y = (v) => T + H - (v / 1e6 / top) * H;
    yAxis(svg, top, step, (v) => T + H - (v / top) * H, L, W - R);
    let below = points.map(() => 0);
    KINDS.forEach((_, i) => {
      const above = points.map((p, j) => below[j] + p[i + 1]);
      const edge = points.map((p, j) => `${x(p[0]).toFixed(1)},${y(above[j]).toFixed(1)}`);
      const back = points.map((p, j) => `${x(p[0]).toFixed(1)},${y(below[j]).toFixed(1)}`).reverse();
      svgNode(svg, "polygon", { points: [...edge, ...back].join(" "), fill: `var(--k${i})` });
      below = above;
    });
    svgNode(svg, "polyline", { points: points.map((p) => `${x(p[0]).toFixed(1)},${y(p[5]).toFixed(1)}`).join(" "),
      fill: "none", stroke: "var(--ink2)", "stroke-width": 1.5, "stroke-dasharray": "4 3" });
    const bounds = [0, ...ends, events];
    ["forward", "backward", "optimizer"].forEach((name, i) => {
      if (i) svgNode(svg, "line", { x1: x(bounds[i]), x2: x(bounds[i]), y1: T, y2: T + H, stroke: "var(--axis)" });
      if (x(bounds[i + 1]) - x(bounds[i]) > name.length * 6.5) {
        svgNode(svg, "text", { x: (x(bounds[i]) + x(bounds[i + 1])) / 2, y: 12,
          "text-anchor": "middle", class: "stage" }).textContent = name;
      }
    });
    svgNode(svg, "line", { x1: x(peak), x2: x(peak), y1: T, y2: T + H,
      stroke: "var(--ink)", "stroke-dasharray": "2 2" });
    svgNode(svg, "text", { x: x(peak) + 4, y: T + 10, class: "run" }).textContent = "peak";
  }
}

function drawChart(title, unit, series, yMax) {
  const d = data, samples = d.samples;
  series = series.filter(([, key]) => samples.some((s) => s[key] != null));
  if (!series.length) return;
  const labelled = title === CHARTS[0][0];  // run and stage names go on the first chart
  const card = node("div", null, "card"), heading = node("h2", `${title} (${unit})`);
  if (series.length > 1) {
    series.forEach(([name], i) =>
      heading.append(keyed(`var(--s${i + 1})`, name, "span", "label legend")));
  }
  card.append(heading);
  if (title === CHARTS[0][0]) {
    card.append(node("div", "Estimated RAM use for the whole device, including the OS and " +
      "other programs. Sampled about twice per second; brief peaks may be missed.", "note"));
  }
  $("charts").append(card);
  const W = card.clientWidth - 28, H = 150, L = 46, R = 10, T = labelled ? 34 : 12, B = 22;
  const svg = svgNode(card, "svg", { viewBox: `0 0 ${W} ${T + H + B}` });
  // Once the run is done, keep the axis on the run instead of the idle tail.
  const last = samples.length ? samples[samples.length - 1].t - d.t0 : 1;
  const lastStage = d.stages.length ? d.stages[d.stages.length - 1].t - d.t0 : 0;
  const tEnd = Math.max(1, d.done ? Math.min(last, lastStage + 10) : last);
  const values = samples.filter((s) => s.t - d.t0 <= tEnd)
    .flatMap((s) => series.map(([, k]) => s[k])).filter((v) => v != null);
  const [niceHigh, step] = niceTop(yMax || Math.max(1, ...values) * 1.05);
  const top = yMax || niceHigh;
  const x = (t) => L + (t / tEnd) * (W - L - R);
  const y = (v) => T + H - (v / top) * H;
  yAxis(svg, top, yMax ? yMax / 5 : step, y, L, W - R);
  for (let i = 0; i <= 5; i++) {
    svgNode(svg, "text", { x: x((tEnd * i) / 5), y: T + H + 16,
      "text-anchor": i === 5 ? "end" : "middle", class: "tick" })
      .textContent = `${Math.round((tEnd * i) / 5)} s`;
  }
  let previous = null;
  for (const s of d.stages) {
    const sx = x(s.t - d.t0);
    svgNode(svg, "line", { x1: sx, x2: sx, y1: T, y2: T + H, stroke: "var(--axis)" });
    if (labelled && s.name === "start") {
      svgNode(svg, "text", { x: sx + 4, y: 12, class: "run" }).textContent = s.run;
    } else if (labelled && sx - previous > s.name.length * 6.5 + 8) {
      svgNode(svg, "text", { x: previous + 4, y: 28, class: "stage" }).textContent = s.name;
    }
    previous = sx;
  }
  series.forEach(([, key], i) => {  // the line breaks where there is no sample
    let path = "", pen = "M";
    for (const s of samples.filter((s) => s.t - d.t0 <= tEnd)) {
      if (s[key] == null) { pen = "M"; continue; }
      path += `${pen}${x(s.t - d.t0).toFixed(1)},${y(s[key]).toFixed(1)} `;
      pen = "L";
    }
    svgNode(svg, "path", { d: path, fill: "none", stroke: `var(--s${i + 1})`,
      "stroke-width": 2, "stroke-linejoin": "round" });
  });
  const cross = svgNode(svg, "line", { y1: T, y2: T + H, stroke: "var(--mute)", visibility: "hidden" });
  const show = (t, event) => {
    const s = samples.reduce((a, b) =>
      Math.abs(b.t - d.t0 - t) < Math.abs(a.t - d.t0 - t) ? b : a);
    cross.setAttribute("x1", x(s.t - d.t0));
    cross.setAttribute("x2", x(s.t - d.t0));
    cross.setAttribute("visibility", "visible");
    if (!event) return;
    const tip = $("tip");
    tip.replaceChildren(node("div", `${(s.t - d.t0).toFixed(1)} s · ${stageAt(s.t)}`, "label"),
      ...series.map(([name, key]) => node("div", `${s[key] == null ? "–" : +s[key].toFixed(1)} ${unit}  ${name}`)));
    tip.hidden = false;
    tip.style.left = `${Math.min(event.clientX + 14, innerWidth - 220)}px`;
    tip.style.top = `${event.clientY + 14}px`;
  };
  svg.addEventListener("pointermove", (event) => {
    const r = svg.getBoundingClientRect();
    hoverT = (((event.clientX - r.left) * (W / r.width) - L) / (W - L - R)) * tEnd;
    show(hoverT, event);
  });
  svg.addEventListener("pointerleave", () => { hoverT = null; $("tip").hidden = true; });
  if (hoverT != null && samples.length) show(hoverT);
}

async function poll() {
  try {
    const response = await fetch("data.json", { cache: "no-store" });
    data = await response.json();
    render();
  } catch (error) { /* the run has not written data yet, or the server stopped */ }
  setTimeout(poll, 1000);
}
if (EMBEDDED) { data = EMBEDDED; render(); } else poll();
addEventListener("resize", () => data && render());
</script>
"""


if __name__ == "__main__":
    sys.exit(main())
