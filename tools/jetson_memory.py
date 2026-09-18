"""Profile the memory of MemFLoRA and full fine-tuning on a Jetson.

    python tools/jetson_memory.py --model tresnet --rank 2
    python tools/jetson_memory.py --model mobilenetv2 --rank 2

Each method runs in its own fresh process, so each pays for its own libraries.
Inside PyTorch's allocator, the measured training step is recorded allocation by
allocation, and every block alive at the step's peak is attributed to what owns
it. Outside the allocator, each phase is charged the used RAM it added; CPU and GPU
share that RAM on a Jetson. Keep the Jetson otherwise idle while measuring.

The report is served at http://127.0.0.1:8000 and refreshes itself every second.
Over SSH, open a tunnel first: ssh -L 8000:localhost:8000 <jetson>. Press Ctrl+C
when done; a standalone copy is saved as report.html. Numbers are in MB (10^6 B).

The step uses random weights and random windows of the Opportunity shape (97
channels x 60 steps); its memory depends only on tensor shapes.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
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
    " --adabn-calibration-mode ema_no_reset --adabn-calib-batches 1"
    " --bnpa-bottleneck-bn on --adabn-stat-source target --train_mode_adaBN off"
)
# Rows of the peak breakdown; the first block follows Table 3 of the paper.
PEAK_BLOCKS = [
    [
        "Training state at the step's peak",
        [
            "Model state",
            "Optimizer state",
            "Parameter gradients",
            "Saved activations",
            "Saved ReLU bitmasks",
        ],
    ],
    [
        "Other PyTorch memory at the peak",
        [
            "Temporaries and workspace",
            "Input batch",
            "Best-checkpoint copy",
            "Other allocations kept between steps",
            "Allocator cache and rounding",
        ],
    ],
]
MARK = "@jetson_memory "  # marks the lines a measuring child sends back
LOCK = threading.Lock()


def meminfo() -> dict:
    """Used RAM (MemTotal - MemAvailable) and, on a Jetson, GPU memory (NvMap)."""
    with open("/proc/meminfo") as f:
        info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    used = info["MemTotal"] - info["MemAvailable"]
    return {"used": used, "nvmap": info.get("NvMapMemUsed")}


def resident() -> dict:
    """This process's resident memory: private (anon) and mapped from files."""
    with open("/proc/self/status") as f:
        info = {
            line.split(":")[0]: int(line.split()[1]) * 1024
            for line in f
            if line.startswith("Rss")
        }
    return {"anon": info["RssAnon"] + info["RssShmem"], "file": info["RssFile"]}


class Monitor(threading.Thread):
    """Samples used RAM, GPU memory, GPU and CPU load and power every 0.5 s."""

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
        memory = meminfo()  # the counters the table uses, not tegrastats' RAM
        sample["ram"] = round(memory["used"] / 1e6, 1)
        if memory["nvmap"] is not None:
            sample["gpu_mem"] = round(memory["nvmap"] / 1e6, 1)
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


def replay(trace: list, before: dict, reserved: int) -> tuple[dict, int]:
    """Replay a step's allocator trace up to its peak of requested bytes.

    `before` maps each block allocated when the trace began to its requested size.
    Returns the blocks live at the peak, as address -> (index of the block's alloc
    event, or -1 if it predates the trace; size), and the bytes reserved then.
    """

    def run(events):
        live = {address: (-1, size) for address, size in before.items()}
        held, total = reserved, sum(before.values())
        peak, peak_at = total, 0
        for i, event in enumerate(events):
            action, address = event["action"], event.get("addr")
            if action == "alloc":
                live[address] = (i, event["size"])
                total += event["size"]
            elif action in ("free_requested", "free_completed") and address in live:
                total -= live.pop(address)[1]
            elif action in ("segment_alloc", "segment_map"):
                held += event["size"]
            elif action in ("segment_free", "segment_unmap"):
                held -= event["size"]
            if total > peak:
                peak, peak_at = total, i + 1
        return live, held, peak_at

    live, held, _ = run(trace[: run(trace)[2]])
    return live, held


def attribute(trace, before, reserved, forward_allocs, saved, grads, owners):
    """Attribute every block live at the step's peak to a row of the table.

    `forward_allocs` is how many allocations the forward pass made, `saved` maps
    the storage of each tensor autograd saved to its row, `grads` holds the
    gradients' storages and `owners` maps tensors kept between steps to their row.
    Returns the rows, the bytes reserved and requested at the peak, and the bytes
    the forward pass saved.
    """
    live, held = replay(trace, before, reserved)
    allocs = [(i, e) for i, e in enumerate(trace) if e["action"] == "alloc"]
    # A saved tensor is the last block allocated at its address in the forward
    # pass; a gradient is the last block allocated at its address at all.
    forward = {event["addr"]: i for i, event in allocs[:forward_allocs]}
    saved_at = {forward[a]: row for a, row in saved.items() if a in forward}
    last = {event["addr"]: i for i, event in allocs}
    grads_at = {last[a] for a in grads if a in last}
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
    peak = sum(size for _, size in live.values())
    rows["Allocator cache and rounding"] = held - peak
    return rows, held, peak, sum(trace[i]["size"] for i in saved_at)


def emit(**message) -> None:
    print(MARK + json.dumps(message), flush=True)


def measure(cli, method: str) -> None:
    """Measure one method. Runs in a child process and emits what it finds."""
    torch = None
    device = None

    def stage(name: str) -> dict:
        row = {"name": name, "t": time.time(), **meminfo(), **resident()}
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
            row["reserved"] = torch.cuda.memory_reserved()
        emit(stage=row)
        return row

    def outside(a: dict, b: dict) -> int:
        """Used RAM a phase added, minus what PyTorch's allocator reserved in it."""
        return b["used"] - a["used"] - b.get("reserved", 0) + a.get("reserved", 0)

    s_start = stage("start")
    import torch

    s_torch = stage("import torch")
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

    s_project = stage("import project")
    if cli.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.zeros(1, device=device).add_(1).item()  # fails early on a wrong build
    torch.manual_seed(1)
    s_context = stage("cuda context")

    argv = [*COMMON_ARGS.split(), *MODELS[cli.model][1].split(), "--method", method]
    args = parse_args(argv + ["--rank", str(cli.rank)])
    apply_dataset_defaults(args)
    args = argparse.Namespace(
        **{
            **vars(args),
            "rank": cli.rank,
            "method": method,
            "batch_size": cli.batch,
            "adabn_calib_batches": 1,
            "adapt_lr": 1e-3,
        }
    )
    windows = torch.randn(cli.batch, args.expected_input_channels, args.window_size)
    labels = torch.randint(args.expected_num_classes, (cli.batch,))
    loader = DataLoader(TensorDataset(windows, labels), batch_size=cli.batch)

    model = build_backbone(args).to(device)
    configure_method(model, args)
    model.to(device)
    s_model = stage("model")
    calibrate_minimal_adabn_if_needed(model, method, loader, args, device)
    stage("calibration")

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
        if tensor.numel():
            storage = tensor.untyped_storage()
            bits = tensor.dtype in (torch.uint8, torch.bool)
            row = "Saved ReLU bitmasks" if bits else "Saved activations"
            saved[storage.data_ptr()] = (row, storage.nbytes())
        return tensor

    start_step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.memory._record_memory_history("all", context=None)
        before = {
            block["address"]: block["requested_size"]
            for segment in torch.cuda.memory._snapshot()["segments"]
            for block in segment["blocks"]
            if block["state"] == "active_allocated"
        }
        reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = loss_fn(model(x), y)
    if cuda:  # how many allocations the forward pass made
        count = torch.cuda.memory_stats()["allocation.all.allocated"]
        allocations = count - allocations
    loss.backward()
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
        traces = torch.cuda.memory._snapshot()["device_traces"]
        trace = traces[torch.cuda.current_device()]
        torch.cuda.memory._record_memory_history(None)
        kinds = {address: row for address, (row, _) in saved.items()}
        found, held, peak, saved_total = attribute(
            trace, before, reserved, allocations, kinds, grads, owners
        )
        rows.update(found)
        verified = peak == torch.cuda.memory_stats()["requested_bytes.all.peak"]
    else:  # no allocator on the CPU: count every saved tensor and gradient
        for address, row in owners.items():
            rows[row] += sizes[address]
        for address, (row, size) in saved.items():
            if address not in owners:
                rows[row] += size
        rows["Parameter gradients"] = sum(
            p.grad.nbytes for p in params if p.grad is not None
        )
        saved_total = rows["Saved activations"] + rows["Saved ReLU bitmasks"]
        unmeasured = {"Temporaries and workspace", "Allocator cache and rounding"}
        unmeasured.add("Other allocations kept between steps")
        held = verified = None

    groups = [
        [block, [[row, None if row in unmeasured else rows[row]] for row in names]]
        for block, names in PEAK_BLOCKS
    ]
    groups += [
        [
            "Outside PyTorch's allocator: added used RAM",
            [
                ["Python interpreter", s_start["used"] - cli.baseline["used"]],
                ["PyTorch import", outside(s_start, s_torch)],
                ["Project imports", outside(s_torch, s_project)],
                ["CUDA context and driver", outside(s_project, s_context)],
                ["Model and data setup", outside(s_context, s_model)],
                ["cuDNN and cuBLAS at first use", outside(s_model, s_done)],
            ],
        ],
        [
            "Reclaimable, not counted above",
            [["Library files mapped into the process", s_done["file"]]],
        ],
    ]
    for group in groups:  # each block carries its total
        group.insert(1, sum(value for _, value in group[1] if value is not None))
    nvmap = [s_done["nvmap"], cli.baseline["nvmap"]]
    summary = [
        ["Training state at the peak", groups[0][1]],
        ["Saved activations, end of forward pass", saved_total],
        ["Optimizer state", rows["Optimizer state"]],
        ["PyTorch memory at the peak", held],
        ["Whole run, added used RAM", s_done["used"] - cli.baseline["used"]],
        ["GPU memory added (NvMap)", None if None in nvmap else nvmap[0] - nvmap[1]],
    ]
    emit(device=str(device), verified=verified, groups=groups, summary=summary)


def run_child(label: str, method: str, report: dict) -> None:
    """Measure one method in a fresh process, so it pays for its own libraries."""
    baseline = json.dumps(meminfo())  # the RAM in use before the process starts
    command = [sys.executable, __file__, *sys.argv[1:], "--child", method]
    child = subprocess.Popen(
        command + ["--baseline", baseline], stdout=subprocess.PIPE, text=True
    )
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
    if child.wait():
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=str.lower, choices=MODELS, default="tresnet")
    parser.add_argument("--rank", type=int, default=2, help="MemFLoRA's rank")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, help="default: runs/jetson_memory_*")
    parser.add_argument("--cpu", action="store_true", help="measure on the CPU")
    parser.add_argument("--child", help=argparse.SUPPRESS)  # the method to measure
    parser.add_argument("--baseline", type=json.loads, help=argparse.SUPPRESS)
    cli = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if cli.child:
        measure(cli, cli.child)
        return 0

    out = cli.out or ROOT / "runs" / f"jetson_memory_{cli.model}_r{cli.rank}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "title": f"{MODELS[cli.model][0]}: MemFLoRA (rank {cli.rank}) vs full"
        f" fine-tuning, batch {cli.batch}",
        "t0": time.time(),
        "runs": list(RUNS),
        "samples": [],
        "stages": [],
        "results": {},
        "done": False,
    }
    (out / "index.html").write_text(PAGE.replace("__DATA__", "null"))

    def save() -> None:
        with LOCK:
            text = json.dumps(report)
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

    for label, method in RUNS.items():
        time.sleep(3)  # idle, so each run starts from a settled baseline
        run_child(label, method, report)
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
  --s1: #2a78d6; --s2: #eb6834; --warn: #b3261e; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d0d0d; --card: #1a1a19;
  --ink: #fff; --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
  --line: rgba(255,255,255,.1); --s1: #3987e5; --s2: #d95926; --warn: #f2b8b5; } }
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
#guide ul { margin: 6px 0 0; padding-left: 18px; color: var(--ink2); }
#guide li { margin: 4px 0; } #guide b { color: var(--ink); }
table { width: 100%; border-collapse: collapse; }
td { padding: 3px 8px; vertical-align: top; }
tr.group td { font-weight: 600; padding-top: 12px; border-bottom: 1px solid var(--line); }
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
  <div class="tiles" id="tiles"></div>
  <div class="card" id="guide">
    <h2>How this is measured</h2>
    <ul>
      <li><b>PyTorch memory at the step's peak.</b> One training step runs with
        PyTorch's allocator history on. Replaying it finds the moment the most memory
        was allocated, and every block alive then is attributed by its address: model
        state, optimizer state, gradients, the tensors autograd saved in this step's
        forward pass, temporaries allocated during the step, and blocks that existed
        before it. Sizes are what each tensor requested; the allocator's cache and
        rounding make up the rest of what it had reserved. The replayed peak is
        checked against PyTorch's own peak counter.</li>
      <li><b>Outside the allocator.</b> Each phase is charged the used RAM
        (MemTotal − MemAvailable) it added, minus what the allocator reserved during
        it: the interpreter, the libraries, the CUDA context, and the kernels and
        handles cuDNN and cuBLAS load at first use. On a Jetson the CPU and GPU share
        this RAM, so it includes GPU memory. The counter is system-wide, so the
        Jetson should be otherwise idle.</li>
      <li><b>Reclaimable memory.</b> Library files mapped into the process count
        in its RSS (what <code>top</code> shows), but they sit in the page cache,
        which the kernel can drop, so they are not in the totals.</li>
      <li><b>Grouping.</b> The first three tiles and the first block follow Table 3
        of the paper (peak training state, saved activations, optimizer state), but
        every number here is measured on the device.</li>
    </ul>
  </div>
  <div class="card" id="breakdown" hidden><table id="table"></table></div>
  <div id="charts"></div>
  <div class="sub">Raw samples: <a href="data.json">data.json</a></div>
</main>
<div id="tip" hidden></div>
<script>
const EMBEDDED = __DATA__;
const $ = (id) => document.getElementById(id);
const MB = (b) => (b == null ? "n/a" : (b / 1e6).toFixed(2) + " MB");
const NS = "http://www.w3.org/2000/svg";
const CHARTS = [
  ["Memory in use", "MB", [["Used RAM", "ram"], ["GPU memory (NvMap)", "gpu_mem"]]],
  ["GPU load", "%", [["GPU", "gpu"]], 100],
  ["CPU load", "%", [["Average of all cores", "cpu"], ["Busiest core", "cpu_max"]], 100],
  ["Input power", "W", [["VDD_IN", "power"]]],
];
const NOTES = {  // shown under each block and row of the table
  "Training state at the step's peak":
    "What training itself holds at the moment of the step's peak.",
  "Other PyTorch memory at the peak":
    "The rest of what PyTorch's allocator had reserved at that moment.",
  "Outside PyTorch's allocator: added used RAM":
    "Used RAM each phase added, minus the allocator's own growth.",
  "Reclaimable, not counted above":
    "Resident, but the kernel can drop it when memory runs short.",
  "Model state": "Every weight and buffer: frozen backbone, adapters, BatchNorm statistics.",
  "Optimizer state": "Adam's two moment buffers for each trainable weight.",
  "Parameter gradients": "Gradients already computed when the peak happens.",
  "Saved activations": "Tensors autograd saved in this step's forward pass and still " +
    "holds at the peak, each storage counted once.",
  "Saved ReLU bitmasks": "MemFLoRA's bit-packed ReLU signs still held at the peak.",
  "Temporaries and workspace": "Everything else allocated during the step and alive at " +
    "the peak: layer outputs, gradients flowing backward, cuDNN workspaces.",
  "Input batch": "The current mini-batch of windows and labels.",
  "Best-checkpoint copy": "The benchmark keeps a copy of the weights to restore its best " +
    "step.",
  "Other allocations kept between steps": "Allocated before the step by something other " +
    "than the tensors above, mostly library workspaces such as cuBLAS's.",
  "Allocator cache and rounding": "Reserved from the driver but not requested by any " +
    "tensor at the peak: cached free blocks, and each block's rounding.",
  "Python interpreter": "Python and this script's standard-library imports.",
  "PyTorch import": "Loading PyTorch and the CUDA libraries it links.",
  "Project imports": "This repository's modules, and torchvision for MobileNetV2.",
  "CUDA context and driver": "Created by the first GPU operation.",
  "Model and data setup": "Host memory used while building the model and the batch.",
  "cuDNN and cuBLAS at first use": "Kernels, library handles and caches, loaded from " +
    "calibration up to the measured step.",
  "Library files mapped into the process": "Code of Python, PyTorch, CUDA and cuDNN " +
    "mapped from disk (RssFile).",
};
let data = null, hoverT = null;

function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text != null) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function keyed(k, text, tag = "span", cls) {  // text behind the colour of run k
  const n = node(tag, null, cls), key = node("span", null, "key");
  key.style.background = `var(--s${k + 1})`;
  n.append(key, text);
  return n;
}
function svgNode(parent, tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  parent.append(n);
  return n;
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
    $("status").append(node("span", " · allocator trace does not match PyTorch's peak " +
      "counter, peak rows are unreliable", "warn"));
  } else if (checks.length && checks.every((v) => v === true)) {
    $("status").append(" · peaks checked against PyTorch's counter");
  }
  if (first) renderTables(d, results, first);
  $("charts").replaceChildren();
  for (const chart of CHARTS) drawChart(...chart);
}

function renderTables(d, results, first) {
  const cell = (r, value) => (r ? MB(value) : d.done ? "failed" : "…");
  $("tiles").replaceChildren(...first.summary.map(([label], i) => {
    const card = node("div", null, "card");
    card.append(node("div", label, "label"));
    d.runs.forEach((run, k) => {
      const pair = node("div", null, "pair");
      pair.append(keyed(k, run), node("span", cell(results[k], results[k]?.summary[i][1])));
      card.append(pair);
    });
    return card;
  }));
  const head = node("tr", null, "group");
  head.append(node("td"), ...d.runs.map((run, k) => keyed(k, run, "td", "num")), node("td"));
  const rows = [head];
  const named = (text, suffix) => {  // a name cell with its explanation underneath
    const n = node("td", text);
    if (suffix) n.append(node("span", suffix, "label"));
    if (NOTES[text]) n.append(node("div", NOTES[text], "note"));
    return n;
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
        fill.style.background = `var(--s${k + 1})`;
        bar.append(fill);
      });
      row.append(named(label),
        ...values[i].map((value, k) => node("td", cell(results[k], value), "num")), bar);
      rows.push(row);
    });
  });
  $("breakdown").hidden = false;
  $("table").replaceChildren(...rows);
}

function drawChart(title, unit, series, yMax) {
  const d = data, samples = d.samples;
  series = series.filter(([, key]) => samples.some((s) => s[key] != null));
  if (!series.length) return;
  const labelled = title === CHARTS[0][0];  // run and stage names go on the first chart
  const card = node("div", null, "card"), heading = node("h2", `${title} (${unit})`);
  if (series.length > 1) {
    series.forEach(([name], i) => heading.append(keyed(i, name, "span", "label legend")));
  }
  card.append(heading);
  $("charts").append(card);
  const W = card.clientWidth - 28, H = 150, L = 46, R = 10, T = labelled ? 34 : 12, B = 22;
  const svg = svgNode(card, "svg", { viewBox: `0 0 ${W} ${T + H + B}` });
  // Once the run is done, keep the axis on the run instead of the idle tail.
  const last = samples.length ? samples[samples.length - 1].t - d.t0 : 1;
  const lastStage = d.stages.length ? d.stages[d.stages.length - 1].t - d.t0 : 0;
  const tEnd = Math.max(1, d.done ? Math.min(last, lastStage + 10) : last);
  const values = samples.filter((s) => s.t - d.t0 <= tEnd)
    .flatMap((s) => series.map(([, k]) => s[k])).filter((v) => v != null);
  const high = yMax || Math.max(1, ...values) * 1.05;
  const raw = high / 5, magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((v) => v >= raw);
  const top = yMax || Math.ceil(high / step) * step;
  const x = (t) => L + (t / tEnd) * (W - L - R);
  const y = (v) => T + H - (v / top) * H;
  for (let v = 0; v <= top + 1e-9; v += step) {
    svgNode(svg, "line", { x1: L, x2: W - R, y1: y(v), y2: y(v),
      stroke: v ? "var(--grid)" : "var(--axis)" });
    svgNode(svg, "text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "tick" })
      .textContent = Math.round(v);
  }
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
  series.forEach(([, key], i) => {
    const points = samples.filter((s) => s[key] != null && s.t - d.t0 <= tEnd)
      .map((s) => `${x(s.t - d.t0).toFixed(1)},${y(s[key]).toFixed(1)}`);
    svgNode(svg, "polyline", { points: points.join(" "), fill: "none",
      stroke: `var(--s${i + 1})`, "stroke-width": 2, "stroke-linejoin": "round" });
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
