"""Measure where MemFLoRA's memory goes on a Jetson, and watch it live.

    python tools/jetson_memory.py --rank 2

The report is served at http://127.0.0.1:8000 and refreshes itself every second.
Over SSH, open a tunnel first: ssh -L 8000:localhost:8000 <jetson>. Press Ctrl+C
when done; a standalone copy is saved as report.html next to data.json.

The training step uses random windows of the Opportunity shape (97 channels x 60
steps). Step memory depends only on tensor shapes, so this matches real data.
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
METHOD = "bnpa_fa_postbn_bnr_off_scaled"  # MemFLoRA
LOCK = threading.Lock()
# The paper's Opportunity + T-ResNet settings.
BENCHMARK_ARGS = (
    "--dataset opportunity --backbone t_resnet_official --t-resnet-feature-maps 64"
    f" --method {METHOD} --window-size 60 --window-stride 30 --adapter-layers all"
    " --adabn-calibration-mode ema_no_reset --adabn-calib-batches 1"
    " --bnpa-bottleneck-bn on --adabn-stat-source target --train_mode_adaBN off"
).split()


def rss() -> int:
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line[:6] == "VmRSS:")


def system_used() -> int:
    with open("/proc/meminfo") as f:
        info = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    return info["MemTotal"] - info["MemAvailable"]


class Monitor(threading.Thread):
    """Samples RAM, CPU, GPU and power with tegrastats, or /proc elsewhere."""

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
                self.add(
                    {
                        "ram": system_used() // 2**20,
                        "cpu": sum(loads) / len(loads),
                        "cpu_max": max(loads),
                    }
                )

    def add(self, sample: dict) -> None:
        if sample.get("ram") is not None:
            with LOCK:
                self.samples.append({"t": time.time(), **sample})


def parse_tegrastats(line: str) -> dict:
    def number(pattern):
        match = re.search(pattern, line)
        return int(match.group(1)) if match else None

    cpu = re.search(r"CPU \[([^\]]*)\]", line)
    cores = [int(c) for c in re.findall(r"(\d+)%@", cpu.group(1))] if cpu else []
    power = number(r"VDD_IN (\d+)mW")
    return {
        "ram": number(r"RAM (\d+)/"),
        "gpu": number(r"GR3D_FREQ (\d+)%"),
        "cpu": sum(cores) / len(cores) if cores else None,
        "cpu_max": max(cores) if cores else None,
        "power": power / 1000 if power is not None else None,
    }


def cpu_times() -> list[list[int]]:
    with open("/proc/stat") as f:
        return [
            [int(v) for v in line.split()[1:8]]
            for line in f
            if line.startswith("cpu") and line[3].isdigit()
        ]


def measure(cli, report: dict) -> None:
    torch = None
    device = None

    def stage(name: str) -> dict:
        row = {"name": name, "t": time.time(), "rss": rss(), "used": system_used()}
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            row.update(
                alloc=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                device=total - free,
            )
        with LOCK:
            report["stages"].append(row)
        return row

    def outside_allocator(a: dict, b: dict) -> int | None:
        """Growth the PyTorch allocator did not hand out (context, kernels)."""
        if "reserved" not in b:
            return None
        grown = b["device"] - a["device"] if "device" in a else b["used"] - a["used"]
        return grown - (b["reserved"] - a.get("reserved", 0))

    def nbytes(tensors) -> int:
        seen = {}
        for x in tensors:
            if x is not None and x.numel():
                storage = x.untyped_storage()
                seen[storage.data_ptr()] = storage.nbytes()
        return sum(seen.values())

    s_start = stage("start")
    import torch

    s_torch = stage("import torch")
    sys.path.insert(0, str(ROOT))
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.benchmark_adabn import calibrate_minimal_adabn_if_needed
    from experiments.benchmark_cli import apply_dataset_defaults, parse_args
    from experiments.minimal_methods_benchmark import build_backbone, configure_method
    from src.train import freeze_bn_eval

    s_project = stage("import project")
    if cli.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.zeros(1, device=device).add_(1).item()  # fails early on a wrong build
    torch.manual_seed(1)
    s_context = stage("cuda context")

    args = parse_args(BENCHMARK_ARGS + ["--rank", str(cli.rank)])
    apply_dataset_defaults(args)
    args = argparse.Namespace(
        **{
            **vars(args),
            "rank": cli.rank,
            "method": METHOD,
            "batch_size": cli.batch,
            "adabn_calib_batches": 1,
            "adapt_lr": 1e-3,
        }
    )
    n = cli.batch * 8
    windows = torch.randn(n, args.expected_input_channels, args.window_size)
    labels = torch.randint(args.expected_num_classes, (n,))
    loader = DataLoader(TensorDataset(windows, labels), batch_size=cli.batch)

    model = build_backbone(args).to(device)
    configure_method(model, args)
    model.to(device)
    s_model = stage("model")
    calibrate_minimal_adabn_if_needed(model, METHOD, loader, args, device)
    stage("calibration")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=args.adapt_weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoint = deepcopy(model.state_dict())  # the runner keeps the best weights
    x, y = (tensor.to(device) for tensor in next(iter(loader)))

    def step(hooks=None):
        model.train()
        freeze_bn_eval(model)
        optimizer.zero_grad(set_to_none=True)
        if hooks:
            with torch.autograd.graph.saved_tensors_hooks(*hooks):
                loss = loss_fn(model(x), y)
        else:
            loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    for _ in range(2):
        step()
    s_warm = stage("warm-up")

    cuda = device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    step()
    if cuda:
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
    grads = nbytes(p.grad for p in params)
    stage("training step")

    state = model.state_dict()  # the model's own storages, not the checkpoint copy
    model_storages = {v.untyped_storage().data_ptr() for v in state.values()}
    saved = {"activations": {}, "bitmasks": {}}

    def pack(tensor):
        storage = tensor.untyped_storage()
        if tensor.numel() and storage.data_ptr() not in model_storages:
            packed = tensor.dtype in (torch.uint8, torch.bool)
            saved["bitmasks" if packed else "activations"][storage.data_ptr()] = (
                storage.nbytes()
            )
        return tensor

    step((pack, lambda tensor: tensor))
    activations = sum(saved["activations"].values())
    bitmasks = sum(saved["bitmasks"].values())

    eval_extra = None
    if cuda:
        model.eval()
        with torch.no_grad():
            eval_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            model(x)
            torch.cuda.synchronize()
            eval_extra = torch.cuda.max_memory_allocated() - eval_before
    s_done = stage("done")

    weights = nbytes([*model.parameters(), *model.buffers()])
    adam = nbytes(v for s in optimizer.state.values() for v in s.values())
    copy = nbytes(checkpoint.values())
    batch = nbytes([x, y])
    context = outside_allocator(s_project, s_context)
    kernels = outside_allocator(s_model, s_warm)
    software = [s_start["rss"], s_torch["rss"] - s_start["rss"]]
    software += [s_project["rss"] - s_torch["rss"], context, kernels]
    groups = [
        [
            "Process and libraries",
            [
                ["Python interpreter", software[0]],
                ["PyTorch import", software[1]],
                ["Project imports", software[2]],
                ["CUDA context and driver", context],
                ["cuDNN / cuBLAS handles and kernels", kernels],
            ],
        ],
        [
            "Live between steps",
            [
                ["Weights and buffers", weights],
                ["Adam optimizer state", adam],
                ["Gradients", grads],
                ["Best-checkpoint copy", copy],
                ["Input batch", batch],
                [
                    "Other allocations (library workspaces)",
                    before - weights - adam - copy - batch if cuda else None,
                ],
            ],
        ],
        [
            "Extra at the peak of a step",
            [
                ["Saved activations", activations],
                ["Saved ReLU bitmasks", bitmasks],
                [
                    "Workspace and intermediates",
                    peak - before - activations - bitmasks - grads if cuda else None,
                ],
            ],
        ],
        [
            "Other",
            [
                ["Allocator cache at peak", peak_reserved - peak if cuda else None],
                ["One no-grad eval batch", eval_extra],
            ],
        ],
    ]
    summary = [
        ["Software stack", sum(v for v in software if v is not None)],
        ["PyTorch allocator at peak", peak_reserved if cuda else None],
        ["MemFLoRA training state", weights + adam + grads + activations + bitmasks],
        ["Whole run, system-wide", s_done["used"] - s_start["used"]],
    ]
    with LOCK:
        report.update(device=str(device), groups=groups, summary=summary, done=True)


def print_summary(report: dict) -> None:
    def mib(value):
        return "n/a" if value is None else f"{value / 2**20:9.2f} MiB"

    for name, value in report["summary"]:
        print(f"{name:<40}{mib(value)}")
    for group, rows in report["groups"]:
        print(f"\n{group}")
        for name, value in rows:
            print(f"  {name:<38}{mib(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, help="default: runs/jetson_memory_rRANK")
    parser.add_argument("--cpu", action="store_true", help="measure on the CPU")
    cli = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    out = cli.out or ROOT / "runs" / f"jetson_memory_r{cli.rank}"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "title": f"MemFLoRA, T-ResNet, rank {cli.rank}, batch {cli.batch}",
        "t0": time.time(),
        "samples": [],
        "stages": [],
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

    time.sleep(3)  # a short idle baseline before the run starts
    measure(cli, report)
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
  --s1: #2a78d6; --s2: #eb6834; }
@media (prefers-color-scheme: dark) { :root { --bg: #0d0d0d; --card: #1a1a19;
  --ink: #fff; --ink2: #c3c2b7; --grid: #2c2c2a; --axis: #383835;
  --line: rgba(255,255,255,.1); --s1: #3987e5; --s2: #d95926; } }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 system-ui, sans-serif; padding: 24px 16px; }
main { max-width: 1000px; margin: auto; }
h1 { font-size: 19px; margin: 0; } h2 { font-size: 15px; margin: 0 0 4px; }
.sub, .label { color: var(--ink2); font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 10px; margin: 16px 0; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; margin-bottom: 10px; }
.value { font-size: 22px; font-weight: 600; }
table { width: 100%; border-collapse: collapse; }
td { padding: 3px 8px; } tr.group td { font-weight: 600; padding-top: 12px; }
td.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums;
  width: 110px; }
td.bar { width: 45%; } .fill { height: 8px; background: var(--s1);
  border-radius: 0 4px 4px 0; }
.key { display: inline-block; width: 14px; height: 2px; margin: 0 4px 3px 10px; }
svg { display: block; width: 100%; }
.tick { fill: var(--mute); font-size: 11px; font-variant-numeric: tabular-nums; }
.stage { fill: var(--ink2); font-size: 11px; }
#tip { position: fixed; pointer-events: none; background: var(--card);
  border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px; font-size: 12px; }
</style>
<main>
  <h1 id="title">MemFLoRA memory</h1>
  <div class="sub" id="status">waiting for data…</div>
  <div class="tiles" id="tiles"></div>
  <div class="card" id="breakdown" hidden><table id="table"></table></div>
  <div id="charts"></div>
  <div class="sub">Raw samples: <a href="data.json">data.json</a></div>
</main>
<div id="tip" hidden></div>
<script>
const EMBEDDED = __DATA__;
const $ = (id) => document.getElementById(id);
const MiB = (b) => (b == null ? "n/a" : (b / 1048576).toFixed(2) + " MiB");
const NS = "http://www.w3.org/2000/svg";
const CHARTS = [
  ["Memory in use", "MB", [["RAM used", "ram"]]],
  ["GPU load", "%", [["GPU", "gpu"]], 100],
  ["CPU load", "%", [["Average of all cores", "cpu"], ["Busiest core", "cpu_max"]], 100],
  ["Input power", "W", [["VDD_IN", "power"]]],
];
let data = null, hoverT = null;

function node(tag, text, cls) {
  const n = document.createElement(tag);
  if (text != null) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}
function svgNode(parent, tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  parent.append(n);
  return n;
}
function stageAt(t) {  // a stage is logged when it ends, so it spans from the previous one
  if (!data.stages.length || t < data.stages[0].t) return "idle";
  const s = data.stages.find((row) => row.t >= t);
  return s ? s.name : "idle";
}

function render() {
  const d = data;
  $("title").textContent = d.title;
  $("status").textContent = (d.done ? "Measurement finished" : "Measuring…") +
    (EMBEDDED ? "" : " · live, refreshes every second") + (d.device ? ` · ${d.device}` : "");
  $("tiles").replaceChildren(...(d.summary || []).map(([label, value]) => {
    const card = node("div", null, "card");
    card.append(node("div", label, "label"), node("div", MiB(value), "value"));
    return card;
  }));
  if (d.groups) {
    $("breakdown").hidden = false;
    const largest = Math.max(1, ...d.groups.flatMap(([, rows]) => rows.map(([, v]) => v || 0)));
    const rows = [];
    for (const [group, items] of d.groups) {
      const head = node("tr", null, "group");
      const cell = node("td", group);
      cell.colSpan = 3;
      head.append(cell);
      rows.push(head);
      for (const [name, value] of items) {
        const row = node("tr"), bar = node("td", null, "bar");
        if (value > 0) {
          const fill = node("div", null, "fill");
          fill.style.width = `max(1px, ${(100 * value) / largest}%)`;
          bar.append(fill);
        }
        row.append(node("td", name), node("td", MiB(value), "num"), bar);
        rows.push(row);
      }
    }
    $("table").replaceChildren(...rows);
  }
  $("charts").replaceChildren();
  for (const chart of CHARTS) drawChart(...chart);
}

function drawChart(title, unit, series, yMax) {
  const d = data, samples = d.samples;
  if (!samples.some((s) => s[series[0][1]] != null)) return;
  const card = node("div", null, "card"), heading = node("h2", `${title} (${unit})`);
  if (series.length > 1) {
    series.forEach(([name], i) => {
      const key = node("span", null, "key");
      key.style.background = `var(--s${i + 1})`;
      heading.append(key, node("span", name, "label"));
    });
  }
  card.append(heading);
  $("charts").append(card);
  const W = card.clientWidth - 28, H = 150, L = 46, R = 10, T = 18, B = 22;
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
    const fits = previous !== null && sx - previous > s.name.length * 6.5 + 8;
    if (fits && title === CHARTS[0][0]) {
      svgNode(svg, "text", { x: previous + 4, y: 12, class: "stage" }).textContent = s.name;
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
