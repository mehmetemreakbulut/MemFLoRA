#!/usr/bin/env python3
"""Small Jetson training-update power/time benchmark; no memory tracing or HTML.

Reports total VDD_IN and power/energy above a matched, model-loaded idle baseline.
The parent is stdlib-only; each configuration/repeat uses a fresh CUDA process.
"""
from __future__ import annotations

import argparse
import bisect
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("tresnet", "mobilenetv2")
METHODS = ("full", "lora_c", "lora_edge", "memflora", "memflora_sg")
METRICS = ("idle_w", "total_w", "above_idle_w", "ms_per_update",
           "total_j_per_update", "above_idle_j_per_update")


def parse_power(line):
    """Read the current, NOT running-average, VDD_IN value in tegrastats."""
    match = re.search(r"\bVDD_IN\s+(\d+(?:\.\d+)?)\s*(mW|W)(?=[/\s]|$)", line)
    if not match:
        return None
    value = float(match[1]) / (1000 if match[2] == "mW" else 1)
    return value if math.isfinite(value) and value >= 0 else None


def power_window(samples, start, end, max_gap, min_samples=10):
    """Trapezoidal integration with interpolated boundaries; never extrapolate."""
    if not end > start or max_gap <= 0:
        raise ValueError("Invalid power-window duration or maximum sample gap")
    times = [s["t"] for s in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Power timestamps must increase strictly")
    if any(not math.isfinite(s["w"]) or s["w"] < 0 for s in samples):
        raise ValueError("Invalid power sample")
    if not times or times[0] > start or times[-1] < end:
        raise ValueError("Power samples do not bracket the entire measurement window")
    left = max(0, bisect.bisect_right(times, start) - 1)
    right = bisect.bisect_left(times, end)
    points = samples[left:right + 1]
    inside = sum(start <= s["t"] <= end for s in points)
    if inside < min_samples:
        raise ValueError(f"Too few in-window power samples: {inside} < {min_samples}")
    energy = 0.0
    for a, b in zip(points, points[1:]):
        if b["t"] - a["t"] > max_gap:
            raise ValueError("Power telemetry gap exceeds tolerance")
        lo, hi = max(start, a["t"]), min(end, b["t"])
        if hi <= lo:
            continue
        slope = (b["w"] - a["w"]) / (b["t"] - a["t"])
        p_lo = a["w"] + slope * (lo - a["t"])
        p_hi = a["w"] + slope * (hi - a["t"])
        energy += (p_lo + p_hi) * (hi - lo) / 2
    return {"start": start, "end": end, "duration_s": end - start,
            "samples": inside, "energy_j": energy, "mean_w": energy / (end - start)}


def make_metrics(idle, active, updates):
    if updates <= 0:
        raise ValueError("No complete training updates")
    # A signed difference is important: never hide a noisy negative result.
    extra = active["energy_j"] - idle["mean_w"] * active["duration_s"]
    return {"idle_w": idle["mean_w"], "total_w": active["mean_w"],
            "above_idle_w": active["mean_w"] - idle["mean_w"],
            "ms_per_update": 1000 * active["duration_s"] / updates,
            "total_j_per_update": active["energy_j"] / updates,
            "above_idle_j_per_update": extra / updates}


class Telemetry:
    """Receive monotonic-timestamped samples; terminate only our own collector."""
    def __init__(self, interval_ms):
        executable = shutil.which("tegrastats")
        if not executable:
            raise RuntimeError("tegrastats is required; run this benchmark on the Jetson")
        command = [executable, "--interval", str(interval_ms)]
        if shutil.which("stdbuf"):
            command = ["stdbuf", "-oL", *command]
        self.samples, self.errors = [], []
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            stamp = time.monotonic()
            power = parse_power(line)
            if power is None:
                self.errors.append(line.strip())
                self.errors[:] = self.errors[-10:]
            else:
                self.samples.append({"t": stamp, "w": power, "raw": line.strip()})

    def wait_after(self, stamp, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.samples and self.samples[-1]["t"] >= stamp:
                return
            if self.process.poll() is not None:
                break
            time.sleep(0.02)
        raise RuntimeError("No timely VDD_IN telemetry: " + " | ".join(self.errors))

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.thread.join(timeout=2)
        self.process.stdout.close()


def shared(args):
    sys.path.insert(0, str(args.repo_root.resolve()))
    from tools import jetson_profile
    return jetson_profile


def build_workload(args, torch, device):
    from experiments.benchmark_cli import apply_dataset_defaults, parse_args
    from experiments.benchmark_common import prepare_batch_inputs, method_forces_bn_eval
    from experiments.benchmark_training import SGSession
    from experiments.minimal_methods_benchmark import build_backbone, configure_method
    from src.adapters.bnpa_sg import BNPASGConfig, iter_bnpa_sg_modules
    from src.train import freeze_bn_eval
    from src.utils._bitpack_cuda import cuda_extension

    profile = shared(args)
    method = profile.METHODS[args.worker_method][1]
    if args.worker_method in ("memflora", "memflora_sg") and cuda_extension() is None:
        raise RuntimeError("CUDA bitpacking must load successfully for this benchmark")
    settings = parse_args([
        "--dataset", "opportunity", "--backbone", profile.MODELS[args.worker_model][1],
        "--adapter-layers", profile.MODELS[args.worker_model][2], "--method", method,
        "--rank", str(args.rank), "--window-size", "60", "--window-stride", "30",
        "--mobilenet-v2-pretrained", "false", "--width-mult", "1.0",
        "--t-resnet-feature-maps", "64", "--adabn-calib-batches", "0",
    ])
    apply_dataset_defaults(settings)
    settings.method, settings.rank = method, args.rank
    settings.bnpa_bottleneck_bn = "on"
    model = build_backbone(settings).to(device)
    configure_method(model, settings)
    model.to(device)
    classes = profile.audit_method_implementation(model, args.worker_method)
    sg = SGSession(model, BNPASGConfig(p_lr=0.001 if args.worker_model == "tresnet" else 0.01)) if args.worker_method == "memflora_sg" else None
    sites = len(list(iter_bnpa_sg_modules(model))) if sg else 0
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=0.001, weight_decay=0.0005)
    # Input RNG is independent of how many random numbers adapter setup uses.
    generator = torch.Generator(device=device).manual_seed(args.seed + 10000)
    x = prepare_batch_inputs(torch.randn(args.batch, 97, 60, device=device, generator=generator), settings.backbone)
    y = torch.randint(17, (args.batch,), device=device, generator=generator)
    criterion = torch.nn.CrossEntropyLoss()
    bn_eval = method_forces_bn_eval(method)

    def update():
        model.train()
        if bn_eval:
            freeze_bn_eval(model)
        optimizer.zero_grad(set_to_none=True)
        with sg.begin() if sg else nullcontext():
            loss = criterion(model(x), y)
            loss.backward()
        stats = sg.accumulate(x, bn_eval) if sg else None
        if sg and (stats is None or not stats.performed or stats.sites_accumulated != sites):
            raise RuntimeError("SG failed to recover every projection gradient")
        optimizer.step()
        if sg:
            sg.finish(stats)
        return loss.detach()

    return update, classes


def read_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        return {"returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def timed_updates(update, sync, seconds, sync_every):
    """Amortized wall time, including host work and one sync per update chunk."""
    sync()
    start, count = time.monotonic(), 0
    while time.monotonic() - start < seconds or count == 0:
        for _ in range(sync_every):
            loss = update()
            count += 1
        sync()
        if not math.isfinite(float(loss.item())):
            raise RuntimeError("Non-finite training loss; no valid performance result")
    return start, time.monotonic(), count


def measure(args):
    profile = shared(args)
    torch, device = profile.prepare_runtime(args)
    update, classes = build_workload(args, torch, device)
    info = profile.provenance(args, torch, device)
    info["schema"] = "memflora-jetson-power-v1"
    info["source_sha256"]["tools/jetson_power.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info["nvpmodel_query"] = read_command(["nvpmodel", "-q"])
    info["clocks_query"] = read_command(["jetson_clocks", "--show"])
    model_file = Path("/proc/device-tree/model")
    info["board"] = model_file.read_text().strip("\x00\n") if model_file.exists() else None
    sync = lambda: torch.cuda.synchronize(device)
    print("Warm-up (not measured)", flush=True)
    timed_updates(update, sync, args.warmup_seconds, args.sync_every)
    collector = Telemetry(args.interval_ms)
    try:
        collector.wait_after(time.monotonic())
        print("Settling, then measuring loaded-idle baseline", flush=True)
        time.sleep(args.settle_seconds)
        idle_start = time.monotonic()
        time.sleep(args.idle_seconds)
        idle_end = time.monotonic()
        collector.wait_after(idle_end)
        print("Measuring repeated complete updates", flush=True)
        start, end, updates = timed_updates(update, sync, args.seconds, args.sync_every)
        collector.wait_after(end)
    finally:
        collector.close()
    max_gap = max(1.0, 3 * args.interval_ms / 1000)
    idle = power_window(collector.samples, idle_start, idle_end, max_gap)
    active = power_window(collector.samples, start, end, max_gap)
    metrics = make_metrics(idle, active, updates)
    return {"status": "ok", "model": args.worker_model, "method": args.worker_method,
            "repeat": args.repeat, "batch": args.batch, "rank": args.rank,
            "seed": args.seed, "updates": updates, "metrics": metrics,
            "idle_window": idle, "active_window": active,
            "warning": "Nonpositive idle-subtracted power; inspect baseline/telemetry" if metrics["above_idle_w"] <= 0 else None,
            "module_class_counts": classes, "provenance": info,
            "protocol": {"warmup_seconds": args.warmup_seconds, "settle_seconds": args.settle_seconds,
                         "idle_seconds": args.idle_seconds, "requested_active_seconds": args.seconds,
                         "interval_ms": args.interval_ms, "sync_every": args.sync_every,
                         "baseline": "same loaded process, post-warmup, no training",
                         "synthetic_inputs": True, "checkpoint_copy": False,
                         "allocation_tracing": False, "sample_clock": "monotonic receipt time"},
            "power_samples": collector.samples}


def summary_rows(results, models, methods, repeats):
    rows = []
    for model in models:
        for method in methods:
            group = [r for r in results if r["model"] == model and r["method"] == method and r["status"] == "ok"]
            row = {"model": model, "method": method, "valid_repeats": len(group),
                   "status": "ok" if len(group) == repeats else "incomplete"}
            for metric in METRICS:
                values = [r["metrics"][metric] for r in group]
                row[metric] = statistics.mean(values) if len(group) == repeats else None
                row[metric + "_sd"] = statistics.stdev(values) if len(group) == repeats and repeats > 1 else None
            rows.append(row)
    return rows


def write_results(out, args, results):
    (out / "results.json").write_text(json.dumps({"schema": "memflora-jetson-power-v1",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": results}, indent=2, allow_nan=False))
    rows = summary_rows(results, args.models, args.methods, args.repeats)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--idle-seconds", type=float, default=10)
    p.add_argument("--warmup-seconds", type=float, default=5)
    p.add_argument("--settle-seconds", type=float, default=5)
    p.add_argument("--interval-ms", type=int, default=200)
    p.add_argument("--sync-every", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-model", choices=MODELS, help=argparse.SUPPRESS)
    p.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    p.add_argument("--repeat", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if any(getattr(args, k) <= 0 for k in ("batch", "rank", "repeats", "threads", "sync_every", "interval_ms")):
        p.error("Counts and sampling interval must be positive")
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in ("seconds", "idle_seconds", "warmup_seconds")):
        p.error("Measurement and warm-up durations must be finite and positive")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        p.error("Settle time must be finite and nonnegative")
    if min(args.seconds, args.idle_seconds) < 12 * args.interval_ms / 1000:
        p.error("Active and idle windows must span at least 12 sampling intervals")
    if len(set(args.models)) != len(args.models) or len(set(args.methods)) != len(args.methods):
        p.error("Duplicate models/methods are not allowed")
    if args.worker_model and (not args.worker_method or args.worker_out is None):
        p.error("Worker requires method and output")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        for model in args.models:
            for method in args.methods:
                print(f"{model} / {method} / B={args.batch} / r={args.rank} / {args.repeats} repeats")
        return 0
    if args.preflight:
        shared(args).runtime_preflight(args)
        collector = Telemetry(args.interval_ms)
        try:
            collector.wait_after(time.monotonic())
        finally:
            collector.close()
        print("VDD_IN telemetry available")
        return 0
    if args.worker_model:
        try:
            result = measure(args)
        except Exception as exc:
            traceback.print_exc()
            result = {"status": "failed", "model": args.worker_model, "method": args.worker_method,
                      "repeat": args.repeat, "error": str(exc)}
        args.worker_out.write_text(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["status"] == "ok" else 1

    # Never silently run two copies of this benchmark on the same device.
    import fcntl
    lock = open(f"/tmp/memflora_jetson_power_{os.getuid()}.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another power benchmark is already running", file=sys.stderr)
        return 1
    common = [sys.executable, str(Path(__file__).resolve()), "--repo-root", str(args.repo_root.resolve())]
    for key in ("device", "threads", "batch", "rank", "seconds", "idle_seconds", "warmup_seconds", "settle_seconds", "interval_ms", "sync_every"):
        common += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    print("Checking CUDA/imports and input-power telemetry...", flush=True)
    if subprocess.run(common + ["--preflight"]).returncode:
        return 1
    out = args.out or args.repo_root / "runs" / ("jetson_power_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    print(f"Output: {out}\nIdle baseline: model loaded, no training. No memory tracing.", flush=True)
    for repeat in range(1, args.repeats + 1):
        for model in args.models:
            for method in args.methods:
                label = f"{model}_{method}_r{repeat}"
                case_path = out / (label + ".json")
                print(f"[{repeat}/{args.repeats}] {model} / {method}", flush=True)
                with (out / (label + ".log")).open("w") as log:
                    completed = subprocess.run(common + ["--worker-model", model, "--worker-method", method,
                        "--repeat", str(repeat), "--seed", str(args.seed + repeat - 1),
                        "--worker-out", str(case_path)], stdout=log, stderr=subprocess.STDOUT)
                result = json.loads(case_path.read_text()) if case_path.exists() else {
                    "status": "failed", "model": model, "method": method, "repeat": repeat,
                    "error": f"Worker exited {completed.returncode}; inspect {label}.log"}
                if completed.returncode and result["status"] == "ok":
                    result["status"], result["error"] = "failed", "Nonzero worker exit"
                results.append(result)
                write_results(out, args, results)
                if result["status"] == "ok":
                    m = result["metrics"]
                    print(f"  {m['ms_per_update']:.2f} ms/update | total {m['total_w']:.3f} W | idle {m['idle_w']:.3f} W | above idle {m['above_idle_w']:.3f} W | above idle {m['above_idle_j_per_update']:.5f} J/update", flush=True)
                    if result["warning"]:
                        print("  WARNING: " + result["warning"], flush=True)
                else:
                    print("  FAILED: " + result["error"], flush=True)
    print(f"Summary: {out / 'summary.csv'}", flush=True)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
