#!/usr/bin/env python3
"""One-command, memory-only Table 3 sweep; no dataset or accuracy evaluation.

Run in the MemFLoRA checkout on Jetson:
    python tools/jetson_profile.py

The parent uses only the standard library. Each cell gets a fresh CUDA process.
See docs/jetson_table3.md for scope, inclusion rules, and validation limits.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "memflora-table3-distinct-storage-v1"
MODELS = {
    "tresnet": ("T-ResNet", "t_resnet_official", "all"),
    "mobilenetv2": ("MobileNetV2", "mobilenet_v2", "pointwise_only"),
}
METHODS = {
    "full": ("Full FT", "full"),
    "lora_c": ("LoRA-C", "lora_c"),
    "lora_edge": ("LoRA-Edge", "lora_edge_optimized"),
    "memflora": ("MemFLoRA", "bnpa_fa_postbn_bnr_off_scaled"),
    "memflora_sg": ("MemFLoRA-SG", "bnpa_q3_sg"),
}
# Exact printed reference, NOT corrected values or target measurements.
PAPER_SOURCE = "MemFLoRA (58).pdf, Table 3, page 5 (2026-09-30 version)"
PAPER_BATCHES = (1, 8, 32, 64)
PAPER_TABLE3 = {
    ("tresnet", "full"): ([(9.02, .90), (13.84, 7.09), (35.08, 28.33), (63.40, 56.65)], 4.49),
    ("tresnet", "lora_c"): ([(5.51, 3.10), (11.53, 9.12), (32.20, 29.79), (59.76, 57.35)], .10),
    ("tresnet", "lora_edge"): ([(2.93, .60), (7.05, 4.72), (21.19, 18.86), (40.04, 37.71)], .02),
    ("tresnet", "memflora"): ([(2.44, .02), (2.50, .11), (2.81, .42), (3.23, .83)], .08),
    ("tresnet", "memflora_sg"): ([(2.50, .02), (2.78, .11), (3.74, .42), (5.01, .83)], .10),
    ("mobilenetv2", "full"): ([(43.20, 16.12), (109.00, 81.91), (346.93, 319.85), (666.64, 639.56)], 17.97),
    ("mobilenetv2", "lora_c"): ([(27.17, 17.61), (90.00, 80.44), (305.40, 295.84), (592.61, 583.05)], .30),
    ("mobilenetv2", "lora_edge"): ([(17.09, 7.66), (69.77, 60.34), (250.39, 240.96), (491.22, 481.79)], .16),
    ("mobilenetv2", "memflora"): ([(9.63, .20), (10.54, 1.11), (13.66, 4.23), (17.83, 8.40)], .16),
    ("mobilenetv2", "memflora_sg"): ([(9.91, .20), (11.73, 1.11), (17.97, 4.23), (26.30, 8.40)], .16),
}
BASE_STATE = {"model", "optimizer", "projection_optimizer"}
EXCLUDED = {"input", "checkpoint"}
SAVED = {"saved", "mask"}
CORE = BASE_STATE | SAVED | {"gradient", "projection_gradient", "sg_capture"}
TRAINING = CORE | {"sg_replay_input"}
# Exclusive attribution of aliases. Independent metrics use the union of roles.
PRIORITY = (
    "model", "optimizer", "projection_optimizer", "input", "checkpoint",
    "gradient", "projection_gradient", "sg_capture", "mask", "saved",
    "sg_replay_input", "other",
)


class AccountingError(RuntimeError):
    pass


def allocation_records(trace, before, tags):
    """Resolve an observed storage to its live allocation *generation*.

    An observation's event offset is the number of events already executed.
    We never use the last-ever allocation at an address to classify an old one.
    Tensor observations contain integers only, so instrumentation retains no
    tensor/storage references. Sizes below are requested backing-allocation bytes.
    """
    records, live = [], {}
    for block in before:
        record = dict(block, start=0, end=len(trace) + 1,
                      roles={block.get("owner", "other")})
        records.append(record)
        if record["address"] in live:
            raise AccountingError("Duplicate initial allocation address")
        live[record["address"]] = record
    observations = defaultdict(list)
    for tag in tags:
        if not 0 <= tag["event"] <= len(trace):
            raise AccountingError("Storage observation outside recorded trace")
        observations[tag["event"]].append(tag)

    def observe(offset):
        for tag in observations[offset]:
            record = live.get(tag["address"])
            if record is None:
                raise AccountingError(f"No live allocation for observation: {tag}")
            if tag["bytes"] > record["bytes"]:
                raise AccountingError("Backing storage exceeds matched allocation")
            record["roles"].add(tag["kind"])

    observe(0)
    for i, event in enumerate(trace, 1):
        action, address = event["action"], event.get("addr")
        if action == "alloc":
            if address in live:
                raise AccountingError("Allocator reused an address still live")
            record = {"address": address, "bytes": event["size"],
                      "start": i, "end": len(trace) + 1, "roles": set()}
            records.append(record)
            live[address] = record
        elif action == "free_requested":
            if address not in live:
                raise AccountingError("Unmatched free request; incomplete trace")
            live.pop(address)["end"] = i
        elif action == "free_completed":
            # Requested bytes stop at free_requested. A completion is NOT a
            # second free, nor may it remove a later generation of the address.
            pass
        elif action == "oom":
            raise AccountingError("OOM event in measured step")
        observe(i)
    return records


def effective_roles(record):
    roles = record["roles"]
    # Model/optimizer storage saved by autograd still counts only in its owner.
    for role in PRIORITY:
        if role in roles and role in BASE_STATE | EXCLUDED:
            return {role}
    return roles - {"other"}


def analyse_trace(trace, before, initial_reserved, tags, boundaries):
    records = allocation_records(trace, before, tags)
    deltas = defaultdict(lambda: defaultdict(int))
    for record in records:
        roles = effective_roles(record)
        category = next((r for r in PRIORITY if r in roles), "other")
        flags = {"requested", "category:" + category}
        if roles & TRAINING:
            flags.add("training")
        if roles & CORE:
            flags.add("core")
        if roles & SAVED:
            flags.add("saved")
        if "mask" in roles:
            flags.add("mask")
        if roles & {"gradient", "projection_gradient"}:
            flags.add("gradients")
        if "sg_capture" in roles:
            flags.add("sg_capture")
        if "sg_replay_input" in roles:
            flags.add("sg_replay")
        for flag in flags:
            deltas[record["start"]][flag] += record["bytes"]
            deltas[record["end"]][flag] -= record["bytes"]
    state = defaultdict(int)
    peaks = defaultdict(int)
    reserved = initial_reserved
    reserved_peak = reserved
    phase_states = {}
    peak_event, peak_parts = 0, {}
    for offset in range(len(trace) + 1):
        if offset:
            event = trace[offset - 1]
            if event["action"] in ("segment_alloc", "segment_map"):
                reserved += event["size"]
            elif event["action"] in ("segment_free", "segment_unmap"):
                reserved -= event["size"]
        for key, delta in deltas[offset].items():
            state[key] += delta
            if state[key] < 0:
                raise AccountingError(f"Negative live bytes for {key}")
        if reserved < state["requested"]:
            raise AccountingError("Requested bytes exceed reserved bytes")
        reserved_peak = max(reserved_peak, reserved)
        if offset == 0 or state["training"] > peaks["training"]:
            peak_event = offset
            peak_parts = {r: state["category:" + r] for r in PRIORITY if r in TRAINING}
        for key, value in state.items():
            peaks[key] = max(peaks[key], value)
        for name, boundary in boundaries.items():
            if offset == boundary:
                phase_states[name] = dict(state)
    if "forward" not in phase_states:
        raise AccountingError("Missing end-forward boundary")
    forward = phase_states["forward"]
    metrics = {
        "training_state_peak_bytes": peaks["training"],
        "training_state_core_peak_bytes": peaks["core"],
        "saved_backward_end_forward_bytes": forward.get("saved", 0),
        "saved_bitmasks_end_forward_bytes": forward.get("mask", 0),
        "parameter_gradients_peak_bytes": peaks["gradients"],
        "sg_captured_gradients_peak_bytes": peaks["sg_capture"],
        "sg_replay_inputs_peak_bytes": peaks["sg_replay"],
        "cuda_requested_peak_bytes": peaks["requested"],
        "cuda_reserved_peak_bytes": reserved_peak,
        "training_peak_event": peak_event,
    }
    for role in BASE_STATE | {"input", "checkpoint"}:
        metrics[role + "_bytes"] = peaks["category:" + role]
    metrics["optimizer_total_bytes"] = metrics["optimizer_bytes"] + metrics["projection_optimizer_bytes"]
    return {"metrics": metrics, "training_peak_components": peak_parts,
            "phase_states": phase_states, "allocation_count": len(records)}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 64])
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--out", type=Path, help="New output directory; existing paths are never overwritten")
    p.add_argument("--checkpoint-copy", action="store_true", help="Keep a best-state copy outside training-state accounting")
    p.add_argument("--trace-stacks", action="store_true", help="Include stacks in optional full snapshots (much slower)")
    p.add_argument("--save-snapshots", action="store_true", help="Also save full PyTorch snapshots; compact allocation traces are always saved")
    p.add_argument("--max-events", type=int, default=1000000)
    p.add_argument("--allow-bitpack-fallback", action="store_true", help="Allow PyTorch mask packing if the compiled CUDA backend is unavailable")
    p.add_argument("--dry-run", action="store_true", help="List the sweep without importing PyTorch or creating outputs")
    p.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-model", choices=MODELS, help=argparse.SUPPRESS)
    p.add_argument("--worker-method", choices=METHODS, help=argparse.SUPPRESS)
    p.add_argument("--worker-batch", type=int, help=argparse.SUPPRESS)
    p.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return p


def sweep_cases(args):
    return list(itertools.product(args.models, args.methods, args.batches))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tree_tensors(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tree_tensors(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tree_tensors(item, torch)


def storage_rows(tensors, device):
    rows = {}
    for tensor in tensors:
        if tensor.numel() and tensor.device == device:
            storage = tensor.untyped_storage()
            rows[storage.data_ptr()] = storage.nbytes()
    return rows


class Observer:
    def __init__(self, torch, device, max_events):
        self.torch, self.device, self.max_events = torch, device, max_events
        self.tags, self.boundaries, self.phase_requested = [], {}, {}
        self.last_offset = 0

    def snapshot(self):
        snap = self.torch.cuda.memory._snapshot()
        trace = snap["device_traces"][self.device.index]
        if len(trace) >= self.max_events or len(trace) < self.last_offset:
            raise AccountingError("Trace history truncated; increase --max-events")
        self.last_offset = len(trace)
        return snap

    def observe(self, tensors, kind):
        rows = storage_rows(tensors, self.device)
        if not rows:
            return
        self.snapshot()
        self.tags.extend({"event": self.last_offset, "address": addr,
                          "bytes": size, "kind": kind} for addr, size in rows.items())

    def pack(self, tensor):
        self.observe([tensor], "mask" if tensor.dtype in (self.torch.uint8, self.torch.bool) else "saved")
        return tensor.detach()  # Same storage; no copy or instrumentation-owned reference.

    def boundary(self, name):
        self.snapshot()
        self.boundaries[name] = self.last_offset
        self.phase_requested[name] = self.torch.cuda.memory_stats(self.device)["requested_bytes.all.current"]


def provenance(args, torch, device):
    from src.utils._bitpack_cuda import backend_status

    root = args.repo_root.resolve()
    sources = [p for folder in ("src", "experiments") for p in sorted((root / folder).rglob("*"))
               if p.suffix in (".py", ".cu", ".cpp")]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    hashes["tools/jetson_profile.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, check=False)
    l4t = Path("/etc/nv_tegra_release")
    return {
        "schema": SCHEMA, "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "l4t": l4t.read_text().strip() if l4t.exists() else None,
        "git_commit": git.stdout.strip() if git.returncode == 0 else None,
        "git_status": status.stdout, "source_sha256": hashes,
        "bitpack": backend_status(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "environment": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES")},
    }


def case_config(args):
    return {
        "model": args.worker_model, "method": args.worker_method,
        "implementation_method": METHODS[args.worker_method][1],
        "batch": args.worker_batch, "rank": args.rank, "seed": args.seed,
        "warmup_steps": args.warmup, "measured_steps": 1, "dtype": "float32",
        "dataset_shape": "Opportunity", "channels": 97, "window": 60, "classes": 17,
        "synthetic_inputs": True, "random_initial_weights": True,
        "adabn_calibration": False, "checkpoint_copy": args.checkpoint_copy,
        "adapter_layers": MODELS[args.worker_model][2],
        "ordinary_adam_lr": 0.001, "ordinary_adam_weight_decay": 0.0005,
        "sg_projection_lr": 0.001 if args.worker_model == "tresnet" else 0.01,
        "sg_projection_weight_decay": 0.0,
    }


def prepare_runtime(args):
    # Set before importing torch. Preserve explicit user choices and record them.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    if not any(k in os.environ for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU tensor counts are not CUDA peak measurements.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Only CUDA devices are supported for measured results")
    device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    torch.cuda.set_device(device)
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("The native CUDA allocator is required for trace/counter validation")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(args.repo_root.resolve()))
    return torch, device


def runtime_preflight(args):
    """Exit this process before measurement workers create CUDA contexts."""
    torch, device = prepare_runtime(args)
    # Verify shared imports too, without building a model or compiling kernels.
    import experiments.minimal_methods_benchmark  # noqa: F401
    import experiments.benchmark_training  # noqa: F401
    print(f"Python: {sys.executable}; PyTorch: {torch.__version__}; CUDA device: {device}")


def check_environment(args):
    command = [sys.executable, str(Path(__file__).resolve()), "--preflight",
               "--repo-root", str(args.repo_root.resolve()), "--device", args.device,
               "--threads", str(args.threads), "--seed", str(args.seed)]
    print("Checking Python/CUDA environment before starting cases...", flush=True)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        print(f"Environment check failed for {sys.executable}. No cases were started.", file=sys.stderr)
        print(completed.stdout + completed.stderr, file=sys.stderr)
        print("Activate your CUDA-enabled MemFLoRA environment, or use its Python explicitly "
              "(on your Jetson: ~/venvs/memflora/bin/python tools/jetson_profile.py).", file=sys.stderr)
        return False
    print(completed.stdout.strip(), flush=True)
    if completed.stderr:
        print(completed.stderr.strip(), file=sys.stderr)
    return True


def audit_method_implementation(model, method):
    """Allocator checks cannot detect a wrongly selected adapter class."""
    counts = dict(sorted(Counter(type(m).__name__ for m in model.modules()).items()))
    if method == "lora_edge":
        from src.adapters.lora_edge import LoRAEdgeConv2d, LoRAEdgeConv2dOptimized
        adapters = [m for m in model.modules() if isinstance(m, LoRAEdgeConv2d)]
        if not adapters or any(type(m) is not LoRAEdgeConv2dOptimized for m in adapters):
            raise RuntimeError(
                "LoRA-Edge implementation mismatch: expected LoRAEdgeConv2dOptimized "
                "at every selected site. Check optimized=True in the adapter factory. "
                f"Observed module classes: {counts}"
            )
    return counts


def measure(args):
    torch, device = prepare_runtime(args)
    from experiments.benchmark_cli import apply_dataset_defaults, parse_args
    from experiments.benchmark_common import prepare_batch_inputs, method_forces_bn_eval
    from experiments.benchmark_training import SGSession, changing_state
    from experiments.minimal_methods_benchmark import build_backbone, configure_method
    from src.adapters.bnpa_sg import BNPASGConfig, iter_bnpa_sg_modules, iter_bnpa_sg_projection_parameters
    from src.train import freeze_bn_eval
    from src.utils._bitpack_cuda import cuda_extension

    config = case_config(args)
    model_key, method = args.worker_model, config["implementation_method"]
    print(f"Building {model_key} / {args.worker_method} / B={args.worker_batch}", flush=True)
    if args.worker_method in ("memflora", "memflora_sg"):
        if cuda_extension() is None and not args.allow_bitpack_fallback:
            raise RuntimeError("CUDA bitpacking unavailable; fix the extension or explicitly allow fallback")
    settings = parse_args([
        "--dataset", "opportunity", "--backbone", MODELS[model_key][1],
        "--adapter-layers", MODELS[model_key][2], "--method", method,
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
    module_classes = audit_method_implementation(model, args.worker_method)
    print(f"Implementation classes: {module_classes}", flush=True)
    sg = SGSession(model, BNPASGConfig(p_lr=config["sg_projection_lr"])) if args.worker_method == "memflora_sg" else None
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0005)
    optimizers = [optimizer, sg.optimizer] if sg else [optimizer]
    checkpoint = changing_state(model, *optimizers) if args.checkpoint_copy else {}
    x = prepare_batch_inputs(torch.randn(args.worker_batch, 97, 60, device=device), settings.backbone)
    y = torch.randint(17, (args.worker_batch,), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    bn_eval = method_forces_bn_eval(method)
    config["force_bn_eval"] = bn_eval
    config["ordinary_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projection_params = list(iter_bnpa_sg_projection_parameters(model)) if sg else []
    config["projection_parameters_updated_by_sg"] = sum(p.numel() for p in projection_params)
    sg_modules = list(iter_bnpa_sg_modules(model)) if sg else []

    def start():
        model.train()
        if bn_eval:
            freeze_bn_eval(model)
        optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        start()
        with sg.begin() if sg else nullcontext():
            loss = criterion(model(x), y)
            loss.backward()
        stats = sg.accumulate(x, bn_eval) if sg else None
        optimizer.step()
        if sg:
            sg.finish(stats)
        if not math.isfinite(loss.item()):
            raise RuntimeError("Non-finite warm-up loss")
        del loss
    print("Warm-up complete; tracing one full adaptation step", flush=True)
    start()
    if sg:
        sg.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)

    owner_tensors = {
        "model": list(model.parameters()) + list(model.buffers()),
        "optimizer": list(tree_tensors(optimizer.state, torch)),
        "projection_optimizer": list(tree_tensors(sg.optimizer.state, torch)) if sg else [],
        "input": [x, y], "checkpoint": list(checkpoint.values()),
    }
    owners = {}
    owner_storage = {role: storage_rows(tensors, device) for role, tensors in owner_tensors.items()}
    for role, storages in owner_storage.items():
        for address in storages:
            if address in owners and owners[address] != role:
                raise AccountingError("Storage shared across incompatible baseline owners")
            owners[address] = role
    cpu = torch.device("cpu")
    cpu_opt = storage_rows(owner_tensors["optimizer"] + owner_tensors["projection_optimizer"], cpu)
    optimizer_cpu_bytes = sum(cpu_opt.values())
    del owner_tensors  # Keep scalar identities, never extra owners of CUDA state.

    torch.cuda.memory._record_memory_history(enabled="all", context="all" if args.trace_stacks else None,
                                            max_entries=args.max_events, clear_history=True)
    observer = Observer(torch, device, args.max_events)
    initial = observer.snapshot()
    before = []
    for segment in initial["segments"]:
        if segment["device"] != device.index:
            continue
        for block in segment["blocks"]:
            if block["state"] == "active_awaiting_free":
                raise AccountingError("Pending asynchronous free at synchronized baseline")
            if block["state"] == "active_allocated":
                before.append({"address": block["address"], "bytes": block["requested_size"],
                               "owner": owners.get(block["address"], "other")})
    if set(owners) - {b["address"] for b in before}:
        raise AccountingError("Baseline tensor storage missing from allocator snapshot")
    initial_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    callbacks, handles = [], []
    replay_enabled = False

    for _, module in sg_modules:
        original = module._bnpa_grad_q_capture_callback

        def capture(grad, module=module, original=original):
            original(grad)
            if module.last_grad_q is not None:
                observer.observe([module.last_grad_q], "sg_capture")

        callbacks.append((module, original))
        module._bnpa_grad_q_capture_callback = capture

        def replay_input(_module, inputs):
            if replay_enabled:
                observer.observe(inputs[:1], "sg_replay_input")

        handles.append(module.register_forward_pre_hook(replay_input))
    try:
        with sg.begin() if sg else nullcontext():
            with torch.autograd.graph.saved_tensors_hooks(observer.pack, lambda tensor: tensor):
                loss = criterion(model(x), y)
            observer.boundary("forward")
            loss.backward()
        ordinary_grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        observer.observe(ordinary_grads, "gradient")
        del ordinary_grads  # Never extend a gradient lifetime through instrumentation.
        observer.boundary("backward")
        replay_enabled = True
        stats = sg.accumulate(x, bn_eval) if sg else None
        replay_enabled = False
        if sg:
            observer.observe([p.grad for p in projection_params if p.grad is not None], "projection_gradient")
            observer.boundary("replay")
            if stats is None or stats.sites_accumulated != len(sg_modules):
                raise AccountingError("SG did not recover every projection gradient")
        optimizer.step()
        if sg:
            sg.finish(stats)
        torch.cuda.synchronize(device)
        observer.boundary("step_end")
        final = observer.snapshot()
        counters = dict(torch.cuda.memory_stats(device))
    finally:
        for handle in handles:
            handle.remove()
        for module, original in callbacks:
            module._bnpa_grad_q_capture_callback = original
        torch.cuda.memory._record_memory_history(None)
    if not math.isfinite(loss.item()):
        raise RuntimeError("Non-finite measured loss")
    trace = [{k: e[k] for k in ("action", "addr", "size", "stream") if k in e}
             for e in final["device_traces"][device.index]]
    # Preserve raw evidence even if attribution or a validation check fails.
    replay_data = {"schema": SCHEMA, "trace": trace, "before": before,
                   "initial_reserved": initial_reserved, "tags": observer.tags,
                   "boundaries": observer.boundaries, "phase_requested": observer.phase_requested,
                   "allocator_counters": counters}
    with gzip.open(args.worker_output.with_suffix(".trace.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(replay_data, stream)
    if args.save_snapshots:
        import pickle
        with args.worker_output.with_suffix(".snapshot.pickle").open("wb") as stream:
            pickle.dump(final, stream)
    result = analyse_trace(trace, before, initial_reserved, observer.tags, observer.boundaries)
    metrics = result["metrics"]
    metrics["optimizer_cpu_state_bytes"] = optimizer_cpu_bytes
    metrics["cuda_allocated_peak_bytes"] = counters["allocated_bytes.all.peak"]
    metrics["saved_nonmask_end_forward_bytes"] = metrics["saved_backward_end_forward_bytes"] - metrics["saved_bitmasks_end_forward_bytes"]
    checks = {
        "lora_edge_class_matches_request": args.worker_method != "lora_edge" or
            module_classes.get("LoRAEdgeConv2dOptimized", 0) > 0,
        "requested_peak_matches_counter": metrics["cuda_requested_peak_bytes"] == counters["requested_bytes.all.peak"],
        "reserved_peak_matches_counter": metrics["cuda_reserved_peak_bytes"] == counters["reserved_bytes.all.peak"],
        "training_peak_partition": sum(result["training_peak_components"].values()) == metrics["training_state_peak_bytes"],
        "training_no_larger_than_requested": metrics["training_state_peak_bytes"] <= metrics["cuda_requested_peak_bytes"],
        "ordinary_adam_materialized": metrics["optimizer_bytes"] > 0,
        "projection_adam_materialized": not sg or metrics["projection_optimizer_bytes"] > 0,
        "sg_capture_observed": not sg or metrics["sg_captured_gradients_peak_bytes"] > 0,
        "sg_replay_inputs_observed": not sg or metrics["sg_replay_inputs_peak_bytes"] > 0,
        "model_storage_stable": storage_rows(list(model.parameters()) + list(model.buffers()), device) == owner_storage["model"],
        "ordinary_adam_storage_stable": storage_rows(tree_tensors(optimizer.state, torch), device) == owner_storage["optimizer"],
        "projection_adam_storage_stable": not sg or storage_rows(tree_tensors(sg.optimizer.state, torch), device) == owner_storage["projection_optimizer"],
    }
    for name, requested in observer.phase_requested.items():
        checks[name + "_requested_matches_counter"] = result["phase_states"][name]["requested"] == requested
    result.update(config=config, status="ok" if all(checks.values()) else "invalid",
                  checks=checks, module_class_counts=module_classes,
                  provenance=provenance(args, torch, device))
    return result


def dump_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_results(out, args, results):
    write_json(out / "results.json", {"schema": SCHEMA, "results": results})
    measurements = []
    for result in results:
        measurements.append({**result["config"], "status": result["status"],
                             **result.get("metrics", {}), "error": result.get("error", "")})
    fields = list(dict.fromkeys(k for row in measurements for k in row))
    dump_csv(out / "measurements.csv", measurements, fields)
    lookup = {(r["config"]["model"], r["config"]["method"], r["config"]["batch"]): r for r in results}
    rows, md, tex = [], [], []
    header = ["Backbone", "Method"] + [f"B={b} peak / saved [MB]" for b in args.batches] + ["Optimizer total [MB]"]
    md.extend(["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"])
    tex.append(" & ".join(["Backbone", "Method"] + [f"$B={b}$" for b in args.batches] + ["Opt. [MB]"]) + r" \\")
    tex.append(r"\hline")
    for model in args.models:
        for method in args.methods:
            values, opts = [], set()
            for batch in args.batches:
                result = lookup.get((model, method, batch))
                if result is None:
                    values.append("PENDING")
                elif result["status"] != "ok":
                    values.append("FAILED")
                else:
                    m = result["metrics"]
                    values.append(f'{m["training_state_peak_bytes"]/1e6:.2f} / {m["saved_backward_end_forward_bytes"]/1e6:.2f}')
                    opts.add(m["optimizer_total_bytes"])
            opt = f"{next(iter(opts))/1e6:.2f}" if len(opts) == 1 else ("VARIES" if opts else "—")
            cells = [MODELS[model][0], METHODS[method][0], *values, opt]
            rows.append(dict(zip(header, cells)))
            md.append("| " + " | ".join(cells) + " |")
            tex.append(" & ".join(cells) + r" \\")
    note = (
        "Peak / end-forward saved-backward state; decimal MB. Optimizer total is already included "
        "in peak and includes SG projection Adam. Distinct requested CUDA backing storage; "
        "peak includes SG captured gradients and adapted-site replay inputs. Inputs, checkpoint "
        "copies, general workspaces, cache and host optimizer scalars are excluded. "
        "Synthetic FP32 Opportunity-shaped inputs, random initial weights, no AdaBN calibration. "
        "This is not a whole-process RAM measurement. PENDING/FAILED cells are not results. "
        "Counter agreement validates totals, not every semantic category."
    )
    (out / "table3.md").write_text("\n".join(md) + "\n\n" + note + "\n")
    (out / "table3.tex").write_text("% " + note + "\n" + "\n".join(tex) + "\n")
    dump_csv(out / "table3.csv", rows, header)
    reductions = []
    for result in results:
        c = result["config"]
        reference = lookup.get((c["model"], "full", c["batch"]))
        if result["status"] != "ok" or not reference or reference["status"] != "ok":
            continue
        row = {"model": c["model"], "method": c["method"], "batch": c["batch"]}
        for short, metric in (("peak_training", "training_state_peak_bytes"), ("saved_backward", "saved_backward_end_forward_bytes"), ("requested_cuda", "cuda_requested_peak_bytes")):
            denominator = reference["metrics"][metric]
            row[short + "_reduction_percent"] = 100 * (1 - result["metrics"][metric] / denominator) if denominator else ""
        reductions.append(row)
    dump_csv(out / "reductions.csv", reductions, ["model", "method", "batch", "peak_training_reduction_percent", "saved_backward_reduction_percent", "requested_cuda_reduction_percent"])
    export_paper_comparison(out, args, results)


def comparison_status(result, rank):
    if rank != 2:
        return "RANK MISMATCH"
    if result is None:
        return "PENDING"
    if result["status"] != "ok":
        return "FAILED"
    if result["config"].get("rank") != 2:
        return "RANK MISMATCH"
    if (result["config"]["method"] == "lora_edge" and
            not result.get("checks", {}).get("lora_edge_class_matches_request", False)):
        return "UNVERIFIED IMPLEMENTATION"
    return "ok"


def export_paper_comparison(out, args, results):
    """Same grid as Table 3, paired paper/Jetson rows; never invent missing cells."""
    lookup = {(r["config"]["model"], r["config"]["method"], r["config"]["batch"]): r for r in results}
    header = ["Backbone", "Method", "Source"] + [f"B={b} peak / saved [MB]" for b in PAPER_BATCHES] + ["Optimizer [MB]"]
    md = ["# Table 3: paper versus Jetson", "", f"Reference: {PAPER_SOURCE}.", "",
          "Each paper row is followed by its Jetson row. Entries are peak / saved state in decimal MB; optimizer is already included in peak.", "",
          "| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    rows = []
    for model in MODELS:
        for method in METHODS:
            reference, paper_opt = PAPER_TABLE3[model, method]
            paper_cells, jetson_cells, opts, statuses = [], [], set(), []
            for batch, (paper_peak, paper_saved) in zip(PAPER_BATCHES, reference):
                paper_cells.append(f"{paper_peak:.2f} / {paper_saved:.2f}")
                result = lookup.get((model, method, batch))
                status = comparison_status(result, args.rank)
                statuses.append(status)
                row = {"backbone": MODELS[model][0], "method": METHODS[method][0],
                       "batch": batch, "status": status, "paper_source": PAPER_SOURCE,
                       "paper_peak_mb": paper_peak, "paper_saved_mb": paper_saved,
                       "paper_optimizer_mb": paper_opt}
                if status == "ok":
                    m = result["metrics"]
                    peak, saved, opt = (m[k] / 1e6 for k in
                        ("training_state_peak_bytes", "saved_backward_end_forward_bytes", "optimizer_total_bytes"))
                    jetson_cells.append(f"{peak:.2f} / {saved:.2f}")
                    opts.add(m["optimizer_total_bytes"])
                    row.update(jetson_peak_bytes=m["training_state_peak_bytes"],
                               jetson_saved_bytes=m["saved_backward_end_forward_bytes"],
                               jetson_optimizer_bytes=m["optimizer_total_bytes"],
                               jetson_peak_mb=peak, jetson_saved_mb=saved, jetson_optimizer_mb=opt,
                               delta_peak_mb=round(peak-paper_peak, 6),
                               delta_saved_mb=round(saved-paper_saved, 6),
                               delta_optimizer_mb=round(opt-paper_opt, 6))
                else:
                    jetson_cells.append(status)
                rows.append(row)
            if all(status == "ok" for status in statuses):
                opt_cell = f"{next(iter(opts))/1e6:.2f}" if len(opts) == 1 else "VARIES"
            else:
                opt_cell = "INCOMPLETE"
            for source, cells, opt in [("Paper", paper_cells, f"{paper_opt:.2f}"), ("Jetson", jetson_cells, opt_cell)]:
                md.append("| " + " | ".join([MODELS[model][0], METHODS[method][0], source, *cells, opt]) + " |")
    md.extend(["", "Paper values are transcribed exactly, including the known MobileNetV2 Full FT saved-state cells (16.12 and 81.91) and SG optimizer cell (0.16). They are references, not targets or validated corrections.",
               "", "CSV deltas are Jetson minus the PRINTED, rounded paper value, not differences from the original unrounded experiment. The paper uses historical reference-counting/single-buffer estimates; Jetson uses distinct concurrent backing allocations. Differences are not automatically hardware effects.",
               "", "Paper rank is 2. Missing, failed, rank-mismatched or unverified LoRA-Edge cells have no numerical comparison. Partial sweeps leave the other cells PENDING. Optimizer is marked INCOMPLETE until all four batches of a row are verified.",
               "", "All 40 cells are retained in paper order even for a subset run. Non-paper batch sizes are available in table3.md and measurements.csv, not this comparison."])
    (out / "table3_comparison.md").write_text("\n".join(md) + "\n")
    fields = ["backbone", "method", "batch", "status", "paper_source", "paper_peak_mb", "paper_saved_mb", "paper_optimizer_mb",
              "jetson_peak_bytes", "jetson_saved_bytes", "jetson_optimizer_bytes", "jetson_peak_mb", "jetson_saved_mb", "jetson_optimizer_mb",
              "delta_peak_mb", "delta_saved_mb", "delta_optimizer_mb"]
    dump_csv(out / "table3_comparison.csv", rows, fields)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.rank < 1 or args.warmup < 2 or args.threads < 1 or any(b < 1 for b in args.batches):
        p.error("Positive rank/batches/threads and at least two warm-up steps are required")
    if args.max_events < 1000:
        p.error("--max-events must be at least 1000")
    for name in ("models", "methods", "batches"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            p.error(f"Duplicate --{name} entries")
    if args.preflight:
        try:
            runtime_preflight(args)
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.worker_model:
        if not args.worker_method or not args.worker_batch or args.worker_output is None:
            p.error("Incomplete worker arguments")
        try:
            result = measure(args)
        except Exception as exc:
            traceback.print_exc()
            result = {"status": "failed", "config": case_config(args), "error": f"{type(exc).__name__}: {exc}"}
        write_json(args.worker_output, result)
        return 0 if result["status"] == "ok" else 1
    cases = sweep_cases(args)
    if args.dry_run:
        for i, (model, method, batch) in enumerate(cases, 1):
            print(f"{i:02d}/{len(cases)} {MODELS[model][0]} | {METHODS[method][0]} | rank {args.rank} | batch {batch}")
        print("No model execution, accuracy evaluation, or output files.")
        return 0
    if not (args.repo_root / "src" / "methods.py").is_file():
        p.error("Run from the MemFLoRA checkout or specify --repo-root")
    out = args.out or args.repo_root / "runs" / ("jetson_table3_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out = out.resolve()
    if out.exists():
        p.error(f"Output already exists; choose a new path: {out}")
    if not check_environment(args):
        return 1
    out.mkdir(parents=True)
    case_dir = out / "cases"
    case_dir.mkdir()
    write_json(out / "sweep.json", {"schema": SCHEMA, "command": sys.argv,
                                   "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                   "cases": cases, "order": "one serial pass; no reverse-order repetition"})
    results = []
    export_results(out, args, results)
    print(f"Output: {out}\n{len(cases)} isolated memory-only cases; no pretraining or accuracy evaluation.", flush=True)
    for i, (model, method, batch) in enumerate(cases, 1):
        stem = f"{model}_{method}_b{batch}_r{args.rank}"
        target = case_dir / (stem + ".json")
        command = [sys.executable, str(Path(__file__).resolve()), "--repo-root", str(args.repo_root.resolve()),
                   "--rank", str(args.rank), "--seed", str(args.seed), "--warmup", str(args.warmup),
                   "--device", args.device, "--threads", str(args.threads), "--max-events", str(args.max_events),
                   "--worker-model", model, "--worker-method", method, "--worker-batch", str(batch), "--worker-output", str(target)]
        for flag in ("checkpoint_copy", "trace_stacks", "save_snapshots", "allow_bitpack_fallback"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        print(f"[{i}/{len(cases)}] {MODELS[model][0]} / {METHODS[method][0]} / B={batch}", flush=True)
        with (case_dir / (stem + ".log")).open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if target.exists():
            result = json.loads(target.read_text())
        else:
            result = {"status": "failed", "config": {"model": model, "method": method, "batch": batch, "rank": args.rank},
                      "error": f"Worker exited {completed.returncode} without a result; inspect {stem}.log"}
        if completed.returncode and result["status"] == "ok":
            result["status"] = "failed"
            result["error"] = f"Worker exited {completed.returncode} after writing result"
        results.append(result)
        export_results(out, args, results)
        if result["status"] == "ok":
            m = result["metrics"]
            print(f'  {m["training_state_peak_bytes"]/1e6:.6f} / {m["saved_backward_end_forward_bytes"]/1e6:.6f} MB; optimizer {m["optimizer_total_bytes"]/1e6:.6f} MB', flush=True)
        else:
            detail = result.get("error") or ", ".join(k for k, v in result.get("checks", {}).items() if not v)
            print(f"  {result['status'].upper()}: {detail}; inspect {case_dir / (stem + '.log')}", flush=True)
    failures = sum(r["status"] != "ok" for r in results)
    print(f"Finished: {len(results)-failures}/{len(cases)} valid cases. Table: {out / 'table3.md'}", flush=True)
    print(f"Paper comparison: {out / 'table3_comparison.md'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
