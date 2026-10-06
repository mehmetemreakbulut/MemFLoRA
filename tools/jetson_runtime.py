#!/usr/bin/env python3
"""Short isolated reference/optimized timing + operator trace, not a power sweep."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import jetson_power as power


def trace_coverage(payload):
    """Do not mislabel a CPU-only trace when CUPTI silently fails to start."""
    kernels = sum(
        'kernel' in event.get('cat', '').split(',')
        for event in payload.get('traceEvents', [])
    )
    return {
        'cuda_kernel_events': kernels,
        'cuda_trace_available': kernels > 0,
        'warning': None if kernels else (
            'No CUDA kernel events captured; trace is CPU-only. Check CUPTI '
            'permissions/support. Synchronized timing is independent of tracing.'
        ),
    }


def worker(args, out):
    settings = power.parse_args(['--runtime-mode', args.runtime_mode, '--batch', str(args.batch)])
    settings.worker_model, settings.worker_method = args.model, args.worker
    profile = power.shared(settings)
    torch, device = profile.prepare_runtime(settings)
    update, classes = power.build_workload(settings, torch, device)
    for _ in range(args.warmup):
        update()
    torch.cuda.synchronize(device)
    # This measures an already warmed allocator, not a cold process requirement.
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = update()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError('Non-finite update loss')
    stats = torch.cuda.memory_stats(device)
    info = profile.provenance(settings, torch, device)
    for name in ('jetson_runtime.py', 'jetson_power.py'):
        path = ROOT / 'tools' / name
        info['source_sha256']['tools/' + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    info['nvpmodel_query'] = power.read_command(['nvpmodel', '-q'])
    info['clocks_query'] = power.read_command(['jetson_clocks', '--show'])
    result = {'model': args.model, 'method': args.worker, 'runtime_mode': args.runtime_mode,
              'steps': args.steps, 'warmup_steps': args.warmup, 'batch': args.batch, 'rank': settings.rank,
              'ms_per_update': seconds * 1000 / args.steps,
              'requested_peak_bytes': stats['requested_bytes.all.peak'],
              'allocated_peak_bytes': stats['allocated_bytes.all.peak'],
              'reserved_peak_bytes': stats['reserved_bytes.all.peak'],
              'module_classes': classes, 'provenance': info,
              'note': 'Diagnostic only: timing excludes profiler, includes host work; no power sampling.'}
    if args.trace:
        # The instrumentation window is separate from the timing window.
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as trace:
            for _ in range(3):
                update()
            torch.cuda.synchronize(device)
        trace_path = out.with_suffix('.trace.json')
        trace.export_chrome_trace(str(trace_path))
        result['trace'] = trace_coverage(json.loads(trace_path.read_text()))
        has_cuda_trace = result['trace']['cuda_trace_available']
        if not has_cuda_trace:
            print('WARNING: ' + result['trace']['warning'], flush=True)
        with out.with_suffix('.operators.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['operator', 'calls', 'self_cpu_us', 'self_device_us'])
            events = sorted(
                trace.key_averages(),
                key=lambda e: e.self_device_time_total if has_cuda_trace else e.self_cpu_time_total,
                reverse=True,
            )
            writer.writerows(
                (e.key, e.count, e.self_cpu_time_total,
                 e.self_device_time_total if has_cuda_trace else None)
                for e in events
            )
    out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'{args.worker} / {args.runtime_mode}: {result["ms_per_update"]:.3f} ms/update, '
          f'{result["requested_peak_bytes"]/1e6:.3f} MB requested', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=power.MODELS, default='tresnet')
    parser.add_argument('--methods', nargs='+', choices=power.METHODS, default=['full', 'memflora'])
    parser.add_argument('--modes', nargs='+', choices=['reference', 'optimized'], default=['reference', 'optimized'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--trace', action='store_true', help='Save three-update CPU/CUDA operator traces separately')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--runtime-mode', choices=['reference', 'optimized'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 2 or args.steps < 1 or args.batch < 1:
        parser.error('Require warmup >= 2 and positive batch/steps')
    if args.worker:
        worker(args, args.out)
        return
    out = args.out or ROOT / 'runs' / time.strftime('jetson_runtime_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    for mode in args.modes:
        for method in args.methods:
            command = [sys.executable, __file__, '--worker', method, '--runtime-mode', mode,
                       '--model', args.model, '--batch', str(args.batch), '--warmup', str(args.warmup),
                       '--steps', str(args.steps), '--out', str(out / f'{method}_{mode}.json')]
            if args.trace:
                command.append('--trace')
            subprocess.run(command, check=True)
    print(f'Diagnostics: {out.resolve()}')


if __name__ == '__main__':
    main()
