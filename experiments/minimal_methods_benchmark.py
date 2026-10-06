"""Run the paper's dataset, method, rank, and adaptation sweeps."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.benchmark_adabn import calibrate_minimal_adabn_if_needed
from experiments.benchmark_cli import (
    apply_dataset_defaults,
    create_minimal_benchmark_dir,
    parse_adabn_calib_batch_values,
    parse_args,
    resolve_minimal_targets,
)
from experiments.benchmark_common import (
    BNPA_ADABN_AUDIT_METHODS,
    BNPA_SG_METHODS,
    RANK_FREE_METHODS,
    build_bnpa_sg_configs,
    format_float_label,
    method_forces_bn_eval,
    parse_float_list,
    parse_int_list,
)
from experiments.benchmark_reporting import (
    append_result,
    make_display_method,
    make_result_row,
    profile_trial,
    result_fieldnames,
    write_benchmark_config,
    write_benchmark_summary,
)
from experiments.benchmark_training import (
    adaptation_clock,
    evaluate,
    fit_epochs_best,
    fit_steps_best,
)
from src.data.benchmark import (
    build_minimal_split,
    load_minimal_dataset_arrays,
    make_loaders,
    minimal_target_key,
)
from src.models import OfficialTResNet2D, TorchvisionMobileNetV2HAR
from src.models.adapter_injection import inject_adapters

TIME_SPEC_RATIO = 0.85


@dataclass
class Trial:
    args: argparse.Namespace
    loaders: dict
    label: str
    sg_config: object = None


def build_backbone(args):
    if args.backbone == "mobilenet_v2":
        return TorchvisionMobileNetV2HAR(
            num_classes=args.expected_num_classes,
            pretrained=args.mobilenet_v2_pretrained,
            width_mult=args.width_mult,
        )
    return OfficialTResNet2D(
        input_channels=args.expected_input_channels,
        num_classes=args.expected_num_classes,
        n_feature_maps=args.t_resnet_feature_maps,
    )


def configure_method(model, args):
    method = args.method
    if method in ("zero_shot", "full"):
        for parameter in model.parameters():
            parameter.requires_grad_(method == "full")
        layers = []
    else:
        layers = inject_adapters(
            model,
            method=method,
            rank=args.rank,
            backbone="t_resnet" if args.backbone == "t_resnet_official" else args.backbone,
            adapter_layers=args.adapter_layers,
            bnpa_bottleneck_bn=args.bnpa_bottleneck_bn,
        )
    if getattr(args, "runtime_mode", "reference") == "optimized":
        from src.runtime import configure_runtime_optimizations
        configure_runtime_optimizations(model)
    return layers


def iter_trials(args, method, split, loaders, target_domain):
    """Preserve the original nesting and loader construction order for seeded runs."""
    bn_methods = ("bnpa", "bnpa_q3init", "bnpa_sg", "bnpa_sg_q3init")
    bn_values = args.bnpa_bottleneck_bn if method in bn_methods else ["on"]
    proj_values = args.proj_init if method in ("bnpa", "bnpa_sg") else ["random"]
    sg_configs = build_bnpa_sg_configs(args) if method in BNPA_SG_METHODS else [None]
    calib_values = parse_adabn_calib_batch_values(args.adabn_calib_batches)
    if method not in BNPA_ADABN_AUDIT_METHODS:
        calib_values = calib_values[:1]
    lr_values = (
        args.adapt_lr if method not in ("zero_shot", "full") else args.adapt_lr[:1]
    )
    # Full/zero-shot trials historically repeat across ranks. Keep their RNG consumption.
    ranks = [0] if method in RANK_FREE_METHODS else args.rank
    for proj, bn, sg in product(proj_values, bn_values, sg_configs):
        base = make_display_method(method, bn, len(bn_values), len(proj_values), sg)
        for calib, lr in product(calib_values, lr_values):
            label = base
            if method in BNPA_ADABN_AUDIT_METHODS and (
                calib != 1 or len(calib_values) > 1
            ):
                label += f"_calib{'all' if calib is None else calib}"
            if len(lr_values) > 1:
                label += f"_lr{format_float_label(lr)}"
            for batch_size in args.batch_size_values:
                batch_loaders = (
                    loaders
                    if batch_size == args.batch_size
                    else make_loaders(split, batch_size, args.seed)
                )
                batch_label = label + (
                    f"_bs{batch_size}" if len(args.batch_size_values) > 1 else ""
                )
                for rank, steps in product(ranks, args.steps_adapt):
                    trial_args = argparse.Namespace(
                        **{
                            **vars(args),
                            "batch_size": batch_size,
                            "method": method,
                            "rank": rank,
                            "steps_adapt": steps,
                            "adapt_lr": lr,
                            "proj_init": proj,
                            "bnpa_bottleneck_bn": bn,
                            "adabn_calib_batches": calib,
                            "target_domain": str(target_domain),
                        }
                    )
                    yield Trial(trial_args, batch_loaders, batch_label, sg)


def run_trial(
    trial,
    source_state,
    source_metrics,
    zero_shot,
    device,
    result_dir,
    full_references,
    fields,
):
    args, loaders = trial.args, trial.loaders
    method = args.method
    model = build_backbone(args).to(device)
    model.load_state_dict(source_state)
    layers = configure_method(model, args)
    calibrate_minimal_adabn_if_needed(
        model, method, loaders["Shift_train"], args, device
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    reference_key = (args.target_domain, args.seed, args.batch_size, args.steps_adapt)
    threshold = None
    if args.profile_time_spec and method not in ("zero_shot", "full"):
        threshold = TIME_SPEC_RATIO * full_references[reference_key]
    elapsed = 0.0
    if method == "zero_shot":
        metrics, target = {**zero_shot, "step": 0}, zero_shot
    else:
        start = adaptation_clock(device)
        metrics = fit_steps_best(
            model,
            loaders["Shift_train"],
            loaders["Shift_test"],
            args.steps_adapt,
            args.full_adapt_lr if method == "full" else args.adapt_lr,
            args.adapt_weight_decay,
            args.adapt_eval_every_steps,
            device,
            args.backbone,
            force_bn_eval=method_forces_bn_eval(method),
            sg_config=trial.sg_config,
            time_spec_threshold_macro_f1=threshold,
            time_spec_start_time=start if args.profile_time_spec else None,
        )
        elapsed = adaptation_clock(device) - start
        target = evaluate(model, loaders["Shift_test"], device, args.backbone)
    if args.profile_time_spec and method == "full":
        full_references.setdefault(reference_key, target["macro_f1"])
    row = make_result_row(
        args,
        trial.label,
        source_metrics,
        metrics,
        zero_shot,
        target,
        layers,
        trainable,
        total,
    )
    row["adaptation_time_sec"] = elapsed
    append_result(result_dir / "minimal_benchmark_results.csv", row, fields)
    profile_trial(model, loaders, args, trial.label, result_dir)
    return row


def main():
    args = parse_args()
    apply_dataset_defaults(args)
    args.steps_adapt = parse_int_list(args.steps_adapt, "--steps-adapt")
    seeds = parse_int_list(args.seed, "--seed")
    args.adapt_lr = parse_float_list(args.adapt_lr, "--adapt-lr")
    args.batch_size_values = parse_int_list(args.batch_size, "--batch-size")
    args.batch_size = args.batch_size_values[0]
    fields = result_fieldnames(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = create_minimal_benchmark_dir(
        Path(args.results_root), args.dataset, args.backbone
    )
    write_benchmark_config(result_dir, args)
    arrays = load_minimal_dataset_arrays(args, result_dir)
    targets = resolve_minimal_targets(args, arrays)
    rows, source_cache, full_references = [], {}, {}
    for seed in seeds:
        args.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for target_domain in targets:
            key = minimal_target_key(args, target_domain)
            split = build_minimal_split(args, arrays, target_domain, seed)
            loaders = make_loaders(split, args.batch_size, seed)
            source_model = build_backbone(args).to(device)
            cached = source_cache.get(key) if args.reuse_first_source_model else None
            if cached is not None:
                source_model.load_state_dict(cached[0])
                source_metrics = deepcopy(cached[1])
            else:
                source_metrics = fit_epochs_best(
                    source_model,
                    loaders["Pretrain_train"],
                    loaders["Pretrain_val"],
                    args.pretrain_epochs,
                    args.pretrain_lr,
                    args.pretrain_weight_decay,
                    device,
                    args.backbone,
                )
                if args.reuse_first_source_model:
                    source_cache[key] = (
                        {
                            k: v.detach().cpu().clone()
                            for k, v in source_model.state_dict().items()
                        },
                        deepcopy(source_metrics),
                    )
            source_state = deepcopy(source_model.state_dict())
            zero_shot = evaluate(
                source_model, loaders["Shift_test"], device, args.backbone
            )
            for method in args.method:
                for trial in iter_trials(args, method, split, loaders, target_domain):
                    rows.append(
                        run_trial(
                            trial,
                            source_state,
                            source_metrics,
                            zero_shot,
                            device,
                            result_dir,
                            full_references,
                            fields,
                        )
                    )
    write_benchmark_summary(result_dir / "minimal_benchmark_summary.csv", rows)
    print(f"Wrote results to {result_dir}")


if __name__ == "__main__":
    main()
