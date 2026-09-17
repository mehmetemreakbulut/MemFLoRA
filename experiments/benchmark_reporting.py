"""Compact trial results, summaries, and optional profiling output."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import math
import shlex
import sys

import numpy as np

from experiments.benchmark_common import (
    BNPA_SG_METHODS,
    RANK_FREE_METHODS,
    PreparedInputModel,
    format_float_label,
    method_forces_bn_eval,
)
from src.train import freeze_bn_eval
from src.utils.csv import write_csv

RESULT_FIELDNAMES = (
    "dataset",
    "backbone",
    "method",
    "base_method",
    "target_domain",
    "rank",
    "seed",
    "steps_adapt",
    "batch_size",
    "n_injected_layers",
    "trainable_params",
    "total_params",
    "eval_accuracy",
    "eval_macro_f1",
    "eval_weighted_f1",
    "zero_shot_macro_f1",
    "delta_macro_f1_vs_zero_shot",
    "source_val_macro_f1",
    "best_adapt_step",
    "adaptation_time_sec",
)
TIME_FIELDS = ("time_spec_reached", "time_spec_reach_step", "time_spec_reach_time_sec")
SG_FIELDS = ("bnpa_sg_updates", "bnpa_sg_sites_accumulated_mean")


def result_fieldnames(args):
    fields = RESULT_FIELDNAMES
    if args.profile_time_spec:
        fields += TIME_FIELDS
    if any(method in BNPA_SG_METHODS for method in args.method):
        fields += SG_FIELDS
    return fields


def make_result_row(
    args, label, source_metrics, metrics, zero_shot, target, layers, trainable, total
):
    row = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "method": label,
        "base_method": args.method,
        "target_domain": args.target_domain,
        "rank": (
            0 if args.method in RANK_FREE_METHODS | {"zero_shot", "full"} else args.rank
        ),
        "seed": args.seed,
        "steps_adapt": args.steps_adapt,
        "batch_size": args.batch_size,
        "n_injected_layers": len(layers),
        "trainable_params": trainable,
        "total_params": total,
        "eval_accuracy": target["accuracy"],
        "eval_macro_f1": target["macro_f1"],
        "eval_weighted_f1": target["weighted_f1"],
        "zero_shot_macro_f1": zero_shot["macro_f1"],
        "delta_macro_f1_vs_zero_shot": target["macro_f1"] - zero_shot["macro_f1"],
        "source_val_macro_f1": source_metrics.get("macro_f1", 0.0),
        "best_adapt_step": metrics["step"],
    }
    if args.profile_time_spec:
        row.update({field: metrics.get(field, "") for field in TIME_FIELDS})
    if args.method in BNPA_SG_METHODS:
        row.update({field: metrics.get(field, "") for field in SG_FIELDS})
    return row


def append_result(path, row, fieldnames):
    """Save each completed trial once, with a schema chosen before the run."""
    write_csv(path, [row], fieldnames, append=True)


def make_display_method(method, bottleneck_bn, bn_value_count, proj_value_count, sg):
    """Retain paper result labels for the supported initialization and SG settings."""
    if method not in ("bnpa", "bnpa_q3init", "bnpa_sg", "bnpa_sg_q3init", "bnpa_q3_sg"):
        return method
    label = method
    if (
        bottleneck_bn != "on"
        or bn_value_count > 1
        or proj_value_count > 1
        or method in ("bnpa_sg", "bnpa_q3_sg")
    ):
        label += "_random"
    if bottleneck_bn == "off":
        label += "_no_bn"
    if sg is not None:
        label += f"_sglr{format_float_label(sg.p_lr)}_u1_w0"
        if sg.p_weight_decay != 0.0:
            label += f"_sgwd{format_float_label(sg.p_weight_decay)}"
    return label


def profile_trial(model, loaders, args, label, result_dir):
    """Profile isolated copies; each profiler consumes its original loader batch."""
    metadata = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "method": label,
        "base_method": args.method,
        "target_domain": args.target_domain,
        "rank": args.rank,
        "seed": args.seed,
        "steps_adapt": args.steps_adapt,
        "batch_size": args.batch_size,
    }
    if args.profile_full_sram and args.method != "zero_shot":
        from src.profiling.sram import append_full_sram_rows, profile_full_sram_for_step

        x, y = next(iter(loaders["Shift_train"]))
        device = next(model.parameters()).device
        profile_model = deepcopy(model).to(device)
        profile_model.train()
        if method_forces_bn_eval(args.method):
            freeze_bn_eval(profile_model)
        row = profile_full_sram_for_step(
            PreparedInputModel(profile_model, args.backbone),
            x.to(device, non_blocking=True),
            y.to(device, non_blocking=True),
            metadata=metadata,
            force_bn_eval=method_forces_bn_eval(args.method),
        )
        append_full_sram_rows(result_dir / "full_sram_summary.csv", [row])
    if args.profile_performance:
        from src.profiling.performance import (
            append_performance_rows,
            profile_one_batch_performance,
        )

        x, _ = next(iter(loaders["Shift_train"]))
        device = next(model.parameters()).device
        wrapped = PreparedInputModel(deepcopy(model).to(device), args.backbone)
        row = profile_one_batch_performance(
            wrapped, x.to(device, non_blocking=True), metadata
        )
        append_performance_rows(result_dir / "performance_summary.csv", [row])


def write_benchmark_config(result_dir, args):
    payload = {"argv": sys.argv, "args": vars(args)}
    (result_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    command = shlex.join([sys.executable, *sys.argv])
    (result_dir / "command.txt").write_text(command + "\n")


def write_benchmark_summary(path, rows):
    """Pool trial observations using the existing paper mean/std/stderr formulas."""
    groups = defaultdict(list)
    keys = ("dataset", "backbone", "method", "rank", "steps_adapt", "batch_size")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    mean_fields = [
        "eval_accuracy",
        "delta_macro_f1_vs_zero_shot",
        "adaptation_time_sec",
        "best_adapt_step",
        "trainable_params",
        "total_params",
    ]
    if any("time_spec_reach_step" in row for row in rows):
        mean_fields.append("time_spec_reach_step")
    summaries = []
    for key, group in sorted(groups.items()):
        macro = np.array([row["eval_macro_f1"] for row in group], dtype=np.float64)
        std = float(macro.std(ddof=1)) if len(macro) > 1 else 0.0
        summary = {
            **dict(zip(keys, key)),
            "n_trials": len(group),
            "seeds": len({row["seed"] for row in group}),
            "eval_macro_f1_mean": float(macro.mean()),
            "eval_macro_f1_std": std,
            "eval_macro_f1_stderr": std / math.sqrt(len(macro)),
        }
        for field in mean_fields:
            values = [
                float(row[field]) for row in group if row.get(field) not in ("", None)
            ]
            summary[field + "_mean"] = float(np.mean(values)) if values else ""
        summaries.append(summary)
    write_csv(path, summaries)
