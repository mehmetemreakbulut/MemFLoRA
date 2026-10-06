"""Command-line arguments and dataset defaults for the paper experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from experiments.benchmark_common import METHODS, parse_adabn_calib_batch_value
from src.data.realdisp import (
    REALDISP_NUM_CLASSES,
    canonical_scenario,
    realdisp_feature_indices,
)
from src.data.realworld import (
    REALWORLD_ACTIVITIES,
    REALWORLD_LOCATIONS,
    canonical_location,
    feature_names_for_feature_set,
)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MemFLoRA accuracy, ablation, and profiling experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=("opportunity", "realworld", "realdisp"),
        default="opportunity",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--results-root", default="experiments/results/minimal_ZO_experiments"
    )
    parser.add_argument(
        "--backbone",
        choices=("mobilenet_v2", "t_resnet_official"),
        default="t_resnet_official",
    )
    parser.add_argument("--target-subjects", nargs="+", default=["all"])
    parser.add_argument(
        "--target-locations",
        nargs="+",
        default=None,
        help="RealWorld LOLO locations; default: all.",
    )
    parser.add_argument(
        "--realworld-feature-set", choices=("acc3", "accgyro6", "imu9"), default="imu9"
    )
    parser.add_argument(
        "--realworld-split-strategy",
        choices=("chronological", "stratified_random"),
        default="chronological",
    )
    parser.add_argument("--source-scenario", default="ideal")
    parser.add_argument("--target-scenario", default="self")
    parser.add_argument(
        "--realdisp-feature-set",
        choices=("acc3", "accgyro6", "imu9", "imu13"),
        default="imu13",
    )
    parser.add_argument("--method", nargs="+", choices=METHODS, default=["bnpa"])
    parser.add_argument("--runtime-mode", choices=("reference", "optimized"), default="reference",
                        help="Opt-in runtime optimizations; changes runtime memory, not the method")
    parser.add_argument("--rank", "--ranks", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument(
        "--adapter-layers",
        default="all",
        choices=(
            "all",
            "last",
            "middle_last",
            "pointwise_only",
            "depthwise_only",
            "stem_only",
            "depthwise_excluded",
        ),
    )
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--label-column", default="ml_both_arms")
    parser.add_argument(
        "--window-label-rule",
        choices=("majority", "center", "last"),
        default="majority",
    )
    parser.add_argument("--seed", nargs="+", default=["1"])
    parser.add_argument(
        "--batch-size",
        nargs="+",
        type=int,
        default=[64],
        help="The first size is used for pretraining; all sizes are swept for adaptation.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--steps-adapt", nargs="+", default=["50"])
    parser.add_argument(
        "--adapt-lr",
        nargs="+",
        default=["1e-2"],
        help="LR sweep; full fine-tuning uses 0.001.",
    )
    parser.add_argument("--adapt-eval-every-steps", type=int, default=1)
    parser.add_argument(
        "--adabn-calibration-mode",
        choices=("ema_reset", "ema_no_reset"),
        default="ema_no_reset",
    )
    parser.add_argument(
        "--adabn-calib-batches",
        nargs="+",
        default=["all"],
        help="Initial BNPA target calibration: non-negative batch count, or all.",
    )
    # Keep the paper command spellings; reject alternatives the implementation does not support.
    parser.add_argument(
        "--adabn-stat-source", nargs="+", choices=("target",), default=["target"]
    )
    parser.add_argument(
        "--train-mode-adabn",
        "--train_mode_adaBN",
        dest="train_mode_adabn",
        nargs="+",
        choices=("off",),
        default=["off"],
    )
    parser.add_argument(
        "--proj-init",
        nargs="+",
        choices=("random",),
        default=["random"],
        help="Projection initialization is determined by the selected method.",
    )
    parser.add_argument(
        "--bnpa-bottleneck-bn", nargs="+", choices=("on", "off"), default=["on"]
    )
    parser.add_argument("--bnpa-sg-p-lr", nargs="+", default=["1e-3"])
    parser.add_argument("--bnpa-sg-p-weight-decay", nargs="+", default=["0"])
    parser.add_argument(
        "--bnpa-sg-p-optimizer", nargs="+", choices=("adam",), default=["adam"]
    )
    parser.add_argument(
        "--bnpa-sg-p-update-every", nargs="+", choices=("1",), default=["1"]
    )
    parser.add_argument(
        "--bnpa-sg-p-warmup-steps", nargs="+", choices=("0",), default=["0"]
    )
    parser.add_argument(
        "--bnpa-sg-renorm-p", nargs="+", choices=("none",), default=["none"]
    )
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--mobilenet-v2-pretrained", type=str_to_bool, default=True)
    parser.add_argument("--t-resnet-feature-maps", type=int, default=64)
    parser.add_argument(
        "--reuse-first-source-model",
        type=str_to_bool,
        default=False,
        help="Reuse each target's first pretrained source model across subsequent seeds.",
    )
    parser.add_argument("--profile-full-sram", type=str_to_bool, default=False)
    parser.add_argument("--profile-performance", type=str_to_bool, default=False)
    parser.add_argument(
        "--profile-time-spec",
        type=str_to_bool,
        default=False,
        help="Track the first checkpoint reaching 0.85 times the matching full-FT macro-F1.",
    )
    parser.set_defaults(
        adapt_weight_decay=0.0005,
        full_adapt_lr=0.001,
        pretrain_lr=0.001,
        pretrain_weight_decay=0.0005,
    )
    args = parser.parse_args(argv)
    if args.profile_time_spec:
        if "full" not in args.method:
            parser.error(
                "--profile-time-spec requires method full before the adaptation methods"
            )
        if any(
            method != "zero_shot" for method in args.method[: args.method.index("full")]
        ):
            parser.error("full must precede adaptation methods for --profile-time-spec")
        if args.adapt_eval_every_steps != 1:
            print(
                f"Timing uses evaluation checkpoints every {args.adapt_eval_every_steps} steps."
            )
    return args


def apply_dataset_defaults(args):
    defaults = {
        "opportunity": (60, 30),
        "realworld": (500, 250),
        "realdisp": (250, 125),
    }
    window, stride = defaults[args.dataset]
    if args.data_root is None:
        args.data_root = f"data/{args.dataset}"
    if args.window_size is None:
        args.window_size = window
    if args.window_stride is None:
        args.window_stride = stride
    if args.dataset == "opportunity":
        args.expected_input_channels, args.expected_num_classes = 97, 17
    elif args.dataset == "realworld":
        args.expected_input_channels = len(
            feature_names_for_feature_set(args.realworld_feature_set)
        )
        args.expected_num_classes = len(REALWORLD_ACTIVITIES)
    else:
        args.source_scenario = canonical_scenario(args.source_scenario)
        args.target_scenario = canonical_scenario(args.target_scenario)
        args.expected_input_channels = len(
            realdisp_feature_indices(args.realdisp_feature_set)
        )
        args.expected_num_classes = REALDISP_NUM_CLASSES
    if args.dataset != "opportunity" and args.target_subjects != ["all"]:
        print(
            f"WARNING: {args.dataset} ignores --target-subjects; it uses location/scenario domains."
        )


def resolve_minimal_targets(args, arrays):
    if args.dataset == "realdisp":
        return [canonical_scenario(args.target_scenario)]
    if args.dataset == "realworld":
        available = [
            REALWORLD_LOCATIONS[int(i)] for i in torch.unique(arrays.location_indices)
        ]
        requested = args.target_locations or ["all"]
        return (
            available
            if "all" in requested
            else [canonical_location(item) for item in requested]
        )
    available = sorted(int(i) for i in torch.unique(arrays.subjects))
    return (
        available
        if "all" in args.target_subjects
        else [int(item) for item in args.target_subjects]
    )


def parse_adabn_calib_batch_values(values):
    values = values if isinstance(values, (list, tuple)) else [values]
    return [parse_adabn_calib_batch_value(value) for value in values]


def create_minimal_benchmark_dir(
    results_root: Path, dataset: str, backbone: str
) -> Path:
    base = results_root / dataset / backbone / datetime.now().strftime("%d-%m-%Y")
    base.mkdir(parents=True, exist_ok=True)
    indices = [
        int(path.name[3:]) for path in base.glob("Exp*") if path.name[3:].isdigit()
    ]
    path = base / f"Exp{max(indices, default=0) + 1}"
    path.mkdir(exist_ok=False)
    return path
