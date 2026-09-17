"""Protect trial identity, paper CLI settings, and checkpoint selection."""

import argparse
import csv
import json
import shlex
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.benchmark_cli import apply_dataset_defaults, parse_args
from experiments.benchmark_reporting import (
    append_result,
    make_result_row,
    result_fieldnames,
    write_benchmark_config,
    write_benchmark_summary,
)
from experiments import benchmark_training


@pytest.mark.parametrize(
    "dataset,shape",
    [
        ("opportunity", (97, 17, 60, 30)),
        ("realdisp", (117, 33, 250, 125)),
        ("realworld", (9, 8, 500, 250)),
    ],
)
def test_dataset_defaults_and_explicit_window(dataset, shape):
    args = parse_args(["--dataset", dataset])
    apply_dataset_defaults(args)
    assert (
        args.expected_input_channels,
        args.expected_num_classes,
        args.window_size,
        args.window_stride,
    ) == shape
    explicit = parse_args(
        ["--dataset", dataset, "--data-root", "custom path", "--window-size", "16"]
    )
    apply_dataset_defaults(explicit)
    assert explicit.data_root == "custom path"
    assert explicit.window_size == 16


@pytest.mark.parametrize(
    "flags",
    [
        ["--proj-init", "target_pca"],
        ["--bnpa-sg-p-optimizer", "sgd"],
        ["--bnpa-sg-p-update-every", "2"],
        ["--profile-time-spec", "true", "--method", "bnpa", "full"],
        ["--profile-time-spec", "true", "--method", "bnpa"],
    ],
)
def test_unsupported_or_unordered_experiments_fail_early(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)


def test_csv_retains_domains_batches_and_pooled_observation_counts(tmp_path):
    args = parse_args(["--dataset", "realworld", "--method", "full"])
    fields = result_fieldnames(args)
    assert len(fields) == len(set(fields)) == 20
    assert not any(field.startswith(("bnpa_sg_", "time_spec_")) for field in fields)
    rows = []
    for batch, domain, seed, score in [
        (16, "chest", 1, 0.6),
        (16, "head", 1, 0.8),
        (32, "chest", 2, 0.9),
    ]:
        trial_args = argparse.Namespace(
            **{
                **vars(args),
                "method": "full",
                "batch_size": batch,
                "target_domain": domain,
                "seed": seed,
                "steps_adapt": 50,
                "rank": 2,
            }
        )
        metrics = {
            "accuracy": score,
            "macro_f1": score,
            "weighted_f1": score,
            "step": 50,
        }
        row = make_result_row(
            trial_args, "full", metrics, metrics, metrics, metrics, [], 10, 10
        )
        row["adaptation_time_sec"] = 1.0
        append_result(tmp_path / "results.csv", row, fields)
        rows.append(row)
    with (tmp_path / "results.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    assert [r["dataset"] for r in saved] == ["realworld"] * 3
    assert [r["target_domain"] for r in saved] == ["chest", "head", "chest"]
    write_benchmark_summary(tmp_path / "summary.csv", rows)
    with (tmp_path / "summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    assert [r["batch_size"] for r in summary] == ["16", "32"]
    assert summary[0]["n_trials"] == "2" and summary[0]["seeds"] == "1"
    assert float(summary[0]["eval_macro_f1_mean"]) == pytest.approx(0.7)
    assert float(summary[0]["eval_macro_f1_stderr"]) == pytest.approx(0.1)


def test_recorded_command_preserves_shell_arguments(tmp_path, monkeypatch):
    argv = [
        "experiments/minimal_methods_benchmark.py",
        "--data-root",
        "data with spaces/$literal",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    write_benchmark_config(tmp_path, argparse.Namespace(data_root=argv[-1]))
    assert shlex.split((tmp_path / "command.txt").read_text()) == [
        sys.executable,
        *argv,
    ]
    assert json.loads((tmp_path / "config.json").read_text())["argv"] == argv


def test_best_checkpoint_and_first_threshold_crossing(monkeypatch):
    model = nn.Linear(3, 2)
    loader = DataLoader(
        TensorDataset(torch.ones(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    checkpoints = []
    evaluations = iter([(1.0, 0.2), (0.5, 0.9), (0.8, 0.4)])

    def evaluate(model, *_):
        checkpoints.append(
            {key: value.detach().clone() for key, value in model.state_dict().items()}
        )
        loss, score = next(evaluations)
        return {
            "loss": loss,
            "accuracy": score,
            "macro_f1": score,
            "weighted_f1": score,
        }

    monkeypatch.setattr(benchmark_training, "evaluate", evaluate)
    result = benchmark_training.fit_steps_best(
        model,
        loader,
        loader,
        3,
        0.001,
        0.0,
        1,
        torch.device("cpu"),
        "t_resnet_official",
        time_spec_threshold_macro_f1=0.85,
        time_spec_start_time=0.0,
    )
    assert result["step"] == result["time_spec_reach_step"] == 2
    assert result["time_spec_reached"] is True
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, checkpoints[1][key])
