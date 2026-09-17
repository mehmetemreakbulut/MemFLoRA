"""Dataset loading, source/target splits and benchmark DataLoaders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.opportunity import load_opportunity_windows
from src.data.realdisp import (
    REALDISP_NUM_CLASSES,
    build_realdisp_windows,
    canonical_scenario,
    chronological_target_split as realdisp_target_split,
)
from src.data.realworld import (
    REALWORLD_ACTIVITIES,
    REALWORLD_LOCATIONS,
    build_realworld_windows,
    canonical_location,
    chronological_target_split as realworld_target_split,
)
from src.data.splits import SplitTensors, stratified_target_split, train_test_indices

SPLIT_TEST_SIZE = 0.2


def load_minimal_dataset_arrays(args: argparse.Namespace, result_dir: Path):
    options = dict(
        root=args.data_root,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )
    if args.dataset == "opportunity":
        arrays = load_opportunity_windows(
            **options,
            label_column=args.label_column,
            window_label_rule=args.window_label_rule,
        )
        channels = int(args.expected_input_channels)
        arrays.windows = arrays.windows[:, :channels, :].contiguous()
        arrays.feature_names = arrays.feature_names[:channels]
        arrays.feature_indices = arrays.feature_indices[:channels]
        name = "opportunity_official"
        label_mapping = arrays.raw_label_mapping
        label_column = arrays.label_column_name
    elif args.dataset == "realworld":
        arrays = build_realworld_windows(
            **options, feature_set=args.realworld_feature_set
        )
        name = "realworld"
        label_mapping = {activity: i for i, activity in enumerate(REALWORLD_ACTIVITIES)}
        label_column = "realworld_activity_label"
    elif args.dataset == "realdisp":
        arrays = build_realdisp_windows(
            **options, feature_set=args.realdisp_feature_set
        )
        name = "realdisp"
        label_mapping = {f"A{i + 1}": i for i in range(REALDISP_NUM_CLASSES)}
        label_column = "realdisp_activity_label"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    metadata = {
        "input_channels": int(arrays.windows.shape[1]),
        "window_size": int(arrays.windows.shape[2]),
        "num_classes": len(label_mapping),
        "label_mapping": label_mapping,
        "label_column_name": label_column,
        "window_label_rule": args.window_label_rule,
        "normalization": "none",
    }
    if args.dataset == "opportunity":
        metadata["feature_indices"] = arrays.feature_indices
    (result_dir / f"{name}_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return arrays


def _pair(arrays, indices):
    return (
        arrays.windows[indices].float().contiguous(),
        arrays.labels[indices].long().contiguous(),
    )


def build_official_split(
    arrays, target_subject: int, test_size: float, seed: int
) -> SplitTensors:
    """Split every subject independently, then join source subjects in ID order."""
    pretrain_x, pretrain_y, pretest_x, pretest_y = [], [], [], []
    shift_train = shift_test = None
    for subject in sorted(int(s.item()) for s in torch.unique(arrays.subjects)):
        indices = torch.nonzero(arrays.subjects == subject, as_tuple=False).flatten()
        train, test = train_test_indices(
            arrays.labels[indices].cpu().numpy(), test_size=test_size, seed=seed
        )
        train_pair, test_pair = _pair(arrays, indices[train]), _pair(
            arrays, indices[test]
        )
        if subject == target_subject:
            shift_train, shift_test = train_pair, test_pair
        else:
            pretrain_x.append(train_pair[0])
            pretrain_y.append(train_pair[1])
            pretest_x.append(test_pair[0])
            pretest_y.append(test_pair[1])
    if shift_train is None:
        raise ValueError(f"Opportunity target subject {target_subject} is absent")
    return SplitTensors(
        pretrain=(torch.cat(pretrain_x).contiguous(), torch.cat(pretrain_y)),
        pretest=(torch.cat(pretest_x).contiguous(), torch.cat(pretest_y)),
        shift_train=shift_train,
        shift_test=shift_test,
    )


def build_minimal_split(
    args: argparse.Namespace, arrays, target_domain: object, seed: int
) -> SplitTensors:
    if args.dataset == "opportunity":
        return build_official_split(arrays, int(target_domain), SPLIT_TEST_SIZE, seed)
    if args.dataset == "realworld":
        target = REALWORLD_LOCATIONS.index(canonical_location(str(target_domain)))
        source_mask = arrays.location_indices != target
        target_mask = arrays.location_indices == target
        chronological_split = realworld_target_split
        strategy = args.realworld_split_strategy
    elif args.dataset == "realdisp":
        source = arrays.scenario_mapping[canonical_scenario(args.source_scenario)]
        target = arrays.scenario_mapping[canonical_scenario(str(target_domain))]
        source_mask = arrays.scenario_indices == source
        target_mask = arrays.scenario_indices == target
        chronological_split = realdisp_target_split
        strategy = "chronological"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    source_indices = torch.nonzero(source_mask, as_tuple=False).flatten()
    target_indices = torch.nonzero(target_mask, as_tuple=False).flatten()
    train, test = train_test_indices(
        arrays.labels[source_indices].cpu().numpy(),
        test_size=SPLIT_TEST_SIZE,
        seed=seed,
    )
    target_adapt_ratio = 1.0 - SPLIT_TEST_SIZE
    if strategy == "chronological":
        adapt, evaluate = chronological_split(
            arrays, target_indices, target_adapt_ratio
        )
    elif strategy == "stratified_random":
        adapt, evaluate = stratified_target_split(
            target_indices, arrays.labels, target_adapt_ratio, seed
        )
    else:
        raise ValueError(f"Unsupported target split strategy: {strategy}")
    return SplitTensors(
        pretrain=_pair(arrays, source_indices[torch.tensor(train, dtype=torch.long)]),
        pretest=_pair(arrays, source_indices[torch.tensor(test, dtype=torch.long)]),
        shift_train=_pair(arrays, adapt),
        shift_test=_pair(arrays, evaluate),
    )


def make_loaders(
    split: SplitTensors, batch_size: int, seed: int
) -> dict[str, DataLoader]:
    # Share one generator between shuffled loaders, preserving the experiment's order.
    generator = torch.Generator().manual_seed(seed)
    pairs = {
        "Pretrain_train": (split.pretrain, True),
        "Pretrain_val": (split.pretest, False),
        "Shift_train": (split.shift_train, True),
        "Shift_test": (split.shift_test, False),
    }
    return {
        name: DataLoader(
            TensorDataset(*pair),
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )
        for name, (pair, shuffle) in pairs.items()
    }


def minimal_target_key(args: argparse.Namespace, target_domain: object) -> str:
    if args.dataset == "realworld":
        return f"target_location_{canonical_location(str(target_domain))}"
    if args.dataset == "realdisp":
        return f"target_scenario_{canonical_scenario(str(target_domain))}"
    return f"target_S{int(target_domain)}"
