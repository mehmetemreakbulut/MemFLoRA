"""Shared train/test split helpers used by every HAR loader."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


def stratified_target_split(
    target_indices: torch.Tensor,
    labels: torch.Tensor,
    target_adapt_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    adapt_parts = []
    eval_parts = []
    unique_labels = torch.unique(labels[target_indices])
    can_stratify = True
    for label in unique_labels:
        class_indices = target_indices[labels[target_indices] == label]
        if int(class_indices.numel()) < 2:
            can_stratify = False
            break
        permutation = torch.randperm(int(class_indices.numel()), generator=generator)
        shuffled = class_indices[permutation]
        adapt_count = int(round(float(class_indices.numel()) * target_adapt_ratio))
        adapt_count = min(max(adapt_count, 1), int(class_indices.numel()) - 1)
        adapt_parts.append(shuffled[:adapt_count])
        eval_parts.append(shuffled[adapt_count:])
    if can_stratify and adapt_parts and eval_parts:
        adapt_indices = torch.cat(adapt_parts)
        eval_indices = torch.cat(eval_parts)
        return _shuffle_indices(adapt_indices, generator), _shuffle_indices(
            eval_indices, generator
        )
    permutation = torch.randperm(int(target_indices.numel()), generator=generator)
    shuffled = target_indices[permutation]
    adapt_count = int(round(float(target_indices.numel()) * target_adapt_ratio))
    adapt_count = min(max(adapt_count, 1), int(target_indices.numel()) - 1)
    return shuffled[:adapt_count], shuffled[adapt_count:]


def _shuffle_indices(indices: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    permutation = torch.randperm(int(indices.numel()), generator=generator)
    return indices[permutation]


def chronological_window_split(
    sequence_ids, target_indices, target_adapt_ratio: float, sort_key, refine=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split each target sequence in time: its earliest windows adapt, the rest evaluate.
    `sort_key` orders windows within a sequence. `refine` may return an exact
    (adapt, eval) pair for a sequence -- RealDisp uses it to cut on the segment
    boundary rather than on a window count -- and falls back when it cannot.
    """
    adapt: List[int] = []
    eval_: List[int] = []
    target_set = {int(index.item()) for index in target_indices}
    for sequence_id in sorted(
        {int(sequence_ids[index].item()) for index in target_indices}
    ):
        indices = sorted(
            (i for i in target_set if int(sequence_ids[i].item()) == sequence_id),
            key=sort_key,
        )
        if len(indices) == 1:
            adapt.extend(indices)
            continue
        if refine is not None:
            pair = refine(indices)
            if pair is not None:
                adapt.extend(pair[0])
                eval_.extend(pair[1])
                continue
        split = min(
            max(int(math.floor(len(indices) * target_adapt_ratio)), 1), len(indices) - 1
        )
        adapt.extend(indices[:split])
        eval_.extend(indices[split:])
    if not adapt or not eval_:
        raise ValueError(
            "Chronological target split produced an empty adaptation or evaluation split"
        )
    return torch.tensor(adapt, dtype=torch.long), torch.tensor(eval_, dtype=torch.long)


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    """Fill non-finite samples per channel by linear interpolation over the finite ones.

    Channels with nothing finite become zero. `np.interp` holds the first and last
    finite values constant beyond their range, so no gap survives.
    """
    filled = values.astype(np.float32, copy=True)
    positions = np.arange(filled.shape[0])
    for column in range(filled.shape[1]):
        finite = np.isfinite(filled[:, column])
        if finite.all():
            continue
        if finite.any():
            filled[:, column] = np.interp(
                positions, positions[finite], filled[finite, column]
            ).astype(np.float32)
        else:
            filled[:, column] = 0.0
    return filled


def slide_windows(segments, window_size: int, window_stride: int, dataset: str):
    """Cut every labelled segment into fixed windows, keeping per-window bookkeeping.

    `segments` yields `(values, columns)` pairs, where `values` is time-major and
    `columns` maps a field name to that segment's constant value. Returns the
    stacked channel-major windows, one long tensor per bookkeeping field, and each
    window's start offset within its own segment.
    """
    windows: List[np.ndarray] = []
    fields: dict = {}
    starts: List[int] = []
    for values, columns in segments:
        if values.shape[0] < window_size:
            continue
        for start in range(0, values.shape[0] - window_size + 1, window_stride):
            windows.append(values[start : start + window_size].T.astype(np.float32))
            for key, value in columns.items():
                fields.setdefault(key, []).append(value)
            starts.append(start)
    if not windows:
        raise ValueError(f"{dataset} preprocessing produced no windows")
    stacked = torch.tensor(np.stack(windows, axis=0), dtype=torch.float32)
    columns = {
        key: torch.tensor(values, dtype=torch.long) for key, values in fields.items()
    }
    return stacked, columns, torch.tensor(starts, dtype=torch.long)


@dataclass
class SplitTensors:
    pretrain: tuple[torch.Tensor, torch.Tensor]
    pretest: tuple[torch.Tensor, torch.Tensor]
    shift_train: tuple[torch.Tensor, torch.Tensor]
    shift_test: tuple[torch.Tensor, torch.Tensor]


def train_test_indices(
    y: np.ndarray, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return stratified_train_test_indices(y, test_size=test_size, seed=seed)
    except ValueError:
        rng = np.random.default_rng(seed)
        indices = np.arange(y.shape[0])
        rng.shuffle(indices)
        n_test = min(len(indices) - 1, max(1, int(np.ceil(len(indices) * test_size))))
        return indices[n_test:], indices[:n_test]


def stratified_train_test_indices(
    y: np.ndarray, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for label in np.unique(y):
        label_idx = np.flatnonzero(y == label)
        rng.shuffle(label_idx)
        n_test = min(
            len(label_idx) - 1, max(1, int(np.ceil(len(label_idx) * test_size)))
        )
        test_parts.append(label_idx[:n_test])
        train_parts.append(label_idx[n_test:])
    train_idx = np.concatenate(train_parts, axis=0)
    test_idx = np.concatenate(test_parts, axis=0)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx
