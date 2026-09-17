from __future__ import annotations
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from src.data.splits import (
    chronological_window_split,
    interpolate_missing,
    slide_windows,
)
import numpy as np
import torch

REALDISP_NUM_CLASSES = 33
REALDISP_ACTIVITIES = [f"A{index}" for index in range(1, REALDISP_NUM_CLASSES + 1)]
REALDISP_LOG_COLUMNS = 120
REALDISP_LABEL_COLUMN = 119
REALDISP_FEATURE_START = 2
REALDISP_NUM_SENSORS = 9
REALDISP_MODALITIES_PER_SENSOR = 13
REALDISP_FEATURE_SET_OFFSETS = {
    "acc3": tuple(range(0, 3)),
    "accgyro6": tuple(range(0, 6)),
    "imu9": tuple(range(0, 9)),
    "imu13": tuple(range(0, 13)),
}
SCENARIO_ALIASES = {
    "ideal": "ideal",
    "idealplacement": "ideal",
    "ideal_placement": "ideal",
    "ideal-placement": "ideal",
    "scenario1": "ideal",
    "s1": "ideal",
    "self": "self",
    "selfplacement": "self",
    "self_placement": "self",
    "self-placement": "self",
    "scenario2": "self",
    "s2": "self",
    "induced": "induced",
    "induceddisplacement": "induced",
    "induced_displacement": "induced",
    "induced-displacement": "induced",
    "scenario3": "induced",
    "s3": "induced",
}


@dataclass
class RealDispSegment:
    subject: int
    scenario: str
    activity_class_zero_based: int
    values: np.ndarray
    sequence_id: int


@dataclass
class RealDispArrays:
    windows: torch.Tensor
    labels: torch.Tensor
    subjects: torch.Tensor
    scenario_indices: torch.Tensor
    sequence_ids: torch.Tensor
    starts: torch.Tensor
    segment_lengths: Dict[int, int]
    scenario_mapping: Dict[str, int]
    window_size: int
    window_stride: int


def build_realdisp_windows(
    root: Union[str, Path],
    window_size: Optional[int],
    window_stride: Optional[int],
    feature_set: str = "imu9",
) -> RealDispArrays:
    require_window_args(window_size, window_stride)
    segments, scenario_mapping = load_realdisp_segments(
        root=root, feature_set=feature_set
    )
    windows, columns, starts = slide_windows(
        (
            (
                segment.values,
                {
                    "labels": segment.activity_class_zero_based,
                    "subjects": segment.subject,
                    "scenario_indices": scenario_mapping[segment.scenario],
                    "sequence_ids": segment.sequence_id,
                },
            )
            for segment in segments
        ),
        int(window_size),
        int(window_stride),
        "RealDisp",
    )
    return RealDispArrays(
        windows=windows,
        starts=starts,
        segment_lengths={
            segment.sequence_id: len(segment.values) for segment in segments
        },
        scenario_mapping=scenario_mapping,
        window_size=int(window_size),
        window_stride=int(window_stride),
        **columns,
    )


def load_realdisp_segments(
    root: Union[str, Path], feature_set: str
) -> Tuple[List[RealDispSegment], Dict[str, int]]:
    paths = discover_realdisp_logs(root)
    segments: List[RealDispSegment] = []
    next_sequence_id = 0
    for path in paths:
        subject, scenario = infer_subject_scenario_from_log(path)
        if subject is None or scenario is None:
            raise ValueError(
                f"Could not infer RealDisp subject/scenario from filename: {path}"
            )
        try:
            matrix = read_realdisp_log(path)
            labels = coerce_labels(matrix[:, REALDISP_LABEL_COLUMN])
            feature_values = interpolate_missing(
                matrix[:, realdisp_feature_indices(feature_set)].astype(np.float32)
            )
            file_segments = make_segments_from_labels(
                subject=subject,
                scenario=scenario,
                features=feature_values,
                labels=labels,
                next_sequence_id=next_sequence_id,
            )
            next_sequence_id += len(file_segments)
            segments.extend(file_segments)
        except Exception as exc:
            raise ValueError(f"Could not parse RealDisp log {path}: {exc}") from exc
    if not segments:
        raise ValueError("No usable RealDisp segments were found")
    scenario_mapping = make_scenario_mapping([segment.scenario for segment in segments])
    return segments, scenario_mapping


def discover_realdisp_logs(root: Union[str, Path]) -> List[Path]:
    root = resolve_realdisp_root(root)
    paths = sorted(path for path in root.rglob("*.log") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No RealDisp .log files found under {root}")
    return paths


def resolve_realdisp_root(root: Union[str, Path]) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"RealDisp data root does not exist: {root}")
    candidates = [
        root,
        root / "REALDISP",
        root / "realdisp",
        root / "realdisp_activity_recognition_dataset",
        root / "REALDISP Activity Recognition Dataset",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.log")):
            return candidate
    return root


def infer_subject_scenario_from_log(path: Path) -> Tuple[Optional[int], Optional[str]]:
    match = re.match(r"subject_?0?(\d{1,2})_(.+)\.log$", path.name.lower())
    if not match:
        return None, None
    subject = int(match.group(1))
    scenario = canonical_scenario(match.group(2))
    return subject, scenario


def canonical_scenario(value: str) -> str:
    normalized = normalize_name(value)
    if re.fullmatch(r"mutual\d+", normalized):
        return normalized
    return SCENARIO_ALIASES.get(normalized, normalized)


def read_realdisp_log(path: Path) -> np.ndarray:
    """Use the NumPy parser used for the paper, independent of installed packages."""
    errors: List[str] = []
    for delimiter in (None, ",", ";"):
        try:
            matrix = np.genfromtxt(
                path, delimiter=delimiter, dtype=np.float32, invalid_raise=False
            )
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.ndim == 2 and matrix.shape[1] == REALDISP_LOG_COLUMNS:
                return matrix.astype(np.float32)
            errors.append(
                f"numpy delimiter={delimiter!r} produced shape={matrix.shape}"
            )
        except Exception as exc:
            errors.append(f"numpy delimiter={delimiter!r}: {exc}")
    raise ValueError(
        f"{path} is not a valid RealDisp log with exactly {REALDISP_LOG_COLUMNS} columns. "
        + " | ".join(errors[-4:])
    )


def realdisp_feature_indices(feature_set: str) -> List[int]:
    offsets = REALDISP_FEATURE_SET_OFFSETS[feature_set]
    indices: List[int] = []
    for sensor in range(REALDISP_NUM_SENSORS):
        base = REALDISP_FEATURE_START + sensor * REALDISP_MODALITIES_PER_SENSOR
        indices.extend(base + offset for offset in offsets)
    return indices


def coerce_labels(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(labels)
    labels = np.where(finite, labels, 0)
    return labels.astype(np.int64)


def make_segments_from_labels(
    subject: int,
    scenario: str,
    features: np.ndarray,
    labels: np.ndarray,
    next_sequence_id: int,
) -> List[RealDispSegment]:
    segments: List[RealDispSegment] = []
    start: Optional[int] = None
    current_label: Optional[int] = None
    for index, label_value in enumerate(labels.tolist()):
        label = int(label_value)
        if not (1 <= label <= REALDISP_NUM_CLASSES):
            if start is not None and current_label is not None:
                segments.append(
                    make_segment(
                        subject,
                        scenario,
                        features,
                        start,
                        index,
                        current_label,
                        next_sequence_id,
                    )
                )
                next_sequence_id += 1
            start = None
            current_label = None
            continue
        if start is None:
            start = index
            current_label = label
            continue
        if label != current_label:
            segments.append(
                make_segment(
                    subject,
                    scenario,
                    features,
                    start,
                    index,
                    int(current_label),
                    next_sequence_id,
                )
            )
            next_sequence_id += 1
            start = index
            current_label = label
    if start is not None and current_label is not None:
        segments.append(
            make_segment(
                subject,
                scenario,
                features,
                start,
                len(labels),
                int(current_label),
                next_sequence_id,
            )
        )
    return segments


def make_segment(
    subject: int,
    scenario: str,
    features: np.ndarray,
    start: int,
    end: int,
    label: int,
    sequence_id: int,
) -> RealDispSegment:
    return RealDispSegment(
        subject=subject,
        scenario=scenario,
        activity_class_zero_based=label - 1,
        values=features[start:end],
        sequence_id=sequence_id,
    )


def chronological_target_split(
    arrays: RealDispArrays, target_indices: torch.Tensor, target_adapt_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    def refine(indices):
        sequence_id = int(arrays.sequence_ids[indices[0]])
        boundary = int(
            math.floor(arrays.segment_lengths[sequence_id] * target_adapt_ratio)
        )
        adapt = [
            i for i in indices if int(arrays.starts[i]) + arrays.window_size <= boundary
        ]
        rest = [i for i in indices if int(arrays.starts[i]) >= boundary]
        return (adapt, rest) if adapt and rest else None

    return chronological_window_split(
        arrays.sequence_ids,
        target_indices,
        target_adapt_ratio,
        sort_key=lambda index: int(arrays.starts[index]),
        refine=refine,
    )


def make_scenario_mapping(scenarios: Sequence[str]) -> Dict[str, int]:
    ordered = sorted(set(scenarios), key=scenario_sort_key)
    return {scenario: index for index, scenario in enumerate(ordered)}


def scenario_sort_key(value: str) -> Tuple[int, object]:
    if value == "ideal":
        return (0, 0)
    if value == "self":
        return (1, 0)
    if value == "induced":
        return (2, 0)
    match = re.fullmatch(r"mutual(\d+)", value)
    if match:
        return (3, int(match.group(1)))
    return (4, value)


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def require_window_args(
    window_size: Optional[int], window_stride: Optional[int]
) -> None:
    if window_size is None or window_stride is None:
        raise ValueError(
            "RealDisp requires explicit --window-size and --window-stride. "
            "Use --window-size 250 --window-stride 125 for the current RealDisp setup."
        )
    if int(window_size) <= 0 or int(window_stride) <= 0:
        raise ValueError("window_size and window_stride must be positive")
