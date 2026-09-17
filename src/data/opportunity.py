from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from src.data.splits import interpolate_missing, normalize_column_name

DEFAULT_WINDOW_SIZE = 90
DEFAULT_WINDOW_STRIDE = 45
NULL_LABELS = {0}
DEFAULT_LABEL_COLUMN = "ml_both_arms"
DEFAULT_WINDOW_LABEL_RULE = "majority"


@dataclass
class OpportunityArrays:
    windows: torch.Tensor
    labels: torch.Tensor
    subjects: torch.Tensor
    raw_label_mapping: Dict[int, int]
    feature_names: List[str]
    feature_indices: List[int]
    label_column_index: int
    label_column_name: str
    window_label_rule: str


def windows_from_opportunity_file(
    matrix: np.ndarray,
    label_index: int,
    feature_indices: List[int],
    window_size: int,
    window_stride: int,
    window_label_rule: str,
) -> List[Tuple[np.ndarray, int]]:
    """Fixed-size windows from one recording, dropping unlabelled and NULL windows."""
    raw_labels = matrix[:, label_index]
    features = interpolate_missing(matrix[:, feature_indices].astype(np.float32))
    if features.shape[0] < window_size:
        return []
    out: List[Tuple[np.ndarray, int]] = []
    for start in range(0, features.shape[0] - window_size + 1, window_stride):
        end = start + window_size
        label = window_label_from_rule(raw_labels[start:end], window_label_rule)
        if label is None or label in NULL_LABELS:
            continue
        out.append((features[start:end].T, int(label)))
    return out


def load_opportunity_windows(
    root: Union[str, Path],
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: int = DEFAULT_WINDOW_STRIDE,
    label_column: str = DEFAULT_LABEL_COLUMN,
    window_label_rule: str = DEFAULT_WINDOW_LABEL_RULE,
) -> OpportunityArrays:
    root = Path(root)
    files = discover_opportunity_files(root)
    if not files:
        raise FileNotFoundError(
            f"No Opportunity .dat/.csv data files found under {root}. "
            "Expected files such as S1-ADL1.dat or S1-ADL1.csv."
        )
    global_column_names = load_column_names(root)
    windows: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[int] = []
    schema: Optional[Tuple[List[str], List[int], int, str]] = None
    for path in files:
        matrix, column_names = read_opportunity_matrix(path, global_column_names)
        label_index = select_label_column(
            matrix, column_names, label_column=label_column
        )
        feature_indices = select_wearable_feature_columns(
            matrix, column_names, label_index
        )
        if schema is None:
            name = (
                column_names[label_index]
                if column_names and label_index < len(column_names)
                else f"column_{label_index + 1}"
            )
            schema = (
                feature_names_from_indices(column_names, feature_indices),
                list(feature_indices),
                int(label_index),
                name,
            )
        for window, label in windows_from_opportunity_file(
            matrix,
            label_index,
            feature_indices,
            window_size,
            window_stride,
            window_label_rule,
        ):
            windows.append(window)
            labels.append(label)
            subjects.append(subject_from_path(path))
    if not windows or schema is None:
        raise ValueError("Opportunity preprocessing produced no labeled windows")
    label_mapping = {int(raw): index for index, raw in enumerate(sorted(set(labels)))}
    feature_names, feature_indices_reference, label_column_index, label_column_name = (
        schema
    )
    return OpportunityArrays(
        windows=torch.tensor(np.stack(windows, axis=0), dtype=torch.float32),
        labels=torch.tensor([label_mapping[int(l)] for l in labels], dtype=torch.long),
        subjects=torch.tensor(subjects, dtype=torch.long),
        raw_label_mapping=label_mapping,
        feature_names=feature_names,
        feature_indices=feature_indices_reference,
        label_column_index=label_column_index,
        label_column_name=label_column_name,
        window_label_rule=window_label_rule,
    )


def discover_opportunity_files(root: Path) -> List[Path]:
    files = []
    for suffix in ("*.dat", "*.csv", "*.txt"):
        for path in root.rglob(suffix):
            lower_name = path.name.lower()
            if any(
                token in lower_name for token in ("column", "legend", "readme", "label")
            ):
                continue
            if re.search(r"s\d+", lower_name):
                files.append(path)
    return sorted(files)


def subject_from_path(path: Path) -> int:
    match = re.search(r"s(\d+)", path.name.lower())
    if match is None:
        match = re.search(r"subject[_-]?(\d+)", str(path.parent).lower())
    if match is None:
        raise ValueError(f"Could not infer Opportunity subject ID from {path}")
    return int(match.group(1))


def read_opportunity_matrix(
    path: Path, global_column_names: Optional[List[str]]
) -> Tuple[np.ndarray, Optional[List[str]]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="") as handle:
            first_line = handle.readline()
        has_header = any(char.isalpha() for char in first_line)
        if has_header:
            structured = np.genfromtxt(
                path, delimiter=",", names=True, dtype=np.float32, encoding=None
            )
            names = list(structured.dtype.names or [])
            matrix = np.column_stack([structured[name] for name in names]).astype(
                np.float32
            )
            return matrix, names
        return np.loadtxt(path, delimiter=",", dtype=np.float32), global_column_names
    return np.loadtxt(path, dtype=np.float32), global_column_names


def load_column_names(root: Path) -> Optional[List[str]]:
    candidates = list(root.rglob("column_names.txt")) + list(root.rglob("columns.txt"))
    if not candidates:
        return None
    names = []
    for line in candidates[0].read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"(?:column[:\s]+)?(\d+)[:\s,-]+(.+)", line, flags=re.IGNORECASE
        )
        if match:
            names.append(match.group(2).strip())
    return names or None


def select_label_column(
    matrix: np.ndarray,
    column_names: Optional[List[str]],
    label_column: str = DEFAULT_LABEL_COLUMN,
) -> int:
    normalized_requested = normalize_column_name(label_column)
    if column_names and len(column_names) == matrix.shape[1]:
        normalized_names = [normalize_column_name(name) for name in column_names]
        aliases = label_aliases(normalized_requested)
        for alias in aliases:
            for index, name in enumerate(normalized_names):
                if alias == name or alias in name:
                    return index
        raise ValueError(
            f"Could not find requested Opportunity label column {label_column!r}"
        )
    if matrix.shape[1] >= 250:
        fallback_indices = {
            "locomotion": 243,
            "hl_activity": 244,
            "ll_left_arm": 245,
            "ll_left_arm_object": 246,
            "ll_right_arm": 247,
            "ll_right_arm_object": 248,
            "ml_both_arms": 249,
            "both_arms": 249,
        }
        return fallback_indices.get(normalized_requested, 249)
    return matrix.shape[1] - 1


OPPORTUNITY_IMU_RANGES = (
    (1, 37),
    (37, 50),
    (50, 63),
    (63, 76),
    (76, 89),
    (89, 102),
    (102, 134),
)


def named_wearable_columns(column_names: List[str], label_index: int) -> List[int]:
    """Columns whose name identifies them as a body-worn IMU channel."""
    label_indices = label_column_indices(column_names)
    return [
        index
        for index, name in enumerate(column_names)
        if index != label_index
        and index not in label_indices
        and not is_non_feature_name(name.lower())
        and is_wearable_name(name.lower())
    ]


def named_non_label_columns(column_names: List[str], label_index: int) -> List[int]:
    """Every named column that is not a label or bookkeeping column."""
    label_indices = label_column_indices(column_names)
    return [
        index
        for index, name in enumerate(column_names)
        if index != label_index
        and index not in label_indices
        and not is_non_feature_name(name.lower())
    ]


def positional_wearable_columns(n_columns: int, label_index: int) -> List[int]:
    """Body-worn channel positions in the standard Opportunity UCI layout, used when
    column names are missing or do not match the matrix width."""
    return [
        index
        for start, end in OPPORTUNITY_IMU_RANGES
        for index in range(start, min(end, n_columns))
        if index != label_index
    ]


def select_wearable_feature_columns(
    matrix: np.ndarray, column_names: Optional[List[str]], label_index: int
) -> List[int]:
    """First strategy that yields any column wins: named wearable, named non-label,
    positional layout, then everything but column 0 and the label."""
    if column_names and len(column_names) == matrix.shape[1]:
        for candidate in (
            named_wearable_columns(column_names, label_index),
            named_non_label_columns(column_names, label_index),
        ):
            if candidate:
                return candidate
    if matrix.shape[1] >= 250:
        candidate = positional_wearable_columns(matrix.shape[1], label_index)
        if candidate:
            return candidate
    return [index for index in range(matrix.shape[1]) if index not in (0, label_index)]


def is_non_feature_name(name: str) -> bool:
    stripped = name.strip().lower()
    if stripped.startswith("millisec") or stripped.startswith("timestamp"):
        return True
    return any(
        token in name
        for token in (
            "label",
            "gesture",
            "activity",
            "locomotion",
            "ll_left_arm",
            "ll_right_arm",
            "ml_both_arms",
        )
    )


def is_wearable_name(name: str) -> bool:
    exclude_tokens = (
        "object",
        "bottle",
        "salami",
        "bread",
        "sugar",
        "milk",
        "spoon",
        "knife",
        "plate",
        "cheese",
        "lazy",
        "door",
        "drawer",
        "lowerdrawer",
        "middledrawer",
        "topdrawer",
        "switch",
        "cup",
        "glass",
        "dishwasher",
        "fridge",
        "table",
        "chair",
    )
    if any(token in name for token in exclude_tokens):
        return False
    include_tokens = (
        "inertial",
        "back",
        "rua",
        "rla",
        "lua",
        "lla",
        "shoe",
        "ankle",
        "wrist",
        "arm",
        "leg",
        "hip",
        "body",
        "rkn",
        "lkn",
        "hip",
        "lh",
        "rh",
        "rwr",
        "lwr",
    )
    return any(token in name for token in include_tokens)


def label_aliases(label_column: str) -> List[str]:
    aliases = {
        "ml_both_arms": ["ml_both_arms", "both_arms"],
        "both_arms": ["ml_both_arms", "both_arms"],
        "hl_activity": ["hl_activity"],
        "activity": ["hl_activity", "activity"],
        "locomotion": ["locomotion"],
        "ll_left_arm": ["ll_left_arm"],
        "ll_right_arm": ["ll_right_arm"],
    }
    return aliases.get(label_column, [label_column])


def label_column_indices(column_names: List[str]) -> set[int]:
    track_names = {
        "locomotion",
        "hl_activity",
        "ll_left_arm",
        "ll_left_arm_object",
        "ll_right_arm",
        "ll_right_arm_object",
        "ml_both_arms",
    }
    normalized_names = [normalize_column_name(name) for name in column_names]
    return {
        index
        for index, name in enumerate(normalized_names)
        if name in track_names or any(track_name in name for track_name in track_names)
    }


def feature_names_from_indices(
    column_names: Optional[List[str]], feature_indices: Sequence[int]
) -> List[str]:
    if column_names and len(column_names) > max(feature_indices):
        return [column_names[index] for index in feature_indices]
    return [f"feature_{index}" for index in feature_indices]


def valid_label_mask(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.float64)
    finite = np.isfinite(labels)
    integer_labels = np.zeros_like(labels, dtype=np.int64)
    integer_labels[finite] = labels[finite].astype(np.int64)
    return finite & ~np.isin(integer_labels, list(NULL_LABELS))


def majority_label(labels: np.ndarray) -> int:
    values, counts = np.unique(labels.astype(np.int64), return_counts=True)
    return int(values[np.argmax(counts)])


def window_label_from_rule(labels: np.ndarray, rule: str) -> Optional[int]:
    labels = labels.astype(np.float64)
    if rule == "center":
        label = labels[len(labels) // 2]
        if not np.isfinite(label) or int(label) in NULL_LABELS:
            return None
        return int(label)
    if rule == "last":
        label = labels[-1]
        if not np.isfinite(label) or int(label) in NULL_LABELS:
            return None
        return int(label)
    valid = valid_label_mask(labels)
    if not valid.any():
        return None
    return majority_label(labels[valid])
