from __future__ import annotations
import csv
import re
import zipfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from src.data.splits import (
    chronological_window_split,
    interpolate_missing,
    slide_windows,
)
import numpy as np
import torch
from src.data.splits import normalize_column_name

REALWORLD_ACTIVITIES = [
    "stairs_down",
    "stairs_up",
    "jumping",
    "lying",
    "standing",
    "sitting",
    "running",
    "walking",
]
REALWORLD_ACTIVITY_TO_INDEX = {
    name: index for index, name in enumerate(REALWORLD_ACTIVITIES)
}
REALWORLD_ACTIVITY_ALIASES = {
    "climbingdown": "stairs_down",
    "climbingup": "stairs_up",
    "jumping": "jumping",
    "lying": "lying",
    "standing": "standing",
    "sitting": "sitting",
    "running": "running",
    "walking": "walking",
}
REALWORLD_LOCATIONS = ["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"]
LOCATION_ALIASES = {
    "upperarm": "upperarm",
    "upper_arm": "upperarm",
    "upper-arm": "upperarm",
    "forearm": "forearm",
    "chest": "chest",
    "head": "head",
    "shin": "shin",
    "thigh": "thigh",
    "waist": "waist",
}
FEATURE_SET_SENSORS = {
    "acc3": ("acc",),
    "accgyro6": ("acc", "gyr"),
    "imu9": ("acc", "gyr", "mag"),
}


@dataclass
class SensorSequence:
    location: str
    values: np.ndarray


@dataclass
class AlignedSequence:
    subject: int
    activity: str
    location: str
    values: np.ndarray
    sequence_id: int


@dataclass
class RealWorldArrays:
    windows: torch.Tensor
    labels: torch.Tensor
    subjects: torch.Tensor
    location_indices: torch.Tensor
    sequence_ids: torch.Tensor
    starts: torch.Tensor
    window_size: int
    window_stride: int


def resolve_realworld_root(root: Union[str, Path]) -> Path:
    root = Path(root)
    candidates = [root / "realworld2016_dataset", root]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("proband*/data")):
            return candidate
    raise FileNotFoundError(
        f"No RealWorld proband directories found under {root}. "
        "Expected data/realworld/realworld2016_dataset/probandX/data/*.zip "
        "or a root that directly contains probandX directories."
    )


def build_realworld_windows(
    root: Union[str, Path],
    window_size: int = 500,
    window_stride: int = 250,
    feature_set: str = "imu9",
) -> RealWorldArrays:
    sequences = load_aligned_realworld_sequences(root=root, feature_set=feature_set)
    windows, columns, starts = slide_windows(
        (
            (
                sequence.values,
                {
                    "labels": REALWORLD_ACTIVITY_TO_INDEX[sequence.activity],
                    "subjects": sequence.subject,
                    "location_indices": REALWORLD_LOCATIONS.index(sequence.location),
                    "sequence_ids": sequence.sequence_id,
                },
            )
            for sequence in sequences
        ),
        window_size,
        window_stride,
        "RealWorld",
    )
    return RealWorldArrays(
        windows=windows,
        starts=starts,
        window_size=window_size,
        window_stride=window_stride,
        **columns,
    )


def read_all_sensor_sequences(
    root: Path, required_sensors: Sequence[str]
) -> Dict[Tuple[int, str, str, str], SensorSequence]:
    """Every usable sensor sequence on disk, keyed by (subject, activity, location, sensor)."""
    sequences: Dict[Tuple[int, str, str, str], SensorSequence] = {}
    for subject_dir in proband_dirs(root):
        subject = subject_from_proband_dir(subject_dir)
        for zip_path in sorted((subject_dir / "data").glob("*_csv.zip")):
            parsed = parse_realworld_zip_name(zip_path.name)
            if parsed is None:
                continue
            sensor, activity = parsed
            if sensor not in required_sensors:
                continue
            found = read_sensor_zip(zip_path, sensor=sensor)
            for sequence in found:
                sequences[(subject, activity, sequence.location, sensor)] = sequence
    return sequences


def align_sensor_sequences(
    sensor_sequences: Dict[Tuple[int, str, str, str], SensorSequence],
    required_sensors: Sequence[str],
) -> List[AlignedSequence]:
    """Concatenate each location's sensors channel-wise, truncated to their shortest run."""
    aligned: List[AlignedSequence] = []
    for subject, activity, location in sorted({key[:3] for key in sensor_sequences}):
        per_sensor = {
            s: sensor_sequences.get((subject, activity, location, s))
            for s in required_sensors
        }
        missing = [
            sensor for sensor, sequence in per_sensor.items() if sequence is None
        ]
        if missing:
            raise ValueError(
                f"Missing RealWorld modalities for subject={subject} activity={activity} "
                f"location={location}: {missing}"
            )
        n_aligned = min(sequence.values.shape[0] for sequence in per_sensor.values())
        if n_aligned <= 0:
            continue
        values = np.concatenate(
            [per_sensor[sensor].values[:n_aligned] for sensor in required_sensors],
            axis=1,
        ).astype(np.float32)
        aligned.append(
            AlignedSequence(
                subject=subject,
                activity=activity,
                location=location,
                values=values,
                sequence_id=len(aligned),
            )
        )
    return aligned


def load_aligned_realworld_sequences(
    root: Union[str, Path], feature_set: str = "imu9"
) -> List[AlignedSequence]:
    required_sensors = FEATURE_SET_SENSORS[feature_set]
    sensor_sequences = read_all_sensor_sequences(
        resolve_realworld_root(root), required_sensors
    )
    return align_sensor_sequences(sensor_sequences, required_sensors)


def proband_dirs(root: Path) -> List[Path]:
    dirs = sorted(path for path in root.glob("proband*") if path.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No proband directories found under {root}")
    return dirs


def subject_from_proband_dir(path: Path) -> int:
    match = re.search(r"proband(\d+)", path.name.lower())
    if match is None:
        raise ValueError(f"Could not infer RealWorld subject from {path}")
    return int(match.group(1))


def parse_realworld_zip_name(name: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^(acc|gyr|mag|gps|lig|mic)_([a-z]+)_csv\.zip$", name.lower())
    if match is None:
        return None
    sensor = match.group(1)
    raw_activity = match.group(2)
    if raw_activity not in REALWORLD_ACTIVITY_ALIASES:
        return None
    return sensor, REALWORLD_ACTIVITY_ALIASES[raw_activity]


def read_sensor_zip(zip_path: Path, sensor: str) -> List[SensorSequence]:
    sequences: List[SensorSequence] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in sorted(
            name for name in archive.namelist() if not name.endswith("/")
        ):
            if not member.lower().endswith(".csv"):
                continue
            location = infer_location_from_member(member)
            with archive.open(member) as handle:
                raw = handle.read()
            if location is None:
                location = infer_location_from_csv_bytes(raw)
            if location is None:
                continue
            try:
                matrix, column_names = read_csv_bytes(raw)
                indices = select_axis_columns(matrix, column_names, sensor=sensor)
                values = interpolate_missing(matrix[:, indices].astype(np.float32))
                sequences.append(SensorSequence(location=location, values=values))
            except Exception:
                continue  # unreadable sensor file: skip it
    return sequences


def read_csv_bytes(raw: bytes) -> Tuple[np.ndarray, Optional[List[str]]]:
    text = decode_text(raw)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("CSV member is empty")
    delimiter = detect_delimiter(lines[0])
    first_fields = split_line(lines[0], delimiter)
    has_header = any(re.search(r"[A-Za-z]", field) for field in first_fields)
    data_text = "\n".join(lines[1:] if has_header else lines)
    names = [clean_header_name(field) for field in first_fields] if has_header else None
    data = np.genfromtxt(
        StringIO(data_text), delimiter=delimiter, dtype=np.float32, invalid_raise=False
    )
    if data.size == 0:
        raise ValueError("CSV member has no numeric rows")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if names is not None and len(names) != data.shape[1]:
        names = None
    return data.astype(np.float32), names


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def detect_delimiter(line: str) -> Optional[str]:
    counts = {delimiter: line.count(delimiter) for delimiter in (",", ";", "\t")}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    return delimiter if count > 0 else None


def split_line(line: str, delimiter: Optional[str]) -> List[str]:
    if delimiter is None:
        return line.split()
    return next(csv.reader([line], delimiter=delimiter))


def clean_header_name(name: str) -> str:
    return name.strip().strip('"').strip("'")


def axis_columns_by_name(normalized: List[str], sensor: str) -> Optional[List[int]]:
    """Exact x/y/z column names for this sensor, if the header carries them."""
    for triple in axis_name_triples(sensor):
        if all(name in normalized for name in triple):
            indices = [normalized.index(name) for name in triple]
            return indices
    suffix_indices = suffix_axis_indices(normalized, sensor)
    if suffix_indices is not None:
        return suffix_indices
    return None


def fallback_axis_columns(
    matrix: np.ndarray, column_names: Optional[List[str]]
) -> List[int]:
    """Last three non-time columns, used when no named axes match."""
    numeric = [
        index
        for index in range(matrix.shape[1])
        if not column_names or not is_time_column(column_names[index])
    ]
    if len(numeric) < 3:
        numeric = list(range(matrix.shape[1]))
    return numeric[-3:]


def select_axis_columns(
    matrix: np.ndarray, column_names: Optional[List[str]], sensor: str
) -> List[int]:
    """The three axis columns for `sensor`: by name where possible, else by position."""
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError(
            f"Need at least three columns for sensor {sensor}, got shape {matrix.shape}"
        )
    named = bool(column_names) and len(column_names) == matrix.shape[1]
    if named:
        matched = axis_columns_by_name(
            [normalize_column_name(n) for n in column_names], sensor
        )
        if matched is not None:
            return matched
    return fallback_axis_columns(matrix, column_names)


def axis_name_triples(sensor: str) -> List[Tuple[str, str, str]]:
    sensor_aliases = {
        "acc": ("acc", "accelerometer"),
        "gyr": ("gyr", "gyro", "gyroscope"),
        "mag": ("mag", "magnetic", "magnetometer"),
    }.get(sensor, (sensor,))
    triples = [("x", "y", "z"), ("attr_x", "attr_y", "attr_z")]
    for alias in sensor_aliases:
        triples.append((f"{alias}_x", f"{alias}_y", f"{alias}_z"))
    return triples


def suffix_axis_indices(
    normalized_names: Sequence[str], sensor: str
) -> Optional[List[int]]:
    aliases = {
        "acc": ("acc", "accelerometer"),
        "gyr": ("gyr", "gyro", "gyroscope"),
        "mag": ("mag", "magnetic", "magnetometer"),
    }.get(sensor, (sensor,))
    indices = []
    for axis in ("x", "y", "z"):
        matches = [
            index
            for index, name in enumerate(normalized_names)
            if name.endswith(f"_{axis}") and any(alias in name for alias in aliases)
        ]
        if not matches:
            return None
        indices.append(matches[-1])
    return indices


def chronological_target_split(
    arrays: "RealWorldArrays", target_indices: torch.Tensor, target_adapt_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    return chronological_window_split(
        arrays.sequence_ids,
        target_indices,
        target_adapt_ratio,
        sort_key=lambda index: int(arrays.starts[index].item()),
    )


def infer_location_from_member(member: str) -> Optional[str]:
    normalized = normalize_column_name(member)
    for alias, canonical in LOCATION_ALIASES.items():
        alias_norm = normalize_column_name(alias)
        if re.search(rf"(^|_){re.escape(alias_norm)}($|_)", normalized):
            return canonical
    return None


def infer_location_from_csv_bytes(raw: bytes) -> Optional[str]:
    text = decode_text(raw)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    header = normalize_column_name(lines[0])
    for alias, canonical in LOCATION_ALIASES.items():
        alias_norm = normalize_column_name(alias)
        if re.search(rf"(^|_){re.escape(alias_norm)}($|_)", header):
            return canonical
    return None


def canonical_location(value: str) -> str:
    normalized = normalize_column_name(value)
    if normalized in LOCATION_ALIASES:
        return LOCATION_ALIASES[normalized]
    raise ValueError(
        f"Unknown RealWorld location {value!r}; expected one of {REALWORLD_LOCATIONS}"
    )


def is_time_column(name: str) -> bool:
    normalized = normalize_column_name(name)
    return any(
        token in normalized for token in ("time", "timestamp", "millis", "index", "id")
    )


def feature_names_for_feature_set(feature_set: str) -> List[str]:
    names = []
    for sensor in FEATURE_SET_SENSORS[feature_set]:
        names.extend([f"{sensor}_x", f"{sensor}_y", f"{sensor}_z"])
    return names
