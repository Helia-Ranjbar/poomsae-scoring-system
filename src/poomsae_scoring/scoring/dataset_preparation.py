from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCORE_PATTERN = re.compile(
    r"Game Score\s*:\s*"
    r"Accuracy\s*=\s*(?P<accuracy>\d+(?:\.\d+)?)\s*,\s*"
    r"Presentation\s*=\s*(?P<presentation>\d+(?:\.\d+)?)\s*,\s*"
    r"Negativ Score\s*=\s*(?P<negative>\d+(?:\.\d+)?)\s*,\s*"
    r"Total Score\s*=\s*(?P<total>\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
PLAYER_ID_PATTERN = re.compile(r"^WP(?P<number>\d+)$", flags=re.IGNORECASE)
NON_FEATURE_COLUMNS = {
    "frame_index",
    "time_seconds",
    "status",
    "support_side",
    "mean_keypoint_confidence",
}
CIRCULAR_FEATURES = {"shoulder_tilt_degrees", "hip_tilt_degrees"}


@dataclass(frozen=True, slots=True)
class DatasetPreparationConfig:
    angles: tuple[str, ...] = ("0", "45", "135")
    minimum_detection_rate: float = 0.80
    minimum_mean_keypoint_confidence: float = 0.50
    minimum_usable_frame_rate: float = 0.50
    minimum_usable_frames: int = 30

    def __post_init__(self) -> None:
        if not self.angles:
            raise ValueError("At least one camera angle is required.")
        for name, value in (
            ("minimum_detection_rate", self.minimum_detection_rate),
            ("minimum_mean_keypoint_confidence", self.minimum_mean_keypoint_confidence),
            ("minimum_usable_frame_rate", self.minimum_usable_frame_rate),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.minimum_usable_frames < 1:
            raise ValueError("minimum_usable_frames must be at least 1.")


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    dataset: pd.DataFrame
    features: pd.DataFrame
    targets: pd.DataFrame
    quality_report: pd.DataFrame


def _player_number(player_id: str) -> int:
    match = PLAYER_ID_PATTERN.fullmatch(player_id)
    if match is None:
        raise ValueError(f"Player ID must use the WP<number> format: {player_id}")
    return int(match.group("number"))


def parse_score_file(path: Path) -> dict[str, float | str]:
    score_path = path.expanduser().resolve()
    player_id = score_path.stem.upper()
    _player_number(player_id)
    text = score_path.read_text(encoding="utf-8-sig", errors="replace")
    match = SCORE_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not find the Game Score row in {score_path}")

    accuracy = float(match.group("accuracy"))
    presentation = float(match.group("presentation"))
    negative_score = float(match.group("negative"))
    total = float(match.group("total"))
    expected_total = accuracy + presentation - negative_score
    if not math.isclose(total, expected_total, abs_tol=0.011):
        raise ValueError(
            f"Inconsistent total in {score_path}: got {total}, expected {expected_total:.2f}"
        )
    return {
        "player_id": player_id,
        "score_accuracy": accuracy,
        "score_presentation": presentation,
        "score_negative": negative_score,
        "score_total": total,
    }


def load_score_labels(labels_root: Path) -> pd.DataFrame:
    root = labels_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Score-label directory does not exist: {root}")

    records = [parse_score_file(path) for path in root.glob("WP*.txt")]
    if not records:
        raise FileNotFoundError(f"No WP*.txt score files found in {root}")
    player_ids = [str(record["player_id"]) for record in records]
    duplicates = sorted({item for item in player_ids if player_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate score labels: {', '.join(duplicates)}")

    records.sort(key=lambda record: _player_number(str(record["player_id"])))
    return pd.DataFrame.from_records(records)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _summary_statistics(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "median": math.nan,
            "std": math.nan,
            "p10": math.nan,
            "p90": math.nan,
        }
    return {
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


def _absolute_velocity(
    values: pd.Series,
    frame_indices: pd.Series,
    time_seconds: pd.Series,
    circular: bool,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_frames = pd.to_numeric(frame_indices, errors="coerce")
    numeric_times = pd.to_numeric(time_seconds, errors="coerce")
    delta = numeric_values.diff()
    if circular:
        delta = (delta + 180.0) % 360.0 - 180.0
    elapsed = numeric_times.diff()
    consecutive = numeric_frames.diff().eq(1) & elapsed.gt(0)
    return (delta.abs() / elapsed).where(consecutive)


def summarize_pose_view(
    player_id: str,
    angle: str,
    pose_directory: Path,
    config: DatasetPreparationConfig,
) -> tuple[dict[str, float | int | str], dict[str, float | int | str]]:
    prefix = f"view_{angle}"
    features_path = pose_directory / "features.csv"
    metadata_path = pose_directory / "metadata.json"
    if not features_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing pose outputs for {player_id}, angle {angle}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    frame_features = pd.read_csv(features_path, low_memory=False)
    required_columns = {
        "frame_index",
        "time_seconds",
        "status",
        "support_side",
        "mean_keypoint_confidence",
    }
    missing_columns = sorted(required_columns - set(frame_features.columns))
    if missing_columns:
        raise ValueError(
            f"{features_path} is missing columns: {', '.join(missing_columns)}"
        )

    total_frames = len(frame_features)
    detected_mask = frame_features["status"].eq("detected")
    confidence = pd.to_numeric(
        frame_features["mean_keypoint_confidence"],
        errors="coerce",
    )
    usable_mask = detected_mask & confidence.ge(
        config.minimum_mean_keypoint_confidence
    )
    detected_frames = int(detected_mask.sum())
    usable_frames = int(usable_mask.sum())
    detection_rate = detected_frames / total_frames if total_frames else 0.0
    usable_frame_rate = usable_frames / total_frames if total_frames else 0.0
    reliable = (
        detection_rate >= config.minimum_detection_rate
        and usable_frame_rate >= config.minimum_usable_frame_rate
        and usable_frames >= config.minimum_usable_frames
    )

    fps = _safe_float(metadata.get("fps"))
    duration_seconds = total_frames / fps if math.isfinite(fps) and fps > 0 else math.nan
    record: dict[str, float | int | str] = {
        "player_id": player_id,
        f"{prefix}_available": 1,
        f"{prefix}_reliable": int(reliable),
        f"{prefix}_detection_rate": detection_rate,
        f"{prefix}_usable_frame_rate": usable_frame_rate,
        f"{prefix}_mean_keypoint_confidence": (
            float(confidence[usable_mask].mean()) if usable_frames else math.nan
        ),
        f"{prefix}_duration_seconds": duration_seconds,
    }
    quality: dict[str, float | int | str] = {
        "player_id": player_id,
        "angle": angle,
        "frames": total_frames,
        "detected_frames": detected_frames,
        "usable_frames": usable_frames,
        "detection_rate": detection_rate,
        "usable_frame_rate": usable_frame_rate,
        "mean_keypoint_confidence": (
            float(confidence[usable_mask].mean()) if usable_frames else math.nan
        ),
        "duration_seconds": duration_seconds,
        "reliable": reliable,
        "quality_status": "usable" if reliable else "excluded_low_quality",
    }

    numeric_feature_names = [
        column
        for column in frame_features.columns
        if column not in NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(frame_features[column])
    ]
    if not reliable:
        for feature_name in numeric_feature_names:
            for statistic in ("median", "std", "p10", "p90"):
                record[f"{prefix}_{feature_name}_{statistic}"] = math.nan
            record[f"{prefix}_{feature_name}_velocity_abs_median"] = math.nan
            record[f"{prefix}_{feature_name}_velocity_abs_p90"] = math.nan
        for side in ("left", "right", "unknown"):
            record[f"{prefix}_support_{side}_ratio"] = math.nan
        record[f"{prefix}_support_switches_per_second"] = math.nan
        return record, quality

    usable = frame_features.loc[usable_mask].copy()
    for feature_name in numeric_feature_names:
        for statistic, value in _summary_statistics(usable[feature_name]).items():
            record[f"{prefix}_{feature_name}_{statistic}"] = value

        velocity = _absolute_velocity(
            usable[feature_name],
            usable["frame_index"],
            usable["time_seconds"],
            circular=feature_name in CIRCULAR_FEATURES,
        )
        velocity_summary = _summary_statistics(velocity)
        record[f"{prefix}_{feature_name}_velocity_abs_median"] = velocity_summary[
            "median"
        ]
        record[f"{prefix}_{feature_name}_velocity_abs_p90"] = velocity_summary["p90"]

    support_side = usable["support_side"].fillna("unknown").astype(str).str.lower()
    for side in ("left", "right", "unknown"):
        record[f"{prefix}_support_{side}_ratio"] = float(support_side.eq(side).mean())
    switches = support_side.ne(support_side.shift()) & support_side.isin({"left", "right"})
    record[f"{prefix}_support_switches_per_second"] = (
        float(switches.iloc[1:].sum() / duration_seconds)
        if duration_seconds > 0
        else math.nan
    )
    return record, quality


def prepare_multiview_dataset(
    labels_root: Path,
    players_root: Path,
    config: DatasetPreparationConfig | None = None,
) -> PreparedDataset:
    preparation_config = config or DatasetPreparationConfig()
    labels = load_score_labels(labels_root)
    root = players_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Processed-player directory does not exist: {root}")

    feature_records: list[dict[str, float | int | str]] = []
    quality_records: list[dict[str, float | int | str]] = []
    for player_id in labels["player_id"]:
        player_features: dict[str, float | int | str] = {"player_id": player_id}
        for angle in preparation_config.angles:
            pose_directory = root / player_id / "angles" / angle / "pose"
            view_features, view_quality = summarize_pose_view(
                player_id,
                angle,
                pose_directory,
                preparation_config,
            )
            player_features.update(
                {key: value for key, value in view_features.items() if key != "player_id"}
            )
            quality_records.append(view_quality)
        feature_records.append(player_features)

    features = pd.DataFrame.from_records(feature_records)
    features = features.reindex(
        columns=["player_id", *sorted(set(features.columns) - {"player_id"})]
    )
    targets = labels.copy()
    dataset = targets.merge(features, on="player_id", how="inner", validate="one_to_one")
    quality_report = pd.DataFrame.from_records(quality_records)
    quality_report["player_number"] = quality_report["player_id"].map(_player_number)
    quality_report["angle_number"] = pd.to_numeric(
        quality_report["angle"],
        errors="coerce",
    )
    quality_report = (
        quality_report.sort_values(["player_number", "angle_number"])
        .drop(columns=["player_number", "angle_number"])
        .reset_index(drop=True)
    )
    return PreparedDataset(
        dataset=dataset,
        features=features,
        targets=targets,
        quality_report=quality_report,
    )


def write_prepared_dataset(
    prepared: PreparedDataset,
    output_directory: Path,
    config: DatasetPreparationConfig | None = None,
) -> dict[str, Path]:
    preparation_config = config or DatasetPreparationConfig()
    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset": output / "prepared_dataset.csv",
        "features": output / "features.csv",
        "targets": output / "targets.csv",
        "quality_report": output / "quality_report.csv",
        "metadata": output / "metadata.json",
    }
    prepared.dataset.to_csv(paths["dataset"], index=False)
    prepared.features.to_csv(paths["features"], index=False)
    prepared.targets.to_csv(paths["targets"], index=False)
    prepared.quality_report.to_csv(paths["quality_report"], index=False)

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "players": len(prepared.dataset),
        "feature_columns": len(prepared.features.columns) - 1,
        "target_columns": [
            column for column in prepared.targets.columns if column != "player_id"
        ],
        "reliable_views": int(prepared.quality_report["reliable"].sum()),
        "total_views": len(prepared.quality_report),
        "configuration": asdict(preparation_config),
        "notes": [
            "One row represents one player performance.",
            "Camera views are summarized separately and are never averaged together.",
            "Low-quality views retain quality indicators but have missing pose summaries.",
            "Train/test splitting must be performed by player_id.",
        ],
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return paths
