import json
from pathlib import Path

import pandas as pd
import pytest

from poomsae_scoring.scoring.dataset_preparation import (
    DatasetPreparationConfig,
    parse_score_file,
    prepare_multiview_dataset,
    write_prepared_dataset,
)


def write_score(path: Path, accuracy: float = 3.1, presentation: float = 4.5) -> None:
    path.write_text(
        f'"Game Score : Accuracy = {accuracy} , Presentation = {presentation} , '
        f'Negativ Score = 0 , Total Score = {accuracy + presentation}"',
        encoding="utf-8",
    )


def write_pose_view(path: Path, detected_rows: int = 4, missing_rows: int = 0) -> None:
    path.mkdir(parents=True)
    rows = []
    total = detected_rows + missing_rows
    for frame_index in range(total):
        detected = frame_index < detected_rows
        rows.append(
            {
                "frame_index": frame_index,
                "time_seconds": frame_index / 10,
                "status": "detected" if detected else "missing",
                "motion_energy": frame_index * 0.1 if detected else None,
                "left_knee_angle": 160 + frame_index if detected else None,
                "shoulder_tilt_degrees": 179 if frame_index % 2 == 0 else -179,
                "support_side": "left" if frame_index < 2 else "right",
                "mean_keypoint_confidence": 0.9 if detected else None,
            }
        )
    pd.DataFrame(rows).to_csv(path / "features.csv", index=False)
    (path / "metadata.json").write_text(
        json.dumps({"fps": 10, "frames_processed": total}),
        encoding="utf-8",
    )


def test_parse_score_file_extracts_performance_targets(tmp_path: Path) -> None:
    score_path = tmp_path / "WP12.txt"
    write_score(score_path)

    label = parse_score_file(score_path)

    assert label == {
        "player_id": "WP12",
        "score_accuracy": 3.1,
        "score_presentation": 4.5,
        "score_negative": 0.0,
        "score_total": 7.6,
    }


def test_parse_score_file_rejects_inconsistent_total(tmp_path: Path) -> None:
    score_path = tmp_path / "WP1.txt"
    score_path.write_text(
        '"Game Score : Accuracy = 3.1 , Presentation = 4.5 , '
        'Negativ Score = 0 , Total Score = 9.9"',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Inconsistent total"):
        parse_score_file(score_path)


def test_prepare_multiview_dataset_keeps_views_separate(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    players_root = tmp_path / "players"
    labels_root.mkdir()
    write_score(labels_root / "WP1.txt")
    for angle in ("0", "45", "135"):
        write_pose_view(players_root / "WP1" / "angles" / angle / "pose")
    config = DatasetPreparationConfig(
        minimum_detection_rate=0.8,
        minimum_usable_frame_rate=0.5,
        minimum_usable_frames=2,
    )

    prepared = prepare_multiview_dataset(labels_root, players_root, config)

    assert len(prepared.dataset) == 1
    assert prepared.dataset.loc[0, "player_id"] == "WP1"
    assert "view_0_left_knee_angle_median" in prepared.features
    assert "view_45_left_knee_angle_median" in prepared.features
    assert "view_135_left_knee_angle_median" in prepared.features
    assert not any(column.startswith("score_") for column in prepared.features)
    assert prepared.quality_report["reliable"].all()


def test_low_quality_view_has_indicators_without_pose_summary(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    players_root = tmp_path / "players"
    labels_root.mkdir()
    write_score(labels_root / "WP1.txt")
    write_pose_view(players_root / "WP1" / "angles" / "0" / "pose", 1, 3)
    for angle in ("45", "135"):
        write_pose_view(players_root / "WP1" / "angles" / angle / "pose")
    config = DatasetPreparationConfig(
        minimum_detection_rate=0.8,
        minimum_usable_frame_rate=0.5,
        minimum_usable_frames=2,
    )

    prepared = prepare_multiview_dataset(labels_root, players_root, config)

    assert prepared.features.loc[0, "view_0_reliable"] == 0
    assert pd.isna(prepared.features.loc[0, "view_0_left_knee_angle_median"])
    assert prepared.features.loc[0, "view_45_reliable"] == 1


def test_write_prepared_dataset_exports_all_tables(tmp_path: Path) -> None:
    labels_root = tmp_path / "labels"
    players_root = tmp_path / "players"
    labels_root.mkdir()
    write_score(labels_root / "WP1.txt")
    for angle in ("0", "45", "135"):
        write_pose_view(players_root / "WP1" / "angles" / angle / "pose")
    config = DatasetPreparationConfig(minimum_usable_frames=2)
    prepared = prepare_multiview_dataset(labels_root, players_root, config)

    paths = write_prepared_dataset(prepared, tmp_path / "model_ready", config)

    assert all(path.is_file() for path in paths.values())
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert metadata["players"] == 1
    assert metadata["total_views"] == 3
