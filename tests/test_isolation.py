from pathlib import Path

import numpy as np
import pytest

from poomsae_scoring.preprocessing.isolation import (
    IsolationConfig,
    discover_player_videos,
    group_player_videos,
    mask_ground_contact,
    parse_player_video,
    plausible_same_person,
    player_angle_output_directory,
    select_mat_athlete,
)


def test_parse_player_video_extracts_player_and_angle() -> None:
    video = parse_player_video(Path("data/raw/cut-women-45/WP12-45.MP4"))

    assert video.player_id == "WP12"
    assert video.angle == "45"
    assert video.path.name == "WP12-45.MP4"


def test_discovery_groups_player_views_in_numeric_order(tmp_path: Path) -> None:
    for relative_path in ("cut-135/WP2-135.mp4", "cut-0/WP2-0.mp4", "cut-45/WP2-45.mp4"):
        path = tmp_path / relative_path
        path.parent.mkdir(exist_ok=True)
        path.touch()

    grouped = group_player_videos(discover_player_videos(tmp_path))

    assert list(grouped) == ["WP2"]
    assert [video.angle for video in grouped["WP2"]] == ["0", "45", "135"]


def test_discovery_rejects_duplicate_player_angle(tmp_path: Path) -> None:
    first = tmp_path / "first" / "WP1-0.mp4"
    second = tmp_path / "second" / "WP1-0.mov"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="Duplicate player-angle"):
        discover_player_videos(tmp_path)


def test_output_directory_is_nested_by_player_and_angle(tmp_path: Path) -> None:
    output = player_angle_output_directory(tmp_path, "WP7", "135")

    assert output == tmp_path.resolve() / "WP7" / "angles" / "135"


def test_mask_ground_contact_uses_bottom_mask_band() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:9, 4:7] = 1

    contact = mask_ground_contact(mask, np.array([4, 2, 7, 9]), 10, 10)

    np.testing.assert_allclose(contact, [5, 6.5])


def test_select_mat_athlete_prefers_person_nearest_mat_center() -> None:
    polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
    boxes = np.array([[5, 5, 25, 90], [40, 5, 60, 90]], dtype=float)
    masks = np.zeros((2, 100, 100), dtype=np.uint8)
    masks[0, 5:91, 5:26] = 1
    masks[1, 5:91, 40:61] = 1

    selected = select_mat_athlete(boxes, masks, polygon, 100, 100)

    assert selected == 1


def test_plausible_same_person_rejects_large_jump() -> None:
    assert not plausible_same_person(
        np.array([500, 50, 600, 400]),
        np.array([100, 50, 200, 400]),
        frame_width=640,
        frame_height=480,
    )


def test_isolation_config_rejects_invalid_reference_position() -> None:
    with pytest.raises(ValueError, match="Reference position"):
        IsolationConfig(reference_position=1.1)
