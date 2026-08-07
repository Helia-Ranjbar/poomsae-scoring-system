from pathlib import Path

import numpy as np
import pytest

from poomsae_scoring.preprocessing.video import discover_videos, letterbox_frame


def test_letterbox_frame_preserves_target_shape_and_centers_content() -> None:
    source = np.full((100, 200, 3), 255, dtype=np.uint8)

    result = letterbox_frame(source, target_width=300, target_height=300)

    assert result.shape == (300, 300, 3)
    assert np.all(result[75:225] == 255)
    assert np.all(result[:75] == 0)
    assert np.all(result[225:] == 0)


def test_discover_videos_recursively_filters_extensions(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first_video = tmp_path / "first.mp4"
    second_video = nested / "second.MOV"
    ignored_file = nested / "notes.txt"
    for path in (first_video, second_video, ignored_file):
        path.touch()

    videos = discover_videos(tmp_path)

    assert videos == [first_video.resolve(), second_video.resolve()]


def test_discover_videos_rejects_unsupported_file(tmp_path: Path) -> None:
    input_path = tmp_path / "image.jpg"
    input_path.touch()

    with pytest.raises(ValueError, match="Unsupported video extension"):
        discover_videos(input_path)

