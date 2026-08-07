import pytest

from poomsae_scoring.config import PreprocessingConfig


def test_default_config_is_valid() -> None:
    config = PreprocessingConfig()

    assert config.width == 640
    assert config.height == 640
    assert config.target_fps == 15.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("height", -1),
        ("target_fps", 0),
        ("jpeg_quality", 101),
        ("max_duration_seconds", -1),
    ],
)
def test_invalid_config_values_raise(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(**{field: value})

