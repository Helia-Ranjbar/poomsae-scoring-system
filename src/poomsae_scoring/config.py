from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    width: int = 640
    height: int = 640
    target_fps: float = 15.0
    jpeg_quality: int = 95
    max_duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame dimensions must be positive.")
        if self.target_fps <= 0:
            raise ValueError("Target FPS must be positive.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 1 and 100.")
        if self.max_duration_seconds is not None and self.max_duration_seconds <= 0:
            raise ValueError("Maximum duration must be positive.")

