from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import (
    FootPhaseData,
)


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    plot: bool = False
    show_timeline: bool = False
    show_comparison: bool = False
    show_events: bool = False
    show_video: bool = False
    show_velocity_viewer: bool = False
    show_skeleton_video: bool = False
    save_svg: bool = False


class PhaseDetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    fps: float = Field(..., gt=0)

    min_step_time: float | None = Field(None, gt=0)  # Fixed: added gt=0 constraint

    prominence_ratio_peaks_min: float = Field(..., gt=0)
    prominence_ratio_peaks_max: float | None = Field(None, gt=0)

    prominence_ratio_valleys_min: float = Field(..., gt=0)
    prominence_ratio_valleys_max: float | None = Field(None, gt=0)

    peak_height_min: float | None = None
    peak_height_max: float | None = None
    peak_width: float | None = None

    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @model_validator(mode="after")
    def _validate_prominence_ranges(self) -> PhaseDetectionConfig:
        """Ensure min <= max when both bounds are provided."""
        if (
            self.prominence_ratio_peaks_max is not None
            and self.prominence_ratio_peaks_min > self.prominence_ratio_peaks_max
        ):
            raise ValueError(
                "prominence_ratio_peaks_min must be <= prominence_ratio_peaks_max"
            )
        if (
            self.prominence_ratio_valleys_max is not None
            and self.prominence_ratio_valleys_min > self.prominence_ratio_valleys_max
        ):
            raise ValueError(
                "prominence_ratio_valleys_min must be <= prominence_ratio_valleys_max"
            )
        if (
            self.peak_height_min is not None
            and self.peak_height_max is not None
            and self.peak_height_min > self.peak_height_max
        ):
            raise ValueError("peak_height_min must be <= peak_height_max")
        return self

    # --- scipy.signal.find_peaks compatible properties ---
    # These return the value/tuple expected by the `prominence` and `height`
    # parameters of find_peaks. Call sites should pass them directly without
    # isinstance checks.

    @property
    def prominence_ratio_peaks(self) -> float | tuple[float, float]:
        if self.prominence_ratio_peaks_max is not None:
            return (self.prominence_ratio_peaks_min, self.prominence_ratio_peaks_max)
        return self.prominence_ratio_peaks_min

    @property
    def prominence_ratio_valleys(self) -> float | tuple[float, float]:
        if self.prominence_ratio_valleys_max is not None:
            return (
                self.prominence_ratio_valleys_min,
                self.prominence_ratio_valleys_max,
            )
        return self.prominence_ratio_valleys_min

    @property
    def peak_height(self) -> float | tuple[float | None, float | None] | None:
        """
        Returns a 2-tuple when either bound is set, so find_peaks always
        receives a consistent type. None entries in the tuple mean 'unbounded'.
        """
        if self.peak_height_min is not None or self.peak_height_max is not None:
            return (self.peak_height_min, self.peak_height_max)
        return None


class VelocityPhaseResult(BaseModel):
    """
    Output of velocity-based gait phase detection.
    Contains full foot keypoints, masks, and derived phase points.
    """

    model_config = ConfigDict(frozen=True)

    fps: float = Field(..., gt=0, description="Frames per second of the source video")
    left: FootPhaseData
    right: FootPhaseData

    @property
    def time(self) -> list[float]:
        """
        Derives per-frame timestamps in seconds from fps and the left foot
        foot keypoints length, preventing time/foot keypoints length drift.
        """
        n_frames = len(self.left.foot_keypoints.heel)
        return [i / self.fps for i in range(n_frames)]

    @model_validator(mode="after")
    def _validate_frame_counts(self) -> VelocityPhaseResult:
        """Left and right foot keypoints must have the same number of frames."""
        n_left = len(self.left.foot_keypoints.heel)
        n_right = len(self.right.foot_keypoints.heel)
        if n_left != n_right:
            raise ValueError(
                f"Left and right foot keypoints must have the same frame count, "
                f"got left={n_left}, right={n_right}"
            )
        return self
