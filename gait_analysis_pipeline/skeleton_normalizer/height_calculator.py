from __future__ import annotations

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from pose_estimation.pose_estimation_dataclasses.processing_steps_dataclass import (
    ProcessingStep,
)
from skeleton_normalizer.dataclasses.height_estimator_base_dataclass import (
    HeightEstimator,
)
from skeleton_normalizer.dataclasses.height_signal_smoother_base import (
    HeightSignal,
    HeightSignalSmoother,
)
import skeleton_normalizer.height_estimator_methods  # noqa: F401 — registers estimators


class HeightPipeline:
    """
    Responsible solely for producing a smoothed HeightSignal from a list of poses.

    Composes a HeightEstimator (raw per-frame height values) with a
    HeightSignalSmoother (temporal smoothing), returning the resulting
    HeightSignal for use downstream.

    Deliberately has no knowledge of normalization or visualization — those
    are the caller's concern.
    """

    def __init__(
        self,
        estimator: HeightEstimator,
        smoother: HeightSignalSmoother,
    ) -> None:
        self._estimator = estimator
        self._smoother = smoother

    # =====================================================
    # PUBLIC
    # =====================================================

    def run(self, poses: list[FramePose]) -> tuple[HeightSignal, HeightSignal]:
        """
        Estimate and smooth heights for a list of poses.

        Returns
        -------
        raw : HeightSignal
            Per-frame height estimates before smoothing.
        smoothed : HeightSignal
            Temporally smoothed heights, one value per frame.
        """
        raw = self._estimator.estimate(poses)
        smoothed = self._smoother.smooth(raw)
        return raw, smoothed

    def to_processing_step(self) -> ProcessingStep:
        """
        Build a ProcessingStep describing this pipeline's configuration,
        including the estimator name, joint schema, and smoother parameters.
        """
        return self._estimator.to_processing_step(self._smoother.get_params())
