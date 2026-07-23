from __future__ import annotations

import numpy as np

from gait_measurement_pipeline.gait_dataclasses.contact_detection_dataclass import (
    GaitEventsResult,
)
from gait_measurement_pipeline.gait_dataclasses.derived_metric_dataclass import (
    DerivedMetrics,
)
from pose_estimation.enums.joint_enum import JointEnum
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose


class GaitDerivedMetricsComputer:
    """Computes average cadence and velocity from gait events and pose data."""

    def __init__(
        self,
        results: GaitEventsResult,
        time: list[float],
        fps: float,
        poses: list[FramePose] | None = None,
        joint_enum: type[JointEnum] | None = None,
    ) -> None:
        self.results = results
        self.time = time
        self.fps = fps
        self.poses = poses
        self.joints = joint_enum

    def compute(self) -> DerivedMetrics:
        return DerivedMetrics(
            avg_cadence=self._compute_avg_cadence(),
            avg_velocity=self._compute_avg_velocity(),
        )

    # ------------------------------------------------------------------

    def _compute_avg_cadence(self) -> float | None:
        """Average cadence in steps/min across all foot contacts."""
        all_fc = sorted(
            self.results.left.First_contact_times(self.fps)
            + self.results.right.First_contact_times(self.fps)
        )
        if len(all_fc) < 2:
            return None

        step_intervals = np.diff(all_fc)
        return float(np.mean(60.0 / step_intervals))

    def _compute_avg_velocity(self) -> float | None:
        """Mean mid-hip velocity during the active walking window."""
        if self.poses is None or self.joints is None:
            return None

        left_fc = self.results.left.First_contact_times(self.fps)
        left_lc = self.results.left.Last_contact_times(self.fps)
        right_fc = self.results.right.First_contact_times(self.fps)
        right_lc = self.results.right.Last_contact_times(self.fps)

        if not (left_lc or left_fc or right_lc or right_fc):
            return None

        mid_hips = np.array([self._mid_hip(p) for p in self.poses])
        time = np.array(self.time)

        dist = np.linalg.norm(np.diff(mid_hips, axis=0), axis=1)
        frame_velocities = dist / (np.diff(time) + 1e-6)

        walking_start = min(left_lc + right_lc, default=time[0])
        walking_end = max(left_fc + right_fc, default=time[-1])

        mask = (time[:-1] >= walking_start) & (time[:-1] <= walking_end)
        return float(np.mean(frame_velocities[mask])) if np.any(mask) else None

    def _mid_hip(self, pose: FramePose) -> tuple[float, float]:
        left = pose.get_keypoint(self.joints.LEFT_HIP.name)
        right = pose.get_keypoint(self.joints.RIGHT_HIP.name)
        return ((left.x + right.x) / 2, (left.y + right.y) / 2)
