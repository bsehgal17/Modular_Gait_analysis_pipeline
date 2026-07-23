from __future__ import annotations

import numpy as np

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from pose_estimation.enums.joint_enum import JointEnum

from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import (
    FootKeypoints,
    FootVelocity,
)
from gait_measurement_pipeline.phase_detection.foot_extractor import FootExtractor
from gait_measurement_pipeline.utils.compute_joint_speed import compute_joint_speed


class FootVelocityExtractor:
    """
    Extracts foot keypoints from poses and computes per-component velocities.

    Output FootVelocity is the natural place to apply smoothing before
    passing to FootPhaseDetector.
    """

    def __init__(self, joint_enum: type[JointEnum]) -> None:
        self._extractor = FootExtractor(joint_enum)

    def extract(
        self, poses: list[FramePose]
    ) -> tuple[FootVelocity, FootVelocity, FootKeypoints, FootKeypoints]:
        """
        Extract keypoints and compute velocities for both feet.

        Returns
        -------
        left_vel, right_vel : FootVelocity
            Raw (unsmoothed) velocities — smooth these before phase detection.
        left_kp, right_kp : FootKeypoints
            Retained so the caller can pass them into VelocityPhaseResult later.
        """
        left_kp, right_kp = self._extractor.extract(poses)

        left_vel = self._compute_velocities(left_kp)
        right_vel = self._compute_velocities(right_kp)

        return left_vel, right_vel, left_kp, right_kp

    def _compute_velocities(self, foot_keypoints: FootKeypoints) -> FootVelocity:
        return FootVelocity(
            full_avg=compute_joint_speed(
                FootExtractor.to_numpy(foot_keypoints.full_avg)
            ).tolist(),
            proximal_avg=compute_joint_speed(
                FootExtractor.to_numpy(foot_keypoints.proximal_avg)
            ).tolist(),
            toe=compute_joint_speed(
                FootExtractor.to_numpy(foot_keypoints.toe)
            ).tolist(),
        )
