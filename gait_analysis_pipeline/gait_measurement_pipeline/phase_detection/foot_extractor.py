from __future__ import annotations

import numpy as np

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from pose_estimation.enums.joint_enum import JointEnum
from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import (
    Coordinate,
    FootKeypoints,
)


class FootExtractor:
    """
    Extracts foot keypoints from list[FramePose].
    Shared by VelocityPhaseDetector and PhaseVisualizer.
    """

    def __init__(self, joint_enum: type[JointEnum]) -> None:
        self._joints = joint_enum

    def extract(
        self,
        poses: list[FramePose],
    ) -> tuple[FootKeypoints, FootKeypoints]:
        """
        Returns (left_keypoints, right_keypoints).
        """
        left_heel, left_ankle, left_toe = [], [], []
        right_heel, right_ankle, right_toe = [], [], []

        for pose in poses:
            frame_idx = pose.frame_idx

            left_heel.append(self._to_coord(pose, self._joints.LEFT_HEEL, frame_idx))
            left_ankle.append(self._to_coord(pose, self._joints.LEFT_ANKLE, frame_idx))
            left_toe.append(self._to_coord(pose, self._joints.LEFT_BIG_TOE, frame_idx))

            right_heel.append(self._to_coord(pose, self._joints.RIGHT_HEEL, frame_idx))
            right_ankle.append(
                self._to_coord(pose, self._joints.RIGHT_ANKLE, frame_idx)
            )
            right_toe.append(
                self._to_coord(pose, self._joints.RIGHT_BIG_TOE, frame_idx)
            )

        left = FootKeypoints(heel=left_heel, ankle=left_ankle, toe=left_toe)
        right = FootKeypoints(heel=right_heel, ankle=right_ankle, toe=right_toe)

        return left, right

    @staticmethod
    def _to_coord(pose: FramePose, joint_member, frame_idx: int) -> Coordinate:
        kp = pose.get_keypoint(joint_member.name)
        return Coordinate(x=kp.x, y=kp.y, frame_idx=frame_idx)

    @staticmethod
    def to_numpy(coords: list[Coordinate]) -> np.ndarray:
        """
        Convert list[Coordinate] → np.ndarray shape (N, 2).
        Used when numpy operations are needed.
        """
        return np.array([[c.x, c.y] for c in coords], dtype=np.float64)
