from __future__ import annotations

import numpy as np

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
)
from pose_estimation.pose_estimation_dataclasses.processing_steps_dataclass import (
    ProcessingStep,
)

from filtering_scripts.filter_dataclass import (
    FilterConfig,
    SkelPointsFilter,
)


class FramePoseFilterProcessor:
    """
    Pure transformation operator:
    - input: list[FramePose]
    - output: list[FramePose]
    - NO metadata mutation
    """

    def __init__(
        self,
        filter_impl: SkelPointsFilter,
        filter_config: FilterConfig,
    ):
        self._filter = filter_impl
        self._filter_config = filter_config

    # =====================================================
    # PROCESSING STEP (ONLY OUTPUT METADATA)
    # =====================================================

    @property
    def processing_step(self) -> ProcessingStep:
        return self._filter_config.to_processing_step()

    def _extract_joint_trajectory(
        self,
        poses: list[FramePose],
        joint_idx: int,
    ) -> np.ndarray:
        """
        Extracts (x, y) trajectory for a single joint across all frames.
        Returns shape (num_frames, 2)
        """
        return np.array(
            [
                [
                    pose.joints[joint_idx].keypoint.x,
                    pose.joints[joint_idx].keypoint.y,
                ]
                for pose in poses
            ],
            dtype=float,
        )

    # =====================================================
    # MAIN FILTER
    # =====================================================

    def filter(self, poses: list[FramePose]) -> tuple[list[FramePose], ProcessingStep]:

        if not poses:
            return poses, self.processing_step

        # extract full array (num_frames, num_joints, 2)
        sequence_array = np.array(
            [[[j.keypoint.x, j.keypoint.y] for j in pose.joints] for pose in poses],
            dtype=float,
        )

        joint_count = sequence_array.shape[1]

        for joint_idx in range(joint_count):
            trajectory = sequence_array[:, joint_idx, :]  # (num_frames, 2)

            if np.isnan(trajectory).any():
                mean = np.nanmean(trajectory, axis=0)
                trajectory = np.where(np.isnan(trajectory), mean, trajectory)

            sequence_array[:, joint_idx, :] = self._filter.filter_data(trajectory)

        updated_poses = [
            pose.with_updated_keypoints(sequence_array[i])
            for i, pose in enumerate(poses)
        ]

        return updated_poses, self.processing_step
