from __future__ import annotations

from pose_estimation.pose_estimation_dataclasses.pose_estimation_dataclass import (
    PoseEstimationResult,
)
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose


class TrackedPoseResolver:
    """
    Resolves a tracked person into a semantic pose sequence.
    Designed for single-primary-person selection (not full re-ID system).
    """

    def resolve(self, pose_result: PoseEstimationResult) -> tuple[list[FramePose], int]:
        person = self._select_best_person(pose_result.persons)
        sorted_poses = sorted(person.poses, key=lambda p: p.frame_idx)
        return sorted_poses, person.person_id

    @staticmethod
    def _select_best_person(persons):
        """
        Selects the most reliable tracked person using:
        - detection confidence
        - track stability (pose count)
        """

        if not persons:
            return []

        return max(persons, key=lambda p: p.tracking_score)
