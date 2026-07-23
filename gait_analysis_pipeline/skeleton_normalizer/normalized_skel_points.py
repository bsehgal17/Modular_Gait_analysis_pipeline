from __future__ import annotations

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
    Keypoint,
)
from skeleton_normalizer.dataclasses.height_signal_smoother_base import HeightSignal


class PoseNormalizer:
    """
    Normalizes skeleton joint coordinates using a pre-computed HeightSignal.

    Accepts poses and a HeightSignal produced externally (e.g. by HeightPipeline),
    and returns a new list of FramePose objects with x/y coordinates divided by
    the corresponding per-frame height value. z and confidence are left unchanged.

    This class has no knowledge of how heights were estimated or smoothed — it
    only applies the scaling. Visualization is the caller's responsibility.
    """

    # =====================================================
    # PUBLIC
    # =====================================================

    def normalize(
        self,
        poses: list[FramePose],
        signal: HeightSignal,
    ) -> list[FramePose]:
        """
        Scale each pose's x/y joint coordinates by the corresponding height value.

        Parameters
        ----------
        poses : list[FramePose]
            Input poses, one per frame.
        signal : HeightSignal
            Smoothed height values, one per frame. Must align with ``poses``.

        Returns
        -------
        list[FramePose]
            New FramePose objects with normalized x/y coordinates.
            The original poses are not mutated.
        """
        if not poses:
            return poses

        if len(poses) != len(signal.values):
            raise ValueError(
                f"poses length ({len(poses)}) does not match "
                f"signal length ({len(signal.values)})"
            )

        return [
            self._normalize_pose(pose, height)
            for pose, height in zip(poses, signal.values)
        ]

    # =====================================================
    # PRIVATE
    # =====================================================

    def _normalize_pose(self, pose: FramePose, height: float) -> FramePose:
        """
        Return a copy of ``pose`` with every joint's x/y divided by ``height``.
        z and confidence are preserved unchanged.
        """
        return pose.model_copy(
            update={
                "joints": tuple(
                    joint.model_copy(
                        update={
                            "keypoint": Keypoint(
                                x=joint.keypoint.x / height,
                                y=joint.keypoint.y / height,
                                z=joint.keypoint.z,
                                confidence=joint.keypoint.confidence,
                            )
                        }
                    )
                    for joint in pose.joints
                )
            }
        )
