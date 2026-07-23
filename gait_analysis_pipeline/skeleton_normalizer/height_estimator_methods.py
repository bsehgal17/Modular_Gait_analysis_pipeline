from __future__ import annotations

from typing import ClassVar

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
    Joint,
)
from skeleton_normalizer.dataclasses.height_estimator_base_dataclass import (
    HeightEstimator,
)
from skeleton_normalizer.chain_mix import ChainMixin


class HeadBasedHeightEstimator(HeightEstimator, ChainMixin):
    name: ClassVar[str] = "head_based"

    @property
    def scale_factor(self) -> float:
        return 1.0

    @classmethod
    def required_joints(cls) -> set[str]:
        return {
            "HEAD",
            "NECK",
            "LEFT_SHOULDER",
            "LEFT_HIP",
            "LEFT_KNEE",
            "LEFT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_SHOULDER",
            "RIGHT_HIP",
            "RIGHT_KNEE",
            "RIGHT_ANKLE",
            "RIGHT_HEEL",
        }

    def _extract_chains(self, pose: FramePose) -> tuple[list[Joint], list[Joint]]:
        j = self.joints
        left = self._chain(
            pose,
            j.HEAD,
            j.NECK,
            j.LEFT_SHOULDER,
            j.LEFT_HIP,
            j.LEFT_KNEE,
            j.LEFT_ANKLE,
            j.LEFT_HEEL,
        )
        right = self._chain(
            pose,
            j.HEAD,
            j.NECK,
            j.RIGHT_SHOULDER,
            j.RIGHT_HIP,
            j.RIGHT_KNEE,
            j.RIGHT_ANKLE,
            j.RIGHT_HEEL,
        )
        return left, right


class ShoulderBasedHeightEstimator(HeightEstimator, ChainMixin):
    name: ClassVar[str] = "shoulder_based"

    @property
    def scale_factor(self) -> float:
        return 1.0 / 0.818

    @classmethod
    def required_joints(cls) -> set[str]:
        return {
            "LEFT_SHOULDER",
            "LEFT_HIP",
            "LEFT_KNEE",
            "LEFT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_SHOULDER",
            "RIGHT_HIP",
            "RIGHT_KNEE",
            "RIGHT_ANKLE",
            "RIGHT_HEEL",
        }

    def _extract_chains(self, pose: FramePose) -> tuple[list[Joint], list[Joint]]:
        j = self.joints
        left = self._chain(
            pose, j.LEFT_SHOULDER, j.LEFT_HIP, j.LEFT_KNEE, j.LEFT_ANKLE, j.LEFT_HEEL
        )
        right = self._chain(
            pose,
            j.RIGHT_SHOULDER,
            j.RIGHT_HIP,
            j.RIGHT_KNEE,
            j.RIGHT_ANKLE,
            j.RIGHT_HEEL,
        )
        return left, right


class HipBasedHeightEstimator(HeightEstimator, ChainMixin):
    name: ClassVar[str] = "hip_based"

    @property
    def scale_factor(self) -> float:
        return 1.0 / 0.530

    @classmethod
    def required_joints(cls) -> set[str]:
        return {
            "LEFT_HIP",
            "LEFT_KNEE",
            "LEFT_ANKLE",
            "LEFT_HEEL",
            "RIGHT_HIP",
            "RIGHT_KNEE",
            "RIGHT_ANKLE",
            "RIGHT_HEEL",
        }

    def _extract_chains(self, pose: FramePose) -> tuple[list[Joint], list[Joint]]:
        j = self.joints
        left = self._chain(pose, j.LEFT_HIP, j.LEFT_KNEE, j.LEFT_ANKLE, j.LEFT_HEEL)
        right = self._chain(
            pose, j.RIGHT_HIP, j.RIGHT_KNEE, j.RIGHT_ANKLE, j.RIGHT_HEEL
        )
        return left, right
