from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

from pose_estimation.enums.joint_enum import JointEnum
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
    Joint,
)
from pose_estimation.pose_estimation_dataclasses.processing_steps_dataclass import (
    ProcessingStep,
)
from skeleton_normalizer.dataclasses.height_signal_smoother_base import HeightSignal
from pydantic import BaseModel, ConfigDict


class HeightEstimator(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    joints: type[JointEnum]

    _registry: ClassVar[list[type["HeightEstimator"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        HeightEstimator._registry.append(cls)

    @abstractmethod
    def _extract_chains(self, pose: FramePose) -> tuple[list[Joint], list[Joint]]: ...

    @property
    @abstractmethod
    def scale_factor(self) -> float: ...

    @classmethod
    @abstractmethod
    def required_joints(cls) -> set[str]: ...

    def estimate(self, poses: list[FramePose]) -> HeightSignal:
        """Return one height estimate per frame."""
        values: list[float] = []
        sf = self.scale_factor
        for pose in poses:
            left, right = self._extract_chains(pose)
            values.append(
                ((self._chain_length(left) + self._chain_length(right)) / 2) * sf
            )
        return HeightSignal(values=values)

    def _chain_length(self, chain: list[Joint]) -> float:
        """Returns the summed Euclidean length of a single frame's chain."""
        total = 0.0
        for a, b in zip(chain[:-1], chain[1:]):
            dx = a.keypoint.x - b.keypoint.x
            dy = a.keypoint.y - b.keypoint.y
            total += math.hypot(dx, dy)
        return total

    @classmethod
    def supports(cls, joints: type[JointEnum]) -> bool:
        return cls.required_joints().issubset(set(joints.__members__.keys()))

    def to_processing_step(self, smoother_params: dict) -> ProcessingStep:
        return ProcessingStep(
            step_name=self.__class__.name,
            params={"joint_schema": self.joints.__name__, **smoother_params},
        )
