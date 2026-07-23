from __future__ import annotations

from pose_estimation.enums.joint_enum import JointEnum
from skeleton_normalizer.dataclasses.height_estimator_base_dataclass import (
    HeightEstimator,
)


class HeightEstimatorResolver:
    """
    Resolves compatible height estimator for a given joint schema.
    """

    def resolve(self, joint_schema: type[JointEnum]) -> HeightEstimator:
        for estimator_cls in HeightEstimator._registry:
            if estimator_cls.supports(joint_schema):
                return estimator_cls(joints=joint_schema)

        raise ValueError(
            f"No compatible height estimator found for schema: {joint_schema}"
        )
