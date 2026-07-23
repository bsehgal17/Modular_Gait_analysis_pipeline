from __future__ import annotations

from pydantic import BaseModel

from pose_estimation.enums.joint_enum import StandardJoint
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
    Joint,
)


class ChainMixin(BaseModel):
    """
    Skeletal chain slicing utilities.
    Extracts a list of Joints from a FramePose, by joint NAME -- works
    regardless of which raw schema enum (PredJointsCOCOWholebody, Mediapipe,
    StandardJoint, etc.) the caller's `joint_members` come from, and is
    immune to however FramePose.joints happens to be ordered.
    """

    def _build(self, pose: FramePose, joint_members) -> list[Joint]:
        joints = []
        for member in joint_members:
            try:
                standard_name = StandardJoint[member.name]
            except KeyError:
                raise KeyError(f"'{member.name}' has no corresponding StandardJoint")

            joint = pose.get_joint(standard_name)
            if joint is None:
                raise KeyError(
                    f"Joint '{standard_name}' not present on this FramePose "
                    f"(model={pose.model})."
                )
            joints.append(joint)
        return joints

    def _chain(self, pose: FramePose, *joint_members) -> list[Joint]:
        return self._build(pose, list(joint_members))
