# frame_pose_dataclass.py

from __future__ import annotations

from functools import cached_property
import numpy as np

from pydantic import BaseModel, model_validator
from pose_estimation.enums.joint_enum import StandardJoint
from pose_estimation.enums.pose_enums import PoseModel
from pose_estimation.pose_estimation_dataclasses.bbox_dataclass import BBox


class Keypoint(BaseModel):
    x: float
    y: float
    z: float = 0.0
    confidence: float | None = None


class Joint(BaseModel):
    name: StandardJoint
    keypoint: Keypoint
    visibility: float = 1.0


class FramePose(BaseModel):
    frame_idx: int
    joints: tuple[Joint, ...]
    bbox: BBox
    bbox_score: float
    model: PoseModel

    # ------------------------------------
    # FAST INDEX
    # ------------------------------------
    @cached_property
    def joint_map(self) -> dict[StandardJoint, Joint]:
        return {j.name: j for j in self.joints}

    # ------------------------------------
    # FAST ACCESSORS
    # ------------------------------------
    def get_joint(self, name: StandardJoint) -> Joint | None:
        return self.joint_map.get(name)

    def get_keypoint(self, name: str) -> Keypoint | None:
        aliases = {
            "LEFT_TOE": ["LEFT_BIG_TOE", "LEFT_SMALL_TOE"],
            "LEFT_BIG_TOE": ["LEFT_BIG_TOE", "LEFT_TOE"],
            "LEFT_SMALL_TOE": ["LEFT_SMALL_TOE", "LEFT_TOE"],
            "RIGHT_TOE": ["RIGHT_BIG_TOE", "RIGHT_SMALL_TOE"],
            "RIGHT_BIG_TOE": ["RIGHT_BIG_TOE", "RIGHT_TOE"],
            "RIGHT_SMALL_TOE": ["RIGHT_SMALL_TOE", "RIGHT_TOE"],
        }

        name_upper = name.upper()
        names_to_try = aliases.get(name_upper, [name_upper])

        for n in names_to_try:
            joint = self.joint_map.get(n)
            if joint is not None:
                return joint.keypoint

        return None

    # ------------------------------------
    # NUMPY CONVERSION
    # ------------------------------------
    def to_xy_array(self) -> np.ndarray:
        """
        Returns shape (J, 2) — x/y for every joint in declaration order.
        Missing keypoints are not possible here (all joints are present),
        but z is intentionally excluded — normalizer only needs 2-D.
        """
        return np.array(
            [[j.keypoint.x, j.keypoint.y] for j in self.joints],
            dtype=np.float64,
        )

    @staticmethod
    def sequence_to_array(poses: list["FramePose"]) -> np.ndarray:
        """
        Converts list[FramePose] → np.ndarray shape (F, J, 2).

        All frames must share the same joint count.
        This is the single source of truth for (F, J, 2) extraction
        used by filtering, normalization, and height estimation.
        """
        if not poses:
            return np.empty((0, 0, 2), dtype=np.float64)

        return np.stack(
            [pose.to_xy_array() for pose in poses],
            axis=0,
        )  # shape: (F, J, 2)

    # ------------------------------------
    # VALIDATION
    # ------------------------------------
    @model_validator(mode="after")
    def validate_unique_joints(self):
        names = [j.name for j in self.joints]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate joints detected")
        return self

    # ------------------------------------
    # MUTATION
    # ------------------------------------
    def with_updated_keypoints(self, xy_array: np.ndarray) -> "FramePose":
        """
        Returns a new FramePose with keypoints replaced by xy_array.
        xy_array shape: (J, 2)
        """
        updated_joints = tuple(
            joint.model_copy(
                update={
                    "keypoint": joint.keypoint.model_copy(
                        update={
                            "x": float(xy_array[j, 0]),
                            "y": float(xy_array[j, 1]),
                        }
                    )
                }
            )
            for j, joint in enumerate(self.joints)
        )
        return self.model_copy(update={"joints": updated_joints})
