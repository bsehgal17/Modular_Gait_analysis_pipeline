from __future__ import annotations

from functools import cached_property

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from pose_estimation.pose_estimation_dataclasses.frame_detection_dataclass import (
    FrameDetection,
)
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
)
from pose_estimation.enums.joint_enum import StandardJoint


JOINT_ORDER = tuple(StandardJoint)


class TrackedPerson(BaseModel):
    person_id: int

    detections: list[FrameDetection] = Field(default_factory=list)
    poses: list[FramePose] = Field(default_factory=list)

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    @model_validator(mode="after")
    def validate_unique_pose_frames(self):
        frames = [p.frame_idx for p in self.poses]

        if len(frames) != len(set(frames)):
            raise ValueError("Duplicate pose frame_idx values")

        return self

    # -------------------------------------------------
    # CACHED SORTED VIEWS
    # -------------------------------------------------
    @cached_property
    def sorted_poses(self) -> list[FramePose]:
        return sorted(
            self.poses,
            key=lambda p: p.frame_idx,
        )

    @cached_property
    def sorted_detections(self) -> list[FrameDetection]:
        return sorted(
            self.detections,
            key=lambda d: d.frame_idx,
        )

    # -------------------------------------------------
    # MUTATION API
    # -------------------------------------------------
    def add_detection(
        self,
        det: FrameDetection,
    ):
        self.detections.append(det)

        # invalidate caches
        self.__dict__.pop("sorted_detections", None)

    def add_pose(
        self,
        pose: FramePose,
    ):
        self.poses.append(pose)

        # invalidate caches
        self.__dict__.pop("sorted_poses", None)

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------
    @property
    def tracking_score(self) -> float:
        if not self.detections:
            return 0.0

        return sum(d.score for d in self.detections) / len(self.detections)

    @property
    def num_frames(self) -> int:
        return len(self.poses)

    # -------------------------------------------------
    # NUMPY EXPORTS
    # -------------------------------------------------
    def poses_to_numpy(self) -> np.ndarray:
        """
        Shape:
            (frames, joints, 3)

        Each joint:
            [x, y, z]
        """

        if not self.poses:
            return np.empty(
                (0, 0, 3),
                dtype=np.float64,
            )

        arr = np.full(
            (
                len(self.sorted_poses),
                len(JOINT_ORDER),
                3,
            ),
            np.nan,
            dtype=np.float64,
        )

        for i, pose in enumerate(self.sorted_poses):
            for j, joint_name in enumerate(JOINT_ORDER):
                joint = pose.joint_map.get(joint_name)

                if joint is None:
                    continue

                kp = joint.keypoint

                arr[i, j, 0] = kp.x
                arr[i, j, 1] = kp.y
                arr[i, j, 2] = kp.z

        return arr

    def detections_to_numpy(self) -> np.ndarray:
        """
        Shape:
            (N, 5)

        Columns:
            [x1, y1, x2, y2, score]
        """

        if not self.detections:
            return np.empty(
                (0, 5),
                dtype=np.float64,
            )

        return np.asarray(
            [
                [
                    d.x1,
                    d.y1,
                    d.x2,
                    d.y2,
                    d.score,
                ]
                for d in self.sorted_detections
            ],
            dtype=np.float64,
        )
