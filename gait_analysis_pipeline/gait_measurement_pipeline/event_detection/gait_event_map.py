from pose_estimation.enums.joint_enum import JointEnum
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from gait_measurement_pipeline.gait_dataclasses.contact_detection_dataclass import (
    EventPoints,
)
from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import Coordinate


class GaitEventMapper:
    def __init__(self, joint_enum: type[JointEnum]):
        self._joints = joint_enum

    def joint(self, side: str, name: str) -> JointEnum:
        return getattr(self._joints, f"{side.upper()}_{name.upper()}")

    def build_pose_index(self, poses: list[FramePose] | None) -> dict[int, FramePose]:
        return {} if poses is None else {p.frame_idx: p for p in poses}

    def extract_points(
        self,
        indices: list[int],
        pose_index: dict[int, FramePose],
        side: str,
    ) -> EventPoints:

        def coords(joint: JointEnum) -> list[Coordinate]:
            out = []
            for f in indices:
                kp = pose_index[f].get_keypoint(joint.name)
                out.append(Coordinate(x=kp.x, y=kp.y, frame_idx=f))
            return out

        heel = self.joint(side, "heel")
        ankle = self.joint(side, "ankle")
        toe = self.joint(side, "big_toe")

        return EventPoints(
            heel=coords(heel),
            ankle=coords(ankle),
            toe=coords(toe),
        )
