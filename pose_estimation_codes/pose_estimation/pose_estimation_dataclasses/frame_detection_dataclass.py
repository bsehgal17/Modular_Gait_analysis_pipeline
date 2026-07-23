from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from pose_estimation.enums.pose_enums import ObjectLabel
from pose_estimation.pose_estimation_dataclasses.bbox_dataclass import BBox


class FrameDetection(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_idx: int

    bbox: BBox
    score: float
    label: ObjectLabel

    # -------------------------
    # derived helpers
    # -------------------------
    def iou(self, other: "FrameDetection") -> float:
        return self.bbox.iou(other.bbox)

    def distance(self, other: "FrameDetection") -> float:
        return self.bbox.center_distance(other.bbox)
