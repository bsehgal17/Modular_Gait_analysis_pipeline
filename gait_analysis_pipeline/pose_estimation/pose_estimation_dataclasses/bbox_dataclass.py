from __future__ import annotations

import math
from pydantic import BaseModel, ConfigDict, model_validator


class BBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x1: float
    y1: float
    x2: float
    y2: float

    # -------------------------
    # geometry
    # -------------------------
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    # -------------------------
    # metrics
    # -------------------------
    def iou(self, other: "BBox") -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - inter

        return 0.0 if union <= 0 else inter / union

    def center_distance(self, other: "BBox") -> float:
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return math.hypot(cx1 - cx2, cy1 - cy2)

    # -------------------------
    # validation
    # -------------------------
    @model_validator(mode="after")
    def validate_bbox(self):
        if self.x2 < self.x1:
            raise ValueError("x2 must be >= x1")
        if self.y2 < self.y1:
            raise ValueError("y2 must be >= y1")
        return self
