from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any


class SpatialEvent(BaseModel):
    """
    Represents a spatial gait measurement (Step or Stride).
    """

    model_config = ConfigDict(frozen=True)

    start_pos: List[float] = Field(..., description="[x, y] coordinates at start")
    end_pos: List[float] = Field(..., description="[x, y] coordinates at end")
    value: float = Field(..., description="Calculated distance (e.g., Euclidean norm)")
    time: float = Field(..., description="Timestamp of completion (seconds)")
    side: str = Field(..., description="'left', 'right', or 'bilateral'")


class SpatialMetrics(BaseModel):
    """
    Collection of spatial gait metrics with statistical summaries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # Side-specific
    left_stride: List[SpatialEvent]
    right_stride: List[SpatialEvent]

    # Combined/Global
    stride_lengths: List[SpatialEvent] = Field(default_factory=list)
    step_lengths: List[SpatialEvent] = Field(default_factory=list)

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "left_stride": self.get_stats(self.left_stride),
            "right_stride": self.get_stats(self.right_stride),
            "step": self.get_stats(self.step_lengths),
            "combined_stride": self.get_stats(self.stride_lengths),
        }
