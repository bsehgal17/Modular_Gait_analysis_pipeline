from pydantic import BaseModel, ConfigDict, Field, field_validator
from pose_estimation.enums.joint_enum import JointEnum


class PoseConfig(BaseModel):
    """
    Runtime configuration for pose feature extraction.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    frames_per_second: float = Field(gt=0)

    joint_schema: type[JointEnum]

    rolling_median_window: int = Field(default=5, gt=0)

    @field_validator("rolling_median_window")
    @classmethod
    def validate_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("rolling_median_window must be odd.")
        return value
