from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Side = Literal["left", "right"]
EventSide = Literal["left", "right", "bilateral"]


class TimedSideEvent(BaseModel):
    """
    A timestamp associated with a limb side.
    """

    model_config = ConfigDict(frozen=True)

    time: float = Field(..., ge=0)
    side: Side


class TemporalEvent(BaseModel):
    """
    Represents a temporal gait interval.
    """

    model_config = ConfigDict(frozen=True)

    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., ge=0)

    side: EventSide

    value: float = Field(
        ...,
        ge=0,
        description="Duration of the event in seconds.",
    )


class TemporalMetrics(BaseModel):
    """
    Collection of temporal gait metrics.
    """

    model_config = ConfigDict(frozen=True)

    # Side-specific metrics
    left_stance: list[TemporalEvent]
    right_stance: list[TemporalEvent]

    left_stride: list[TemporalEvent]
    right_stride: list[TemporalEvent]

    # Combined metrics
    step_times: list[TemporalEvent]
    swing_times: list[TemporalEvent]

    stride_times: list[TemporalEvent]

    stance_times: list[TemporalEvent] = Field(
        default_factory=list,
        description="Combined stance phases from both feet.",
    )

    double_support: list[TemporalEvent] = Field(
        default_factory=list,
        description="Intervals where both feet are simultaneously in stance.",
    )

    # Traceability
    first_contacts: list[TimedSideEvent]

    last_contacts: list[TimedSideEvent]
