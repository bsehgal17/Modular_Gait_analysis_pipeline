from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import Coordinate

# ------------------------------------------------------------------
# Internal intermediate type
# ------------------------------------------------------------------


@dataclass
class DetectedEvents:
    """FC and LC frame indices returned by event detection / pairing steps."""

    First_contact: list[int]
    Last_contact: list[int]


# ------------------------------------------------------------------
# Spatial snapshots
# ------------------------------------------------------------------


class EventPoints(BaseModel):
    """Foot-landmark positions captured at gait events."""

    model_config = ConfigDict(frozen=True)

    heel: list[Coordinate]
    ankle: list[Coordinate]
    toe: list[Coordinate]


# ------------------------------------------------------------------
# One limb
# ------------------------------------------------------------------


class GaitSideEvents(BaseModel):
    """Gait events and phase masks for one limb."""

    model_config = ConfigDict(frozen=True)

    First_contact_frames: list[int]
    Last_contact_frames: list[int]

    stance_mask: list[bool]
    swing_mask: list[bool]

    First_contact_points: EventPoints | None = None
    Last_contact_points: EventPoints | None = None

    def First_contact_times(self, fps: float) -> list[float]:
        return [f / fps for f in self.First_contact_frames]

    def Last_contact_times(self, fps: float) -> list[float]:
        return [f / fps for f in self.Last_contact_frames]

    def stance_durations(self, fps: float) -> list[float]:
        """Duration of each stance phase (First_contact → Last_contact) in seconds."""
        n = min(len(self.First_contact_frames), len(self.Last_contact_frames))
        return [
            (self.Last_contact_frames[i] - self.First_contact_frames[i]) / fps
            for i in range(n)
        ]


# ------------------------------------------------------------------
# Bilateral result
# ------------------------------------------------------------------


class GaitEventsResult(BaseModel):
    """Bilateral gait event output for one recording."""

    model_config = ConfigDict(frozen=True)

    fps: float

    left: GaitSideEvents
    right: GaitSideEvents

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def double_support_mask(self) -> list[bool]:
        """Frames where both limbs are simultaneously in stance."""
        return [
            left and right
            for left, right in zip(self.left.stance_mask, self.right.stance_mask)
        ]

    def initiation_latency(self) -> float:
        """Time from video start until the first lift-off on either side."""
        left_first = (
            self.left.Last_contact_frames[0] / self.fps
            if self.left.Last_contact_frames
            else float("inf")
        )
        right_first = (
            self.right.Last_contact_frames[0] / self.fps
            if self.right.Last_contact_frames
            else float("inf")
        )
        return min(left_first, right_first)
