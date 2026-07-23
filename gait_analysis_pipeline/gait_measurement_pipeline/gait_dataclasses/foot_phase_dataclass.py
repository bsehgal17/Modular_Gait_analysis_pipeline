from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Coordinate(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    frame_idx: int


class FootKeypoints(BaseModel):
    """
    all lower foot keypoints for one foot across all frames.
    """

    model_config = ConfigDict(frozen=True)

    heel: list[Coordinate]
    ankle: list[Coordinate]
    toe: list[Coordinate]

    @property
    def full_avg(self) -> list[Coordinate]:
        return [
            Coordinate(
                x=(h.x + a.x + t.x) / 3.0,
                y=(h.y + a.y + t.y) / 3.0,
                frame_idx=h.frame_idx,
            )
            for h, a, t in zip(self.heel, self.ankle, self.toe)
        ]

    @property
    def proximal_avg(self) -> list[Coordinate]:
        return [
            Coordinate(
                x=(h.x + a.x) / 2.0,
                y=(h.y + a.y) / 2.0,
                frame_idx=h.frame_idx,
            )
            for h, a in zip(self.heel, self.ankle)
        ]

    def filter_by_mask(
        self, landmark: list[Coordinate], mask: list[bool]
    ) -> list[Coordinate]:
        """Generic helper to filter any landmark list by a boolean mask."""
        return [c for c, s in zip(landmark, mask) if s]

    def stance_points(
        self, mask: list[bool], landmark: list[Coordinate] | None = None
    ) -> list[Coordinate]:
        """
        Return coordinates for stance frames.
        Defaults to heel if no landmark list is provided.
        """
        return self.filter_by_mask(
            landmark if landmark is not None else self.heel, mask
        )

    def swing_points(
        self, mask: list[bool], landmark: list[Coordinate] | None = None
    ) -> list[Coordinate]:
        """
        Return coordinates for swing frames.
        Defaults to heel if no landmark list is provided.
        """
        swing_mask = [not s for s in mask]
        return self.filter_by_mask(
            landmark if landmark is not None else self.heel, swing_mask
        )


class FootPhaseData(BaseModel):
    """
    all foot keypoints + phase masks for one foot.
    """

    model_config = ConfigDict(frozen=True)

    foot_keypoints: FootKeypoints
    stance_mask: list[bool]

    @property
    def swing_mask(self) -> list[bool]:
        return [not s for s in self.stance_mask]

    @property
    def stance_heel(self) -> list[Coordinate]:
        return self.foot_keypoints.stance_points(
            self.stance_mask, self.foot_keypoints.heel
        )

    @property
    def swing_heel(self) -> list[Coordinate]:
        return self.foot_keypoints.swing_points(
            self.stance_mask, self.foot_keypoints.heel
        )

    @property
    def stance_ankle(self) -> list[Coordinate]:
        return self.foot_keypoints.stance_points(
            self.stance_mask, self.foot_keypoints.ankle
        )

    @property
    def swing_ankle(self) -> list[Coordinate]:
        return self.foot_keypoints.swing_points(
            self.stance_mask, self.foot_keypoints.ankle
        )

    @property
    def stance_toe(self) -> list[Coordinate]:
        return self.foot_keypoints.stance_points(
            self.stance_mask, self.foot_keypoints.toe
        )

    @property
    def swing_toe(self) -> list[Coordinate]:
        return self.foot_keypoints.swing_points(
            self.stance_mask, self.foot_keypoints.toe
        )

    # --- Averaged phase splits ---

    @property
    def stance_full_avg(self) -> list[Coordinate]:
        return self.foot_keypoints.stance_points(
            self.stance_mask, self.foot_keypoints.full_avg
        )

    @property
    def swing_full_avg(self) -> list[Coordinate]:
        return self.foot_keypoints.swing_points(
            self.stance_mask, self.foot_keypoints.full_avg
        )

    @property
    def stance_proximal_avg(self) -> list[Coordinate]:
        return self.foot_keypoints.stance_points(
            self.stance_mask, self.foot_keypoints.proximal_avg
        )

    @property
    def swing_proximal_avg(self) -> list[Coordinate]:
        return self.foot_keypoints.swing_points(
            self.stance_mask, self.foot_keypoints.proximal_avg
        )


class FootVelocity(BaseModel):
    """
    Computed velocities for one foot.
    Each value is a scalar speed (magnitude) per frame in pixels/second.
    """

    model_config = ConfigDict(frozen=True)

    full_avg: list[float]  # speed of avg(heel, ankle, toe) per frame
    proximal_avg: list[float]  # speed of avg(heel, ankle) per frame
    toe: list[float]  # speed of toe joint per frame

    # --- add these two methods ---
    def to_signal_map(self) -> dict[str, list[float]]:
        return {
            "full_avg": self.full_avg,
            "proximal_avg": self.proximal_avg,
            "toe": self.toe,
        }

    @classmethod
    def from_signal_map(cls, signals: dict[str, list[float]]) -> "FootVelocity":
        return cls(**signals)


class FootExtremes(BaseModel):
    """
    Detected swing peaks and stance valleys for one foot.
    Each value is a frame index.

    peaks   - frame indices where swing velocity is at a local maximum.
    valleys - frame indices where stance velocity is at a local minimum
              (foot planted, near-zero speed). Detected as minima, not
              zero-crossings.
    """

    model_config = ConfigDict(frozen=True)

    peaks: list[int]  # frame indices of swing velocity peaks
    valleys: list[int]  # frame indices of stance velocity valleys (minima)
