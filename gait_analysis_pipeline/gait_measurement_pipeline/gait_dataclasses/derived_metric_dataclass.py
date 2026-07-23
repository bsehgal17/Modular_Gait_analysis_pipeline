from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DerivedMetrics(BaseModel):
    """Derived gait metrics."""

    model_config = ConfigDict(frozen=True)

    avg_cadence: float | None = None
    avg_velocity: float | None = None
