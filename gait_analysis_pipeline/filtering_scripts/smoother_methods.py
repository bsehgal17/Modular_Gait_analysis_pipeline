from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from filtering_scripts.filters.savgol import SavgolFilter


# =====================================================
# PROTOCOL — anything with list[float] fields can be smoothed
# =====================================================


@runtime_checkable
class Smoothable(Protocol):
    """
    Any dataclass whose float-list fields should be smoothed.
    Implement to_signal_map() to expose those fields, and
    from_signal_map() to reconstruct the object after smoothing.
    """

    def to_signal_map(self) -> dict[str, list[float]]: ...

    @classmethod
    def from_signal_map(cls, signals: dict[str, list[float]]) -> "Smoothable": ...


# =====================================================
# BASE
# =====================================================


class SignalSmoother(BaseModel):
    """
    Base smoother. Operates on:
      - list[float]          — via smooth_values()
      - any Smoothable       — via smooth(), which round-trips through
                               to_signal_map() / from_signal_map()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def smooth_values(self, values: list[float]) -> list[float]: ...

    def smooth(self, target: Smoothable) -> Smoothable:
        """
        Smooth every signal in a Smoothable (HeightSignal, FootVelocity, etc.).
        Returns a new instance of the same type.
        """
        smoothed = {
            key: self.smooth_values(values)
            for key, values in target.to_signal_map().items()
        }
        return target.from_signal_map(smoothed)

    @abstractmethod
    def get_params(self) -> dict: ...


# =====================================================
# IMPLEMENTATIONS
# =====================================================


class MedianSmoother(SignalSmoother):
    window_size: int

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("window_size must be positive.")
        if v % 2 == 0:
            raise ValueError("window_size must be odd.")
        return v

    def smooth_values(self, values: list[float]) -> list[float]:
        if self.window_size <= 1 or not values:
            return values

        half = self.window_size // 2
        n = len(values)
        smoothed: list[float] = []

        for i in range(n):
            window = sorted(values[max(0, i - half) : min(n, i + half + 1)])
            smoothed.append(float(window[len(window) // 2]))

        return smoothed

    def get_params(self) -> dict:
        return {"smoother": "median", "window_size": self.window_size}


class SavgolSmoother(SignalSmoother):
    window_length: int
    polyorder: int

    def smooth_values(self, values: list[float]) -> list[float]:
        if not values:
            return values

        arr = np.array(values).reshape(-1, 1)
        smoothed = SavgolFilter(
            window_length=self.window_length,
            polyorder=self.polyorder,
        ).filter_data(arr)
        return smoothed[:, 0].tolist()

    def get_params(self) -> dict:
        return {
            "smoother": "savgol",
            "window_length": self.window_length,
            "polyorder": self.polyorder,
        }
