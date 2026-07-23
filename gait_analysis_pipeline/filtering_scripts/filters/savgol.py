from __future__ import annotations

import numpy as np
from pydantic import Field, field_validator
from scipy.signal import savgol_filter

from filtering_scripts.filter_dataclass import FilterConfig, SkelPointsFilter
from filtering_scripts.filter_enums import FilterType


# =========================================================
# FILTER IMPLEMENTATION
# =========================================================


class SavgolFilter(SkelPointsFilter):
    """
    Savitzky-Golay smoothing filter.
    Operates per-axis on (num_frames, N) data.
    """

    def __init__(self, window_length: int, polyorder: int):
        self._window_length = window_length
        self._polyorder = polyorder

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        window = self._window_length

        if len(data) < window:
            window = len(data) if len(data) % 2 == 1 else len(data) - 1

        if window < 3:
            return np.array(data)

        return np.stack(
            [
                savgol_filter(
                    data[:, i],
                    window_length=window,
                    polyorder=self._polyorder,
                )
                for i in range(data.shape[1])
            ],
            axis=1,
        )


# =========================================================
# CONFIG (SELF-CONTAINED BUILDER)
# =========================================================


class SavgolConfig(FilterConfig):
    filter_type: FilterType = Field(default=FilterType.SAVGOL)

    window_length: int = Field(gt=2, default=11)
    polynomial_order: int = Field(gt=0, default=2)

    @field_validator("window_length")
    @classmethod
    def validate_window(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("window_length must be odd.")
        return value

    def create_filter(self, fps: float) -> SavgolFilter:
        return SavgolFilter(
            window_length=self.window_length,
            polyorder=self.polynomial_order,
        )
