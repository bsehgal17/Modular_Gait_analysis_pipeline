from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy.signal import butter, sosfiltfilt

from filtering_scripts.filter_dataclass import FilterConfig, SkelPointsFilter
from filtering_scripts.filter_enums import FilterType


# =========================================================
# FILTER IMPLEMENTATION
# =========================================================


class ButterworthFilter(SkelPointsFilter):
    """
    Zero-phase Butterworth low-pass filter.
    Operates per-axis on (num_frames, N) data.
    """

    def __init__(self, cutoff: float, fs: float, order: int):
        self._sos = butter(
            order,
            cutoff / (0.5 * fs),
            btype="low",
            output="sos",
        )
        self._min_length = 3 * (2 * order - 1)

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        if len(data) <= self._min_length:
            return np.array(data)

        return np.stack(
            [sosfiltfilt(self._sos, data[:, i]) for i in range(data.shape[1])],
            axis=1,
        )


# =========================================================
# CONFIG (SELF-CONTAINED BUILDER)
# =========================================================


class ButterworthConfig(FilterConfig):
    filter_type: FilterType = Field(default=FilterType.BUTTERWORTH)

    cutoff_frequency: float = Field(gt=0, default=1.0)
    filter_order: int = Field(gt=0, default=4)

    def create_filter(self, fps: float) -> ButterworthFilter:
        return ButterworthFilter(
            cutoff=self.cutoff_frequency,
            fs=fps,
            order=self.filter_order,
        )
