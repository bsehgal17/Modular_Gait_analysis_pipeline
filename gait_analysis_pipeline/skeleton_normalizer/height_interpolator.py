# height_outlier_interpolation_processor.py
from __future__ import annotations

import numpy as np

from skeleton_normalizer.dataclasses.height_signal_smoother_base import (
    HeightSignal,
)


class HeightOutlierInterpolator:
    """
    Detects sudden spikes in a 1D height signal using a rolling median
    deviation, nulls them out, and linearly interpolates the gaps.
    """

    def __init__(
        self,
        window_size: int,
        z_thresh: float,
    ):
        self._window_size = window_size
        self._z_thresh = z_thresh

    def filter(self, height_signal: HeightSignal) -> HeightSignal:
        heights = np.array(height_signal.values, dtype=float)
        n = len(heights)

        if n < self._window_size:
            return height_signal

        outlier_mask = self._detect_outliers(heights)

        if outlier_mask.any():
            heights = self._interpolate(heights, outlier_mask)

        return HeightSignal(values=heights.tolist())

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _detect_outliers(self, heights: np.ndarray) -> np.ndarray:
        n = len(heights)
        half = self._window_size // 2
        outlier_mask = np.zeros(n, dtype=bool)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = heights[lo:hi]

            median = np.median(window)
            mad = np.median(np.abs(window - median))

            if mad == 0:
                outlier_mask[i] = heights[i] != median
            else:
                outlier_mask[i] = abs(heights[i] - median) / mad > self._z_thresh

        return outlier_mask

    def _interpolate(self, heights: np.ndarray, mask: np.ndarray) -> np.ndarray:
        heights = heights.copy()
        heights[mask] = np.nan

        if np.isnan(heights).all():
            return np.zeros_like(heights)

        valid = ~np.isnan(heights)
        indices = np.arange(len(heights))
        heights = np.interp(indices, indices[valid], heights[valid])
        return heights
