# velocity_outlier_interpolation_processor.py
from __future__ import annotations

import numpy as np

from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import (
    FootVelocity,
)


class VelocityOutlierInterpolator:
    def __init__(self, window_size: int = 5, z_thresh: float = 2.0):
        self._window_size = window_size
        self._z_thresh = z_thresh

    def filter(self, foot_velocity: FootVelocity) -> FootVelocity:
        return FootVelocity(
            full_avg=self._clean(foot_velocity.full_avg),
            proximal_avg=self._clean(foot_velocity.proximal_avg),
            toe=self._clean(foot_velocity.toe),
        )

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _clean(self, values: list[float]) -> list[float]:
        arr = np.array(values, dtype=float)

        if len(arr) < self._window_size:
            return values

        mask = self._detect_outliers(arr)

        if mask.any():
            arr = self._interpolate(arr, mask)

        return arr.tolist()

    def _detect_outliers(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        half = self._window_size // 2
        mask = np.zeros(n, dtype=bool)

        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            window = series[lo:hi]

            median = np.median(window)
            mad = np.median(np.abs(window - median))

            if mad == 0:
                mask[i] = series[i] != median
            else:
                mask[i] = abs(series[i] - median) / mad > self._z_thresh

        return mask

    def _interpolate(self, series: np.ndarray, mask: np.ndarray) -> np.ndarray:
        series = series.copy()
        series[mask] = np.nan

        if np.isnan(series).all():
            return np.zeros_like(series)

        valid = ~np.isnan(series)
        indices = np.arange(len(series))
        return np.interp(indices, indices[valid], series[valid])
