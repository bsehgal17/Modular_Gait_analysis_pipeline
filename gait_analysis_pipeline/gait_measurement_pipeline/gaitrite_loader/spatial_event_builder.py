from typing import List, Tuple, Optional
import pandas as pd

from gait_measurement_pipeline.gait_dataclasses.spatial_metric_dataclass import (
    SpatialEvent,
    SpatialMetrics,
)


class SpatialEventBuilder:
    """
    Builds production-grade spatial gait events from GaitRite data.

    Spatial metrics can optionally be normalized by participant height:
        normalized_value = value / height
    """

    def __init__(self, height: Optional[float] = None) -> None:
        """
        Args:
            height:
                Participant height used for normalization.
                If None, values are returned unchanged.
        """
        if height is not None and height <= 0:
            raise ValueError("Height must be > 0")

        self.height = height

    def _normalize(self, value: float) -> float:
        """
        Normalize a spatial value by participant height if provided.
        """
        return value / self.height if self.height is not None else value

    def build_spatial_metrics(self, df: pd.DataFrame) -> SpatialMetrics:
        """
        Main entry point: Build the full SpatialMetrics container.
        """
        left_strides, right_strides = self._build_strides(df)
        step_lengths = self._build_steps(df)

        stride_lengths = sorted(
            left_strides + right_strides,
            key=lambda x: x.time,
        )

        return SpatialMetrics(
            left_stride=left_strides,
            right_stride=right_strides,
            stride_lengths=stride_lengths,
            step_lengths=step_lengths,
        )

    def _build_steps(self, df: pd.DataFrame) -> List[SpatialEvent]:
        heel_x = df["Heel X"].dropna().tolist()
        heel_y = df["Heel Y"].dropna().tolist()
        values = df["Step Length"].dropna().tolist()
        times = df["First Contact Time"].dropna().tolist()

        events: List[SpatialEvent] = []

        for i in range(1, min(len(heel_x), len(values), len(times))):
            side = "right" if i % 2 != 0 else "left"

            events.append(
                SpatialEvent(
                    start_pos=[heel_x[i - 1], heel_y[i - 1]],
                    end_pos=[heel_x[i], heel_y[i]],
                    value=self._normalize(float(values[i])),
                    time=float(times[i]),
                    side=side,
                )
            )

        return events

    def _build_strides(
        self, df: pd.DataFrame
    ) -> Tuple[List[SpatialEvent], List[SpatialEvent]]:
        heel_x = df["Heel X"].dropna().tolist()
        heel_y = df["Heel Y"].dropna().tolist()
        values = df["Stride Length"].dropna().tolist()
        times = df["First Contact Time"].dropna().tolist()

        left_strides: List[SpatialEvent] = []
        right_strides: List[SpatialEvent] = []

        for i in range(2, min(len(heel_x), len(values), len(times))):
            side = "left" if i % 2 == 0 else "right"

            event = SpatialEvent(
                start_pos=[heel_x[i - 2], heel_y[i - 2]],
                end_pos=[heel_x[i], heel_y[i]],
                value=self._normalize(float(values[i])),
                time=float(times[i]),
                side=side,
            )

            if side == "left":
                left_strides.append(event)
            else:
                right_strides.append(event)

        return left_strides, right_strides
