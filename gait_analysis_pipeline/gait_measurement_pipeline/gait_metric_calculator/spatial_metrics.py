from __future__ import annotations

from typing import Literal

from gait_measurement_pipeline.gait_dataclasses.contact_detection_dataclass import (
    GaitEventsResult,
    GaitSideEvents,
)
from gait_measurement_pipeline.gait_dataclasses.spatial_metric_dataclass import (
    SpatialMetrics,
)
from gait_measurement_pipeline.gait_metric_calculator.spatial_event_utils import (
    foot_center,
    build_stride_events,
    build_step_candidates,
)
from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import (
    FootKeypoints,
)
from types import SimpleNamespace

Side = Literal["left", "right"]


class SpatialMetricsComputer:
    """
    Computes spatial gait metrics:
    stride + step only (pure orchestration layer)
    """

    def __init__(
        self,
        results: GaitEventsResult,
        left_keypoints: FootKeypoints,
        right_keypoints: FootKeypoints,
    ) -> None:
        self.results = results
        self.left = results.left
        self.right = results.right
        self.left_keypoints = left_keypoints
        self.right_keypoints = right_keypoints

    def compute(self) -> SpatialMetrics:
        left_avg = self._average_keypoints(
            self._slice_keypoints(self.left, self.left_keypoints)
        )
        right_avg = self._average_keypoints(
            self._slice_keypoints(self.right, self.right_keypoints)
        )

        left_stride = self._stride(self.left, "left", left_avg)
        right_stride = self._stride(self.right, "right", right_avg)

        return SpatialMetrics(
            left_stride=left_stride,
            right_stride=right_stride,
            stride_lengths=sorted(
                left_stride + right_stride,
                key=lambda e: e.time,
            ),
            step_lengths=self._steps(left_avg, right_avg),
        )

    def _stride(
        self,
        side: GaitSideEvents,
        side_name: Side,
        avg_windows: list[dict] | None,
    ):
        if side.First_contact_points is None or not avg_windows:
            return []

        times = side.First_contact_times(self.results.fps)

        centers = [
            foot_center(
                self._to_point(avg["heel"]),
                self._to_point(avg["ankle"]),
            )
            for avg in avg_windows
        ]

        return build_stride_events(
            times=times,
            centers=centers,
            side=side_name,
        )

    def _steps(
        self,
        left_avg: list[dict] | None,
        right_avg: list[dict] | None,
    ):
        events = []

        for side_name, side, avg_windows in (
            ("left", self.left, left_avg),
            ("right", self.right, right_avg),
        ):
            if side.First_contact_points is None or not avg_windows:
                continue

            times = side.First_contact_times(self.results.fps)

            centers = [
                foot_center(
                    self._to_point(avg["heel"]),
                    self._to_point(avg["ankle"]),
                )
                for avg in avg_windows
            ]

            for t, c in zip(times, centers):
                events.append((side_name, t, c))

        return build_step_candidates(events)

    def _slice_keypoints(
        self,
        side: GaitSideEvents,
        keypoints: FootKeypoints | None,
    ) -> list[FootKeypoints] | None:
        """Return a list of FootKeypoints slices, one per contact window.
        Each window spans [mid_start, mid_last] where:
            mid       = (start + end) // 2
            mid_start = (start + mid) // 2
            mid_last  = (mid + end) // 2
        """
        if keypoints is None or side.First_contact_points is None:
            return None

        first_frames = [frame for frame in side.First_contact_frames]
        last_frames = [frame for frame in side.Last_contact_frames]

        if not first_frames or not last_frames:
            return None

        def _filter(coords, start_frame, end_frame):
            return [c for c in coords if start_frame <= c.frame_idx <= end_frame]

        windows = []
        for start, end in zip(first_frames, last_frames):
            mid = (start + end) // 2
            mid_start = (start + mid) // 2
            mid_last = (mid + end) // 2

            windows.append(
                FootKeypoints(
                    heel=_filter(keypoints.heel, mid_start, mid_last),
                    ankle=_filter(keypoints.ankle, mid_start, mid_last),
                    toe=_filter(keypoints.toe, mid_start, mid_last),
                )
            )

        return windows

    def _average_keypoints(
        self,
        windows: list[FootKeypoints] | None,
    ) -> list[dict] | None:
        """Return a list of averaged x, y values per window for each keypoint type."""
        if windows is None:
            return None

        averages = []
        for window in windows:
            window_avg = {}
            for part in ("heel", "ankle", "toe"):
                coords = getattr(window, part)
                if coords:
                    avg_x = sum(c.x for c in coords) / len(coords)
                    avg_y = sum(c.y for c in coords) / len(coords)
                else:
                    avg_x, avg_y = None, None

                window_avg[part] = {"x": avg_x, "y": avg_y}

            averages.append(window_avg)

        return averages

    def _to_point(self, d: dict) -> SimpleNamespace:
        return SimpleNamespace(x=d["x"], y=d["y"])
