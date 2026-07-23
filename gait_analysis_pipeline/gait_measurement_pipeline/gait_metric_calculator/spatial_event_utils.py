from __future__ import annotations

from gait_measurement_pipeline.gait_dataclasses.spatial_metric_dataclass import (
    SpatialEvent,
)


def foot_center(heel: list[float], ankle: list[float]) -> list[float]:
    return [
        (heel.x + ankle.x) / 2,
        (heel.y + ankle.y) / 2,
    ]


def distance(a: list[float], b: list[float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return (dx * dx + dy * dy) ** 0.5


def build_stride_events(
    times: list[float],
    centers: list[list[float]],
    side: str,
) -> list[SpatialEvent]:
    events: list[SpatialEvent] = []

    for i in range(len(times) - 1):
        events.append(
            SpatialEvent(
                start_pos=centers[i],
                end_pos=centers[i + 1],
                value=distance(centers[i], centers[i + 1]),
                time=times[i + 1],
                side=side,
            )
        )

    return events


def build_step_candidates(
    all_events: list[tuple[str, float, list[float]]],
) -> list[SpatialEvent]:
    """
    Input: (side, time, position)
    """

    all_events = sorted(all_events, key=lambda x: x[1])

    steps: list[SpatialEvent] = []

    for i in range(len(all_events) - 1):
        side_a, _, pos_a = all_events[i]
        side_b, time_b, pos_b = all_events[i + 1]

        if side_a != side_b:
            steps.append(
                SpatialEvent(
                    start_pos=pos_a,
                    end_pos=pos_b,
                    value=distance(pos_a, pos_b),
                    time=time_b,
                    side=side_b,
                )
            )

    return steps
