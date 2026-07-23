from __future__ import annotations

from gait_measurement_pipeline.gait_dataclasses.temporal_metric_dataclass import (
    TemporalEvent,
)


def build_phase_events(
    starts: list[float],
    ends: list[float],
    side: str,
) -> list[TemporalEvent]:
    events: list[TemporalEvent] = []

    for start_time, end_time in zip(starts, ends):
        if end_time > start_time:
            events.append(
                TemporalEvent(
                    start_time=start_time,
                    end_time=end_time,
                    value=end_time - start_time,
                    side=side,
                )
            )

    return events


def build_double_support(
    left_stances: list[TemporalEvent],
    right_stances: list[TemporalEvent],
) -> list[TemporalEvent]:
    events: list[TemporalEvent] = []

    for left_event in left_stances:
        for right_event in right_stances:
            overlap_start = max(
                left_event.start_time,
                right_event.start_time,
            )

            overlap_end = min(
                left_event.end_time,
                right_event.end_time,
            )

            if overlap_start < overlap_end:
                events.append(
                    TemporalEvent(
                        start_time=overlap_start,
                        end_time=overlap_end,
                        value=overlap_end - overlap_start,
                        side="bilateral",
                    )
                )

    return sorted(
        events,
        key=lambda event: event.start_time,
    )
