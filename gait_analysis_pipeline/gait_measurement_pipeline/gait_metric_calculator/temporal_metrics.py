from __future__ import annotations

from typing import Literal

from gait_measurement_pipeline.gait_dataclasses.contact_detection_dataclass import (
    GaitEventsResult,
)
from gait_measurement_pipeline.gait_dataclasses.temporal_metric_dataclass import (
    TimedSideEvent,
)
from gait_measurement_pipeline.gait_dataclasses.temporal_metric_dataclass import (
    TemporalEvent,
    TemporalMetrics,
)
from gait_measurement_pipeline.gait_metric_calculator.temporal_event_utils import (
    build_double_support,
    build_phase_events,
)

Side = Literal["left", "right"]


class TemporalMetricsComputer:
    def __init__(self, results: GaitEventsResult) -> None:
        self.results = results

        self.left = results.left
        self.right = results.right

        self.left_first_contacts = self.left.First_contact_times(
            fps=results.fps,
        )

        self.right_first_contacts = self.right.First_contact_times(
            fps=results.fps,
        )

        self.left_last_contacts = self.left.Last_contact_times(
            fps=results.fps,
        )

        self.right_last_contacts = self.right.Last_contact_times(
            fps=results.fps,
        )

        self.sorted_first_contacts = self._build_timed_events(
            self.left_first_contacts,
            self.right_first_contacts,
        )

        self.sorted_last_contacts = self._build_timed_events(
            self.left_last_contacts,
            self.right_last_contacts,
        )

    def compute(self) -> TemporalMetrics:
        left_stance = build_phase_events(
            self.left_first_contacts,
            self.left_last_contacts,
            "left",
        )

        right_stance = build_phase_events(
            self.right_first_contacts,
            self.right_last_contacts,
            "right",
        )

        return TemporalMetrics(
            left_stance=left_stance,
            right_stance=right_stance,
            left_stride=self._compute_stride(
                self.left_first_contacts,
                "left",
            ),
            right_stride=self._compute_stride(
                self.right_first_contacts,
                "right",
            ),
            step_times=self._compute_steps(),
            swing_times=self._compute_swings(),
            stride_times=sorted(
                self._compute_stride(
                    self.left_first_contacts,
                    "left",
                )
                + self._compute_stride(
                    self.right_first_contacts,
                    "right",
                ),
                key=lambda event: event.start_time,
            ),
            stance_times=sorted(
                left_stance + right_stance,
                key=lambda event: event.start_time,
            ),
            double_support=build_double_support(
                left_stance,
                right_stance,
            ),
            first_contacts=self.sorted_first_contacts,
            last_contacts=self.sorted_last_contacts,
        )

    @staticmethod
    def _build_timed_events(
        left_times: list[float],
        right_times: list[float],
    ) -> list[TimedSideEvent]:
        events = [TimedSideEvent(time=t, side="left") for t in left_times] + [
            TimedSideEvent(time=t, side="right") for t in right_times
        ]

        return sorted(
            events,
            key=lambda event: event.time,
        )

    def _compute_steps(self) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []

        for i in range(len(self.sorted_first_contacts) - 1):
            start_event = self.sorted_first_contacts[i]
            end_event = self.sorted_first_contacts[i + 1]

            if start_event.side != end_event.side:
                events.append(
                    TemporalEvent(
                        start_time=start_event.time,
                        end_time=end_event.time,
                        value=end_event.time - start_event.time,
                        side="bilateral",
                    )
                )

        return events

    def _compute_swings(self) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []

        side_data = {
            "left": (
                self.left_last_contacts,
                self.left_first_contacts,
            ),
            "right": (
                self.right_last_contacts,
                self.right_first_contacts,
            ),
        }

        for side_name, (
            last_contacts,
            first_contacts,
        ) in side_data.items():
            for last_contact in last_contacts:
                next_first_contact = next(
                    (fc for fc in first_contacts if fc > last_contact),
                    None,
                )

                if next_first_contact is not None:
                    events.append(
                        TemporalEvent(
                            start_time=last_contact,
                            end_time=next_first_contact,
                            value=(next_first_contact - last_contact),
                            side=side_name,
                        )
                    )

        return sorted(
            events,
            key=lambda event: event.start_time,
        )

    def _compute_stride(
        self,
        first_contact_times: list[float],
        side: Side,
    ) -> list[TemporalEvent]:
        events: list[TemporalEvent] = []

        for i in range(len(first_contact_times) - 1):
            start_time = first_contact_times[i]
            end_time = first_contact_times[i + 1]

            events.append(
                TemporalEvent(
                    start_time=start_time,
                    end_time=end_time,
                    value=end_time - start_time,
                    side=side,
                )
            )

        return events
