from typing import List, Tuple
import pandas as pd
import numpy as np
from gait_measurement_pipeline.gait_dataclasses.temporal_metric_dataclass import (
    TemporalEvent,
    TemporalMetrics,
    TimedSideEvent,
)


class TemporalEventBuilder:
    """
    Converts GaitRite DataFrame into the modernized TemporalMetrics dataclass.
    Ensures all events contain 'value' for JSON and Excel serialization.
    """

    def build_temporal_metrics(self, df: pd.DataFrame) -> TemporalMetrics:
        df = df.sort_values("First Contact Time").reset_index(drop=True)

        left_stance, right_stance = self._build_stance_events(df)
        left_stride, right_stride = self._build_stride_events(df)

        step_times = self._build_step_events(df)
        swing_times = self._build_swing_events(df)
        stride_times = sorted(left_stride + right_stride, key=lambda x: x.start_time)
        stance_times = sorted(left_stance + right_stance, key=lambda x: x.start_time)

        # Wrap raw floats into TimedSideEvent objects
        fc_times = self._build_contact_events(df, contact_col="First Contact Time")
        lc_times = self._build_contact_events(df, contact_col="Last Contact Time")

        return TemporalMetrics(
            left_stance=left_stance,
            right_stance=right_stance,
            left_stride=left_stride,
            right_stride=right_stride,
            step_times=step_times,
            swing_times=swing_times,
            stride_times=stride_times,
            stance_times=stance_times,
            first_contacts=fc_times,
            last_contacts=lc_times,
        )

    def _build_contact_events(
        self, df: pd.DataFrame, contact_col: str
    ) -> List[TimedSideEvent]:
        """Wraps raw contact timestamps into TimedSideEvent objects."""
        events = []
        for i in range(len(df)):
            val = df.iloc[i][contact_col]
            if pd.isna(val):
                continue

            side_val = df.iloc[i]["Left/Right Foot"]
            side_label = "left" if side_val in [0, "L", "l"] else "right"

            events.append(
                TimedSideEvent(
                    time=float(val),
                    side=side_label,
                )
            )
        return events

    def _build_stance_events(
        self, df: pd.DataFrame
    ) -> Tuple[List[TemporalEvent], List[TemporalEvent]]:
        """
        Build stance events split by side, similar to the stride builder.
        Each event represents the duration a single foot is on the ground.
        """
        left_stance: List[TemporalEvent] = []
        right_stance: List[TemporalEvent] = []

        for i in range(len(df)):
            val = df.iloc[i]["Stance Time"]
            if pd.isna(val):
                continue

            side_val = df.iloc[i]["Left/Right Foot"]
            side_label = "left" if side_val in [0, "L", "l"] else "right"

            # Timing for a STANCE is within the same footfall: FC to LC
            event = TemporalEvent(
                start_time=float(df.iloc[i]["First Contact Time"]),
                end_time=float(df.iloc[i]["Last Contact Time"]),
                side=side_label,
                value=float(val),
            )

            if side_label == "left":
                left_stance.append(event)
            else:
                right_stance.append(event)

        return left_stance, right_stance

    def _build_stride_events(
        self, df: pd.DataFrame
    ) -> Tuple[List[TemporalEvent], List[TemporalEvent]]:
        """Stride = Time between same-foot contacts."""
        left_stride, right_stride = [], []
        for i in range(2, len(df)):
            val = df.iloc[i]["Stride Time"]
            if pd.isna(val):
                continue

            side_val = df.iloc[i]["Left/Right Foot"]
            side_label = "left" if side_val in [0, "L", "l"] else "right"

            event = TemporalEvent(
                start_time=float(df.iloc[i - 2]["First Contact Time"]),
                end_time=float(df.iloc[i]["First Contact Time"]),
                side=side_label,
                value=float(val),
            )

            if side_label == "left":
                left_stride.append(event)
            else:
                right_stride.append(event)
        return left_stride, right_stride

    def _build_step_events(self, df: pd.DataFrame) -> List[TemporalEvent]:
        """Step = Time between alternating foot contacts."""
        steps = []
        for i in range(1, len(df)):
            val = df.iloc[i]["Step Time"]
            if pd.isna(val):
                continue

            steps.append(
                TemporalEvent(
                    start_time=float(df.iloc[i - 1]["First Contact Time"]),
                    end_time=float(df.iloc[i]["First Contact Time"]),
                    side="bilateral",
                    value=float(val),
                )
            )
        return steps

    def _build_swing_events(self, df: pd.DataFrame) -> List[TemporalEvent]:
        """Swing = Time foot spends in the air."""
        swings = []
        for i in range(len(df)):
            val = df.iloc[i]["Swing Time"]
            if pd.isna(val):
                continue

            side_val = df.iloc[i]["Left/Right Foot"]
            side_label = "left" if side_val in [0, "L", "l"] else "right"

            end = float(df.iloc[i]["First Contact Time"])
            swings.append(
                TemporalEvent(
                    start_time=end - float(val),
                    end_time=end,
                    side=side_label,
                    value=float(val),
                )
            )
        return sorted(swings, key=lambda x: x.start_time)
