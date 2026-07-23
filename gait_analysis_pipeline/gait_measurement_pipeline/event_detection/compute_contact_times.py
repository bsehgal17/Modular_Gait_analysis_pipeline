from __future__ import annotations

from typing import List, Optional

from gait_measurement_pipeline.gait_dataclasses.contact_detection_dataclass import (
    GaitEventsResult,
    GaitSideEvents,
)

from gait_measurement_pipeline.gait_dataclasses.phase_detection_dataclass import (
    VelocityPhaseResult,
)

from gait_measurement_pipeline.event_detection.gait_event_core import (
    GaitEventDetectorCore,
)
from gait_measurement_pipeline.event_detection.gait_event_map import GaitEventMapper
from gait_measurement_pipeline.visualizations.gait_event_plot import GaitEventPlotter
from gait_measurement_pipeline.visualizations.gait_event_video_render import (
    GaitEventVideoRenderer,
)

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import (
    FootPhaseData,
)


class GaitEventDetector:
    def __init__(self, joint_enum, min_phase_frames: int = 5):
        self.core = GaitEventDetectorCore()
        self.mapper = GaitEventMapper(joint_enum)
        self.plotter = GaitEventPlotter()
        self.renderer = GaitEventVideoRenderer()
        self.min_phase_frames = min_phase_frames

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def detect(
        self,
        phases: VelocityPhaseResult,
        poses: Optional[List[FramePose]] = None,
        video_file: Optional[str] = None,
        do_plot: bool = False,
        do_video: bool = False,
    ) -> GaitEventsResult:

        time = list(phases.time)
        pose_index = self.mapper.build_pose_index(poses)

        left = self._process("left", phases.left, pose_index)
        right = self._process("right", phases.right, pose_index)

        result = GaitEventsResult(
            fps=phases.fps,
            left=left,
            right=right,
        )

        if do_plot:
            self.plot(time, result)

        if do_video and video_file is not None and poses is not None:
            self.render_video(video_file, result, poses)

        return result

    # ---------------------------------------------------------
    # INTERNAL PROCESSING
    # ---------------------------------------------------------

    def _process(
        self,
        side: str,
        foot_phase: FootPhaseData,
        pose_index: dict[int, FramePose],
    ) -> GaitSideEvents:

        stance = foot_phase.stance_mask

        events = self.core.detect(stance)
        events = self.core.pair(events)

        return GaitSideEvents(
            First_contact_frames=events.First_contact,
            Last_contact_frames=events.Last_contact,
            stance_mask=stance,
            swing_mask=foot_phase.swing_mask,
            First_contact_points=(
                self.mapper.extract_points(events.First_contact, pose_index, side)
                if pose_index
                else None
            ),
            Last_contact_points=(
                self.mapper.extract_points(events.Last_contact, pose_index, side)
                if pose_index
                else None
            ),
        )

    # ---------------------------------------------------------
    # VISUALIZATION WRAPPERS
    # ---------------------------------------------------------

    def plot(self, time: List[float], result: GaitEventsResult) -> None:
        self.plotter.plot(time, result)

    def render_video(
        self,
        video_file: str,
        result: GaitEventsResult,
        poses: List[FramePose],
    ) -> None:
        self.renderer.render(video_file, result, poses, self.mapper)
