from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from gait_measurement_pipeline.utils.compute_joint_speed import compute_joint_speed
from gait_measurement_pipeline.gait_dataclasses.contact_detection_dataclass import (
    GaitEventsResult,
    GaitSideEvents,
)
from gait_measurement_pipeline.phase_detection.foot_extractor import FootExtractor
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose


class EventVisualizer:
    """
    Visualizes gait events (FC/LC) and velocity profiles for both feet.

    Accepts list[FramePose] — one object per frame — matching the
    convention used throughout VelocityPhaseDetector and FootExtractor.
    """

    def __init__(self, joint_enum: Any) -> None:
        self._joints = joint_enum

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _get_toe_coords(self, poses: list[FramePose], side: str) -> np.ndarray:
        """
        Extract big-toe (x, y) for every frame as shape (N, 2).

        Uses get_keypoint(name) — identical to FootExtractor._to_coord —
        so no integer joint indexing.
        """
        attr = f"{side.upper()}_BIG_TOE"
        joint_name = getattr(self._joints, attr).name
        coords = [
            [pose.get_keypoint(joint_name).x, pose.get_keypoint(joint_name).y]
            for pose in poses
        ]
        return np.array(coords, dtype=np.float64)

    # --------------------------------------------------
    # 1. Velocity + event markers
    # --------------------------------------------------

    def plot_velocity_events(
        self,
        time: list[float],
        poses: list[FramePose],
        side_data: GaitSideEvents,
        side: str,
        fps: float,
    ) -> None:
        """
        Plot toe velocity with FC and LC event markers overlaid.

        Args:
            time:      Per-frame timestamps in seconds.
            poses:     Ordered list of FramePose (one per frame).
            side_data: Event data for this limb.
            side:      "left" or "right".
            fps:       Frames per second (used by compute_velocity).
        """
        toe_coords = self._get_toe_coords(poses, side)
        velocity = compute_joint_speed(toe_coords)

        t = np.array(time)
        FC_indices = side_data.FC_frames()
        LC_indices = side_data.LC_frames()

        plt.figure(figsize=(10, 4))
        plt.plot(t, velocity, label="Velocity")
        plt.scatter(
            t[FC_indices], velocity[FC_indices], label="FC (Foot Contact)", marker="o"
        )
        plt.scatter(
            t[LC_indices], velocity[LC_indices], label="LC (Foot Lift)", marker="x"
        )
        plt.title(f"{side.capitalize()} Velocity + Events")
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (px/s)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 2. Stance / swing timeline
    # --------------------------------------------------

    def plot_timeline_events(
        self,
        time: list[float],
        side_data: GaitSideEvents,
        side: str,
    ) -> None:
        """
        Plot the stance/swing timeline with FC/LC markers.

        Args:
            time:      Per-frame timestamps in seconds.
            side_data: Event data for this limb.
            side:      "left" or "right".
        """
        t = np.array(time)
        stance = np.array(side_data.stance_mask, dtype=int)
        FC_indices = side_data.FC_frames()
        LC_indices = side_data.LC_frames()

        plt.figure(figsize=(12, 2))
        plt.plot(t, stance, label="Stance")
        plt.scatter(t[FC_indices], np.ones(len(FC_indices)), label="FC", marker="o")
        plt.scatter(t[LC_indices], np.zeros(len(LC_indices)), label="LC", marker="x")
        plt.title(f"{side.capitalize()} Timeline + Events")
        plt.xlabel("Time (s)")
        plt.yticks([0, 1], ["Swing", "Stance"])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # Master runner
    # --------------------------------------------------

    def run(
        self,
        time: list[float],
        poses: list[FramePose],
        results: GaitEventsResult,
    ) -> None:
        """
        Run both visualizations for left and right feet.

        Args:
            time:    Per-frame timestamps in seconds.
            poses:   Ordered list of FramePose (one per frame).
            results: Bilateral gait event output.
        """
        for side in ("left", "right"):
            side_data: GaitSideEvents = getattr(results, side)
            self.plot_velocity_events(time, poses, side_data, side, results.fps)
            self.plot_timeline_events(time, side_data, side)
