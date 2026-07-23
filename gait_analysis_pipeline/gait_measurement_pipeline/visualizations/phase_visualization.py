from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cv2

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from gait_measurement_pipeline.gait_dataclasses.phase_detection_dataclass import (
    PhaseDetectionConfig,
    VelocityPhaseResult,
)
from gait_measurement_pipeline.phase_detection.foot_extractor import FootExtractor
from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import FootExtremes
import os


class PhaseVisualizer:
    """
    All gait phase visualizations.
    Driven entirely by VelocityPhaseResult and list[FramePose].
    """

    def __init__(
        self,
        config: PhaseDetectionConfig,
        connections: list[tuple[int, int]],
    ) -> None:
        self._config = config
        self._connections = connections

    # =====================================================
    # MASTER RUNNER
    # =====================================================

    def run(
        self,
        time: np.ndarray,
        left_vel: np.ndarray,
        right_vel: np.ndarray,
        result: VelocityPhaseResult,
        raw_poses: list[FramePose],
        left_foot_extremes: FootExtremes | None = None,  # ← new
        right_foot_extremes: FootExtremes | None = None,  # ← new
        video_path: str | None = None,
    ) -> None:
        v = self._config.visualization

        if v.show_timeline:
            self._plot_phase_timeline(time, result)

        if v.show_comparison:
            self._plot_velocity_comparison(time, left_vel, right_vel)

        if v.show_events:
            self._plot_events(time, left_vel, result.left.stance_mask, "Left Events")
            self._plot_events(time, right_vel, result.right.stance_mask, "Right Events")

        if v.plot:
            self._plot_velocity_with_phases(
                time,
                left_vel,
                result.left.stance_mask,
                "Left",
                foot_extremes=left_foot_extremes,
            )
            self._plot_velocity_with_phases(
                time,
                right_vel,
                result.right.stance_mask,
                "Right",
                foot_extremes=right_foot_extremes,
            )
        if v.plot and v.save_svg:
            self._plot_velocity_with_phases(
                time,
                left_vel,
                result.left.stance_mask,
                "Left",
                foot_extremes=left_foot_extremes,
                save_svg=True,
            )
            self._plot_velocity_with_phases(
                time,
                right_vel,
                result.right.stance_mask,
                "Right",
                foot_extremes=right_foot_extremes,
                save_svg=True,
            )

        if v.show_video and video_path:
            self._render_video(video_path, raw_poses, result)

    # =====================================================
    # PLOTS
    # =====================================================

    def _plot_phase_timeline(
        self,
        time: np.ndarray,
        result: VelocityPhaseResult,
    ) -> None:
        left_stance = np.array(result.left.stance_mask)
        right_stance = np.array(result.right.stance_mask)

        fig, ax = plt.subplots(figsize=(12, 2))
        ax.imshow(
            np.vstack([left_stance, right_stance]),
            aspect="auto",
            cmap="gray_r",
            extent=[time[0], time[-1], 0, 2],
        )
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["Left", "Right"])
        ax.set_title("Gait Phase Timeline (White = Stance)")
        ax.set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def _plot_velocity_comparison(
        self,
        time: np.ndarray,
        left_vel: np.ndarray,
        right_vel: np.ndarray,
    ) -> None:

        fig, ax = plt.subplots(figsize=(10, 4))

        # Paper-style font sizes
        title_size = 16
        label_size = 14
        tick_size = 12
        legend_size = 12

        ax.plot(time, left_vel, label="Left Velocity", linewidth=1.8)

        ax.plot(time, right_vel, label="Right Velocity", linewidth=1.8)

        ax.set_title("Left vs Right Foot Velocity", fontsize=title_size, pad=10)

        ax.set_xlabel("Time (s)", fontsize=label_size)

        ax.set_ylabel("Normalized Foot Height Velocity (1/s)", fontsize=label_size)

        ax.tick_params(axis="both", labelsize=tick_size)

        ax.legend(fontsize=legend_size)

        ax.grid(True, linewidth=0.5)

        fig.tight_layout()

        # Save as vector SVG
        fig.savefig("velocity_comparison.svg", format="svg", bbox_inches="tight")

        plt.show()
        plt.close(fig)

    def _plot_velocity_with_phases(
        self,
        time: np.ndarray,
        vel: np.ndarray,
        stance_mask: list[bool],
        foot: str,
        foot_extremes: FootExtremes | None = None,
        save_svg: bool = False,
        output_path: str | None = None,
    ) -> None:

        stance = np.array(stance_mask)
        swing = ~stance

        fig, ax = plt.subplots(figsize=(14, 4))

        # Paper-style font sizes
        ax_title_size = 16
        ax_label_size = 14
        ax_tick_size = 12
        ax_legend_size = 12

        ax.plot(
            time, vel, color="black", label=f"{foot} Foot Avg Velocity", linewidth=1.5
        )

        ax.fill_between(
            time,
            0,
            vel,
            where=stance,
            color="navy",
            alpha=0.3,
            label="Stance",
        )

        ax.fill_between(
            time,
            0,
            vel,
            where=swing,
            color="orange",
            alpha=0.3,
            label="Swing",
        )

        # Swing boundaries
        diff = np.diff(swing.astype(int))

        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0]

        if swing[0]:
            starts = np.insert(starts, 0, 0)

        if swing[-1]:
            ends = np.append(ends, len(swing) - 1)

        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.vlines(
                time[s],
                0,
                vel[s],
                color="green",
                linestyles="--",
                linewidth=1.5,
                label="Swing Start" if i == 0 else "",
            )

            ax.vlines(
                time[e],
                0,
                vel[e],
                color="purple",
                linestyles="--",
                linewidth=1.5,
                label="Swing End" if i == 0 else "",
            )

        ax.set_title(
            f"{foot} Foot Velocity with Detected Phases", fontsize=ax_title_size, pad=10
        )

        # Extrema markers
        if foot_extremes is not None:
            peaks = np.array(foot_extremes.peaks, dtype=int)

            valleys = np.array(foot_extremes.valleys, dtype=int)

            vel_arr = np.array(vel)

            if len(peaks):
                ax.scatter(
                    time[peaks],
                    vel_arr[peaks],
                    color="red",
                    zorder=5,
                    marker="^",
                    s=70,
                    label="Peaks (swing)",
                )

            if len(valleys):
                ax.scatter(
                    time[valleys],
                    vel_arr[valleys],
                    color="blue",
                    zorder=5,
                    marker="v",
                    s=70,
                    label="Valleys (stance)",
                )

        ax.set_xlabel("Time (s)", fontsize=ax_label_size)

        ax.set_ylabel("Velocity", fontsize=ax_label_size)

        ax.tick_params(axis="both", labelsize=ax_tick_size)

        ax.legend(fontsize=ax_legend_size, loc="best")

        ax.grid(True, linewidth=0.5)

        fig.tight_layout()

        # Save vector SVG for paper
        if save_svg:
            if output_path is None:
                output_path = f"{foot.lower()}_velocity_phases.svg"

            fig.savefig(output_path, format="svg", bbox_inches="tight")

            plt.close(fig)

            print(f"Saved velocity plot SVG: {output_path}")

        else:
            plt.show()

    def _plot_events(
        self,
        time: np.ndarray,
        vel: np.ndarray,
        stance_mask: list[bool],
        title: str,
    ) -> None:
        stance = np.array(stance_mask)
        heel_strike = np.where((~stance[:-1]) & (stance[1:]))[0]
        toe_off = np.where((stance[:-1]) & (~stance[1:]))[0]

        plt.figure(figsize=(10, 4))
        plt.plot(time, vel, label="Velocity")
        plt.scatter(
            time[heel_strike], vel[heel_strike], label="Heel Strike", marker="o"
        )
        plt.scatter(time[toe_off], vel[toe_off], label="Toe Off", marker="x")
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # =====================================================
    # VIDEO RENDERING
    # =====================================================

    def _render_video(
        self,
        video_path: str,
        raw_poses: list[FramePose],
        result: VelocityPhaseResult,
        output_path: str = "output_gait.mp4",
    ) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        left_stance = result.left.stance_mask
        right_stance = result.right.stance_mask

        left_toe = result.left.trajectory.toe
        right_toe = result.right.trajectory.toe

        frame_idx = 0
        while cap.isOpened() and frame_idx < len(raw_poses):
            ret, frame = cap.read()
            if not ret:
                break

            text = (
                f"Frame {frame_idx} | "
                f"L: {'STANCE' if left_stance[frame_idx] else 'SWING'} | "
                f"R: {'STANCE' if right_stance[frame_idx] else 'SWING'}"
            )
            cv2.putText(
                frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            lx, ly = left_toe[frame_idx].x, left_toe[frame_idx].y
            rx, ry = right_toe[frame_idx].x, right_toe[frame_idx].y
            cv2.circle(frame, (int(lx), int(ly)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(rx), int(ry)), 5, (255, 0, 0), -1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Saved gait video to: {output_path}")
