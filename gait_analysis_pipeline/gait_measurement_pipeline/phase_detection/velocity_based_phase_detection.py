from __future__ import annotations

import numpy as np

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from pose_estimation.enums.joint_enum import JointEnum
from pose_estimation.utils.create_body_connections import JointConnections

from gait_measurement_pipeline.gait_dataclasses.phase_detection_dataclass import (
    PhaseDetectionConfig,
    VelocityPhaseResult,
)
from gait_measurement_pipeline.gait_dataclasses.foot_phase_dataclass import (
    FootExtremes,
    FootKeypoints,
    FootPhaseData,
    FootVelocity,
)
from gait_measurement_pipeline.utils.peak_detection import detect_peaks
from gait_measurement_pipeline.visualizations.phase_visualization import PhaseVisualizer
from gait_measurement_pipeline.visualizations.videovelocityviewer import (
    VideoVelocityViewer,
)


class VelocityPhaseDetector:
    """
    Detects gait phases (stance/swing) from pre-computed FootVelocity signals
    and handles all related visualization.

    Expects velocities that have already been extracted (and optionally smoothed)
    by FootVelocityExtractor. Has no knowledge of poses or keypoint extraction.
    """

    def __init__(
        self,
        config: PhaseDetectionConfig,
        joint_enum: type[JointEnum],
        connections: JointConnections,
        video_path: str | None = None,
    ) -> None:
        self._config = config
        self._joints = joint_enum
        self._connections = connections
        self._video_path = video_path

    # =====================================================
    # MAIN
    # =====================================================

    def run(
        self,
        left_vel: FootVelocity,
        right_vel: FootVelocity,
        left_kp: FootKeypoints,
        right_kp: FootKeypoints,
        raw_poses: list[FramePose],
    ) -> VelocityPhaseResult:
        """
        Args:
            left_vel, right_vel : FootVelocity
                Velocity signals — smooth these upstream before passing in.
            left_kp, right_kp : FootKeypoints
                Keypoints stored in the result (not used for computation here).
            raw_poses : list[FramePose]
                Raw (un-normalized) poses used for video overlay only.
        """
        if self._config.visualization.show_velocity_viewer and self._video_path:
            self._run_velocity_viewer(left_vel, right_vel, raw_poses)

        left_extremes = self._detect_peaks_and_valleys(left_vel)
        right_extremes = self._detect_peaks_and_valleys(right_vel)

        left_mask = self._adaptive_phase_detection(left_vel, left_extremes)
        right_mask = self._adaptive_phase_detection(right_vel, right_extremes)

        result = VelocityPhaseResult(
            fps=self._config.fps,
            left=FootPhaseData(foot_keypoints=left_kp, stance_mask=left_mask.tolist()),
            right=FootPhaseData(
                foot_keypoints=right_kp, stance_mask=right_mask.tolist()
            ),
        )

        if self._any_visualization_enabled():
            PhaseVisualizer(
                config=self._config,
                connections=self._connections,
            ).run(
                time=np.array(result.time),
                left_vel=left_vel.full_avg,
                right_vel=right_vel.full_avg,
                result=result,
                raw_poses=raw_poses,
                video_path=self._video_path,
                left_foot_extremes=left_extremes,
                right_foot_extremes=right_extremes,
            )

        return result

    # =====================================================
    # PEAK DETECTION
    # =====================================================

    def _detect_peaks_and_valleys(self, vel: FootVelocity) -> FootExtremes:
        full_avg = np.array(vel.full_avg)
        peaks, _ = detect_peaks(
            full_avg,
            fps=self._config.fps,
            prominence_ratio=self._config.prominence_ratio_peaks,
        )
        valleys, _ = detect_peaks(
            -full_avg,
            fps=self._config.fps,
            prominence_ratio=self._config.prominence_ratio_valleys,
            min_step_time=self._config.min_step_time,
        )
        return FootExtremes(peaks=peaks.tolist(), valleys=valleys.tolist())

    # =====================================================
    # ADAPTIVE PHASE DETECTION
    # =====================================================

    def _adaptive_phase_detection(
        self,
        vel: FootVelocity,
        foot_extremes: FootExtremes,
    ) -> np.ndarray:
        """
        Labels each frame as stance (True) or swing (False).

        Swing start uses toe velocity  — toe leaves the ground last at onset.
        Swing end   uses proximal_avg  — heel/ankle arrive first at contact.
        """
        toe = np.array(vel.toe)
        proximal = np.array(vel.proximal_avg)
        full_avg = np.array(vel.full_avg)

        peaks = np.array(foot_extremes.peaks)
        valleys = np.array(foot_extremes.valleys)

        n = len(full_avg)
        swing_mask = np.zeros(n, dtype=bool)

        for p_idx in peaks:
            pre = valleys[valleys < p_idx]
            post = valleys[valleys > p_idx]

            if len(pre) == 0 or len(post) == 0:
                continue

            v_pre = pre[-1]
            v_post = post[0]

            thresh_start = self._compute_threshold(toe, v_pre, v_post, p_idx, 70)
            thresh_end = self._compute_threshold(proximal, v_pre, v_post, p_idx, 60)

            if thresh_start is None or thresh_end is None:
                continue

            swing_start = p_idx
            while swing_start > v_pre and toe[swing_start] > thresh_start:
                swing_start -= 1

            swing_end = p_idx
            while swing_end < v_post and proximal[swing_end] > thresh_end:
                swing_end += 1

            min_swing_frames = max(2, int(0.10 * self._config.fps))
            if (swing_end - swing_start) < min_swing_frames:
                continue

            swing_mask[swing_start : swing_end + 1] = True

        return ~swing_mask

    def _compute_threshold(
        self,
        signal: np.ndarray,
        v_pre: int,
        v_post: int,
        p_idx: int,
        percentile: int,
    ) -> float | None:
        stance_region = signal[v_pre : v_post + 1]
        thresh = np.percentile(stance_region, percentile)
        if signal[p_idx] <= thresh:
            return None
        return thresh

    # =====================================================
    # HELPERS
    # =====================================================

    def _any_visualization_enabled(self) -> bool:
        v = self._config.visualization
        return any([
            v.plot,
            v.show_timeline,
            v.show_comparison,
            v.show_events,
            v.show_video,
            v.show_skeleton_video,
        ])

    def _run_velocity_viewer(
        self,
        left_vel: FootVelocity,
        right_vel: FootVelocity,
        raw_poses: list[FramePose],
    ) -> None:
        keypoints = np.array([
            [[j.keypoint.x, j.keypoint.y] for j in pose.joints] for pose in raw_poses
        ])
        VideoVelocityViewer(
            video_path=self._video_path,
            velocity={"left": left_vel.full_avg, "right": right_vel.full_avg},
            fps=self._config.fps,
            joint_enum=self._joints,
            show_skeleton=True,
            keypoints=keypoints,
            skeleton_edges=self._connections,
            save_svg=self._config.visualization.save_svg,
        ).run()
