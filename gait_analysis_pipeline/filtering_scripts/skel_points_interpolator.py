# confidence_interpolation_processor.py
from __future__ import annotations

import numpy as np

from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import FramePose
from pose_estimation.pose_estimation_dataclasses.processing_steps_dataclass import (
    ProcessingStep,
)


class ConfidenceInterpolationProcessor:
    """
    Drops keypoints with confidence < threshold and linearly interpolates
    the gaps across the temporal sequence.

    Pure transformation: list[FramePose] -> list[FramePose]
    """

    PROCESSING_STEP_NAME = "confidence_interpolation"

    def __init__(self, confidence_threshold: float, visibility_threshold: float):
        self._confidence_threshold = confidence_threshold
        self._visibility_threshold = visibility_threshold

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    @property
    def processing_step(self) -> ProcessingStep:
        return ProcessingStep(
            step_name=self.PROCESSING_STEP_NAME,
            params={
                "confidence_threshold": self._confidence_threshold,
                "visibility_threshold": self._visibility_threshold,
            },
        )

    def filter_low_scores(
        self, poses: list[FramePose]
    ) -> tuple[list[FramePose], ProcessingStep]:
        if not poses:
            return poses, self.processing_step

        num_frames = len(poses)
        num_joints = len(poses[0].joints)

        # Shape: (num_frames, num_joints, 2)
        coords = np.array(
            [[[j.keypoint.x, j.keypoint.y] for j in pose.joints] for pose in poses],
            dtype=float,
        )

        # Shape: (num_frames, num_joints) — True where confidence is low
        low_conf_mask = np.array(
            [[self._is_low_confidence(j) for j in pose.joints] for pose in poses],
            dtype=bool,
        )

        # Null-out low-confidence entries so we can interpolate over them
        coords[low_conf_mask] = np.nan  # broadcasts over the last axis (x, y)

        # Interpolate each joint independently along the time axis
        for joint_idx in range(num_joints):
            for xy in range(2):
                series = coords[:, joint_idx, xy]
                if np.isnan(series).all():
                    # No valid anchor points — fill with 0 (or leave NaN)
                    coords[:, joint_idx, xy] = 0.0
                    continue
                if not np.isnan(series).any():
                    continue  # nothing to fix

                valid = ~np.isnan(series)
                frame_indices = np.arange(num_frames)
                coords[:, joint_idx, xy] = np.interp(
                    frame_indices,
                    frame_indices[valid],
                    series[valid],
                )

        updated_poses = [
            pose.with_updated_keypoints(coords[i]) for i, pose in enumerate(poses)
        ]

        return updated_poses, self.processing_step

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _is_low_confidence(self, joint) -> bool:
        """
        Decide whether a joint counts as "low confidence" using whichever
        of visibility / confidence is actually available.

        - If both are present: low if either is below its threshold (OR).
        - If only one is present (the other is None): judge using that
          one alone, ignoring the missing field.
        - If neither is present: treated as low confidence, since there is
          no signal to confirm the keypoint is usable.
        """
        visibility = joint.visibility
        confidence = joint.keypoint.confidence

        visibility_available = visibility is not None
        confidence_available = confidence is not None

        if not visibility_available and not confidence_available:
            return True

        if visibility_available and not confidence_available:
            return visibility < self._visibility_threshold

        if confidence_available and not visibility_available:
            return confidence < self._confidence_threshold

        # both available
        return (
            visibility < self._visibility_threshold
            or confidence < self._confidence_threshold
        )
