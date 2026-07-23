import numpy as np
import cv2

from pose_estimation.pose_estimation_dataclasses.pose_estimation_dataclass import (
    PoseEstimationResult,
)
from pose_estimation.pose_estimation_dataclasses.frame_detection_dataclass import (
    FrameDetection,
)
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    BBox,
)
from pose_estimation.enums.pose_enums import ObjectLabel
from pose_estimation.processors.person_tracker import PersonTracker
from pose_estimation.enums.joint_enum import JointConverter


class FrameProcessor:
    def __init__(self, detector, estimator, visualizer, config):
        self.detector = detector
        self.estimator = estimator
        self.visualizer = visualizer
        self.config = config
        self.person_tracker = PersonTracker(
            overlap_threshold=0,
            distance_threshold=0,
        )
        self._joint_converter = JointConverter("coco_wholebody")

    def process_frame(
        self,
        frame,
        frame_idx: int,
        pose_est_data: PoseEstimationResult,
    ) -> tuple:
        """
        Returns:
            (processed_frame, raw_keypoints)

            processed_frame: Annotated BGR frame ready to write to video.
            raw_keypoints:   List of plain (133, 3) numpy arrays (x, y, score),
                             one per detected person, extracted from the mmpose
                             PoseDataSample objects RTMw produces. None when no
                             humans are detected. Saved as-is to
                             {video_name}_raw.json by the pipeline.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        all_bboxes, all_scores, all_labels = self.detector.detect_humans(rgb)

        human_indices = [
            i
            for i, (score, label) in enumerate(zip(all_scores, all_labels))
            if label == 0 and score > self.config.processing.detection_threshold
        ]

        if not human_indices:
            return frame, None

        human_bboxes_raw = [all_bboxes[i] for i in human_indices]
        human_scores = [all_scores[i] for i in human_indices]

        def to_bbox_xyxy(b):
            return BBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3])

        bboxes_struct = [to_bbox_xyxy(b) for b in human_bboxes_raw]

        # ---------------- POSE ESTIMATION ----------------
        _, pose_data_samples = self.estimator.estimate_pose(
            frame, human_bboxes_raw)
        # pose_data_samples: list of mmpose PoseDataSample objects, one per
        # detected person. Keypoint data lives at
        # pose_data_samples[i].pred_instances.keypoints (shape (1, 133, 2))
        # and .keypoint_scores (shape (1, 133)).

        # ---------------- EXTRACT PLAIN RAW KEYPOINTS (x, y, score) ----------
        raw_keypoints = [
            np.concatenate(
                [
                    pose.pred_instances.keypoints[0],
                    pose.pred_instances.keypoint_scores[0][:, None],
                ],
                axis=1,
            )
            for pose in pose_data_samples
        ]
        # raw_keypoints: list of (133, 3) arrays, exact RTMw output.

        # ---------------- CONVERT JOINTS (StandardJoint order) ------------
        named_pose_results = [
            self._joint_converter.to_direct_only(kp) for kp in raw_keypoints
        ]

        # ---------------- ASSIGN IDS ----------------
        person_ids = list(range(len(bboxes_struct)))

        detections = [
            FrameDetection(
                frame_idx=frame_idx,
                bbox=bbox,
                score=score,
                label=ObjectLabel.PERSON,
            )
            for bbox, score in zip(bboxes_struct, human_scores)
        ]

        # ---------------- STORE RESULTS (standard dataclass) ----------------
        self.person_tracker.store_frame(
            pose_est_data=pose_est_data,
            frame_idx=frame_idx,
            detections=detections,
            person_ids=person_ids,
            pose_results=named_pose_results,
        )

        # ---------------- VISUALIZATION ----------------
        # Visualizer needs the original PoseDataSample (it clones and mutates
        # .pred_instances directly), not the extracted plain array.
        visualized_frame = self.visualizer.visualize_pose(
            frame,
            pose_data_samples[0],
        )

        return visualized_frame, raw_keypoints
