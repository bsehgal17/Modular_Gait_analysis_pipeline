from pose_estimation.pose_estimation_dataclasses.pose_estimation_dataclass import (
    PoseEstimationResult,
)
from pose_estimation.pose_estimation_dataclasses.frame_detection_dataclass import (
    FrameDetection,
)
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
    Joint,
    Keypoint,
)
from pose_estimation.pose_estimation_dataclasses.bbox_dataclass import BBox
from pose_estimation.enums.pose_enums import ObjectLabel, PoseModel
from pose_estimation.enums.joint_enum import StandardJoint


class PersonTracker:
    def __init__(
        self,
        overlap_threshold: float,
        distance_threshold: float,
    ):
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.next_person_id = 0

        # (person_id, FrameDetection, frame_idx)
        self.active = []

    # -------------------------------------------------
    # STEP 1: ASSIGN IDS (USES SINGLE DETECTION OBJECT)
    # -------------------------------------------------
    def assign_person_ids(
        self,
        pose_est_data: PoseEstimationResult,
        frame_idx: int,
        bboxes: list[BBox],
        scores: list[float],
    ) -> tuple[list[int], list[FrameDetection]]:
        """
        Returns:
            person_ids
            detections (shared objects used later in storage)
        """

        frame_ids: list[int] = []
        detections: list[FrameDetection] = []
        updated = []

        for bbox, score in zip(bboxes, scores):
            # CREATE ONCE
            current_det = FrameDetection(
                frame_idx=frame_idx,
                bbox=bbox,
                score=score,
                label=ObjectLabel.PERSON,
            )

            detections.append(current_det)

            best = None
            best_score = 0.0

            for i, (pid, prev_det, last_frame) in enumerate(self.active):
                if frame_idx - last_frame > 5:
                    continue

                iou = current_det.iou(prev_det)
                dist = current_det.distance(prev_det)

                if iou >= self.overlap_threshold and dist <= self.distance_threshold:
                    score_match = iou * (1.0 - dist / self.distance_threshold)

                    if score_match > best_score:
                        best = i
                        best_score = score_match

            if best is not None:
                pid, _, _ = self.active[best]
                frame_ids.append(pid)
                updated.append((pid, current_det, frame_idx))
                self.active.pop(best)
            else:
                pid = self.next_person_id
                self.next_person_id += 1

                frame_ids.append(pid)
                updated.append((pid, current_det, frame_idx))

                pose_est_data.get_or_create_person(pid)

        self.active.extend(updated)

        self.active = [
            (pid, det, fidx) for pid, det, fidx in self.active if frame_idx - fidx <= 5
        ]

        return frame_ids, detections

    # -------------------------------------------------
    # STEP 2: STORE RESULTS (REUSES SAME DETECTION OBJECTS)
    # -------------------------------------------------
    def store_frame(
        self,
        pose_est_data: PoseEstimationResult,
        frame_idx: int,
        detections: list[FrameDetection],
        person_ids: list[int],
        pose_results: list,
    ):
        for i, (det, pid) in enumerate(zip(detections, person_ids)):
            person = pose_est_data.get_or_create_person(pid)
            person.add_detection(det)

            if i < len(pose_results):
                records = pose_results[i]  # list of dicts: name, keypoint

                joints = []
                for rec in records:
                    x, y, score = float(rec["keypoint"][0]), float(
                        rec["keypoint"][1]), float(rec["keypoint"][2])
                    joints.append(
                        Joint(
                            name=rec["name"],
                            keypoint=Keypoint(
                                x=x,
                                y=y,
                                z=0.0,
                                confidence=score,
                            ),
                            visibility=1.0 if score > 0 else 0.0,
                        )
                    )

                person.add_pose(
                    FramePose(
                        frame_idx=frame_idx,
                        joints=tuple(joints),
                        bbox=det.bbox,
                        bbox_score=det.score,
                        model=PoseModel.RTMW,
                    )
                )
