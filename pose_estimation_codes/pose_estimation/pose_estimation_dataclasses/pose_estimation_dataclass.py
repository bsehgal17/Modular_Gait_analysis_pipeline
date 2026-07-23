from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from pose_estimation.pose_estimation_dataclasses.frame_detection_dataclass import (
    FrameDetection,
)
from pose_estimation.pose_estimation_dataclasses.processing_steps_dataclass import (
    ProcessingStep,
)
from pose_estimation.pose_estimation_dataclasses.frame_pose_dataclass import (
    FramePose,
    Joint,
    Keypoint,
    BBox,
)
from pose_estimation.pose_estimation_dataclasses.tracked_person_dataclass import (
    TrackedPerson,
)


class PoseEstimationResult(BaseModel):
    video_name: str

    fps: float | None = None
    width: int | None = None
    height: int | None = None

    persons: list[TrackedPerson] = Field(default_factory=list)

    # =====================================================
    # PROCESSING HISTORY (GENERAL, NOT FILTER-SPECIFIC)
    # =====================================================
    processing_steps: list[ProcessingStep] = Field(default_factory=list)

    detection_config: Optional[dict] = None

    # -------------------------------------------------
    # FAST LOOKUP MAPS
    # -------------------------------------------------
    @cached_property
    def person_map(self) -> dict[int, TrackedPerson]:
        return {p.person_id: p for p in self.persons}

    @cached_property
    def detection_frame_map(self) -> dict[int, list[FrameDetection]]:
        out: dict[int, list[FrameDetection]] = {}

        for person in self.persons:
            for det in person.detections:
                out.setdefault(det.frame_idx, []).append(det)

        return out

    @cached_property
    def pose_frame_map(self) -> dict[int, list[FramePose]]:
        out: dict[int, list[FramePose]] = {}

        for person in self.persons:
            for pose in person.poses:
                out.setdefault(pose.frame_idx, []).append(pose)

        return out

    # -------------------------------------------------
    # PERSON HELPERS
    # -------------------------------------------------
    def get_or_create_person(self, pid: int) -> TrackedPerson:
        person = self.person_map.get(pid)

        if person is not None:
            return person

        person = TrackedPerson(person_id=pid)

        self.persons.append(person)

        # keep caches synchronized
        self.person_map[pid] = person

        return person

    # -------------------------------------------------
    # DETECTION HELPERS
    # -------------------------------------------------
    def all_detections(self) -> list[FrameDetection]:
        detections: list[FrameDetection] = []

        for person in self.persons:
            detections.extend(person.detections)

        return detections

    def detections_by_frame(
        self,
        frame_idx: int,
    ) -> list[FrameDetection]:
        return self.detection_frame_map.get(frame_idx, [])

    def poses_by_frame(
        self,
        frame_idx: int,
    ) -> list[FramePose]:
        return self.pose_frame_map.get(frame_idx, [])

    def _denormalize_pose(self, pose: FramePose, width: int, height: int) -> FramePose:
        new_joints = tuple(
            joint.model_copy(
                update={
                    "keypoint": joint.keypoint.model_copy(
                        update={
                            "x": joint.keypoint.x * width,
                            "y": joint.keypoint.y * height,
                            # z is typically depth — scale by width or leave as-is
                            "z": joint.keypoint.z * width,
                        }
                    )
                }
            )
            for joint in pose.joints
        )
        return pose.model_copy(update={"joints": new_joints})

    def with_pixels_keypoints(self) -> "PoseEstimationResult":
        """
        Converts normalized [0,1] keypoints to pixel coordinates
        for all persons/poses, using self.width and self.height.
        """
        if self.width is None or self.height is None:
            raise ValueError("width and height must be set before converting keypoints")

        new_persons = []
        for person in self.persons:
            new_poses = [
                self._denormalize_pose(pose, self.width, self.height)
                for pose in person.poses
            ]
            new_persons.append(person.model_copy(update={"poses": new_poses}))

        result = self.model_copy(update={"persons": new_persons})
        return result.add_processing_step(
            ProcessingStep(step_name="denormalize_keypoints")
        )

    # -------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------
    def save_json(self, path: str | Path):
        path = Path(path)

        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )

    # -------------------------------------------------
    # DESERIALIZATION
    # -------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "PoseEstimationResult":
        persons = []
        for p in data.get("persons", []):
            detections = [
                FrameDetection.model_construct(**d) for d in p.get("detections", [])
            ]
            poses = []
            for pose in p.get("poses", []):
                joints = tuple(
                    Joint.model_construct(
                        name=j["name"],
                        keypoint=Keypoint.model_construct(**j["keypoint"]),
                        visibility=j.get("visibility", 1.0),
                    )
                    for j in pose.get("joints", [])
                )
                poses.append(
                    FramePose.model_construct(
                        frame_idx=pose["frame_idx"],
                        joints=joints,
                        bbox=BBox.model_construct(**pose["bbox"]),
                        bbox_score=pose["bbox_score"],
                        model=pose["model"],
                    )
                )
            persons.append(
                TrackedPerson.model_construct(
                    person_id=p["person_id"],
                    detections=detections,
                    poses=poses,
                )
            )

        processing_steps = [
            ProcessingStep.model_construct(**s)
            for s in data.get("processing_steps", [])
        ]

        return cls.model_construct(
            video_name=data["video_name"],
            fps=data.get("fps"),
            width=data.get("width"),
            height=data.get("height"),
            persons=persons,
            processing_steps=processing_steps,
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
    ) -> "PoseEstimationResult":
        path = Path(path)

        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def add_processing_step(self, step: ProcessingStep) -> PoseEstimationResult:
        return self.model_copy(
            update={"processing_steps": [*self.processing_steps, step]}
        )

    def replace_poses(
        self,
        new_poses: list[FramePose],
        person_id: int,
    ) -> "PoseEstimationResult":
        """
        Replaces poses for a specific person only.
        """
        new_persons = []

        for person in self.persons:
            if person.person_id == person_id:
                new_persons.append(person.model_copy(update={"poses": new_poses}))
            else:
                new_persons.append(person)

        return self.model_copy(update={"persons": new_persons})
