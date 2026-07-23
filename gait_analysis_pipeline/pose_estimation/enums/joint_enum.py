from __future__ import annotations
import numpy as np
from enum import Enum
from strenum import StrEnum
from typing import Optional
from dataclasses import dataclass, field


class JointEnum(Enum):
    """
    Base class for all joint indexing schemas.

    Provides:
    - shared API
    - type safety
    - utility methods
    - validation helpers

    All pose/skeleton schemas should inherit from this class.
    """

    # =====================================================
    # COMMON UTILITIES
    # =====================================================

    @property
    def indices(self) -> list[int]:
        """
        Return joint indices as a standardized list.

        Supports:
        - single joint -> [idx]
        - grouped joints -> [idx1, idx2]
        """

        value = self.value

        if isinstance(value, int):
            return [value]

        return list(value)

    @property
    def is_group(self) -> bool:
        """
        Whether this joint represents multiple indices.
        """

        return not isinstance(self.value, int)

    @classmethod
    def has_joint(cls, name: str) -> bool:
        """
        Check whether schema contains joint.
        """

        return name.upper() in cls.__members__

    @classmethod
    def from_name(cls, name: str) -> "JointEnum":
        """
        Safe case-insensitive joint lookup.
        """

        try:
            return cls[name.upper()]
        except KeyError as exc:
            valid = ", ".join(cls.__members__.keys())

            raise ValueError(f"Unknown joint '{name}'. Valid joints: {valid}") from exc

    @classmethod
    def joint_names(cls) -> list[str]:
        """
        Return all available joint names.
        """

        return list(cls.__members__.keys())

    @classmethod
    def num_joints(cls) -> int:
        """
        Approximate number of unique joints.

        Useful for validation/debugging.
        """

        unique_indices: set[int] = set()

        for joint in cls:
            unique_indices.update(joint.indices)

        return len(unique_indices)


class StandardJoint(StrEnum):
    NOSE = "NOSE"
    LEFT_EYE = "LEFT_EYE"
    RIGHT_EYE = "RIGHT_EYE"
    LEFT_EAR = "LEFT_EAR"
    RIGHT_EAR = "RIGHT_EAR"
    NECK = "NECK"
    HEAD = "HEAD"
    LEFT_SHOULDER = "LEFT_SHOULDER"
    RIGHT_SHOULDER = "RIGHT_SHOULDER"
    LEFT_ELBOW = "LEFT_ELBOW"
    RIGHT_ELBOW = "RIGHT_ELBOW"
    LEFT_WRIST = "LEFT_WRIST"
    RIGHT_WRIST = "RIGHT_WRIST"
    LEFT_HIP = "LEFT_HIP"
    RIGHT_HIP = "RIGHT_HIP"
    LEFT_KNEE = "LEFT_KNEE"
    RIGHT_KNEE = "RIGHT_KNEE"
    LEFT_ANKLE = "LEFT_ANKLE"
    RIGHT_ANKLE = "RIGHT_ANKLE"
    LEFT_HEEL = "LEFT_HEEL"
    RIGHT_HEEL = "RIGHT_HEEL"
    LEFT_BIG_TOE = "LEFT_BIG_TOE"
    RIGHT_BIG_TOE = "RIGHT_BIG_TOE"
    PELVIS = "PELVIS"
    SPINE = "SPINE"
    MID_HIP = "MID_HIP"


class GTJointsHumanSC3D(JointEnum):
    """
    HumanSC3D dataset uses 25 joints format.
    Based on the limb connections in convert_3D_to_2D.py:

    Limbs: [10,9], [9,8], [8,11], [8,14], [11,12], [14,15], [12,13], [15,16],
           [8,7], [7,0], [0,1], [0,4], [1,2], [4,5], [2,3], [5,6],
           [13,21], [13,22], [16,23], [16,24], [3,17], [3,18], [6,19], [6,20]
    """

    HEAD = 7
    NECK = 8
    PELVIS = 0

    # Arms/Upper body
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 14
    LEFT_ELBOW = 12
    RIGHT_ELBOW = 15
    LEFT_WRIST = 13
    RIGHT_WRIST = 16

    # Legs/Lower body
    LEFT_HIP = 1
    RIGHT_HIP = 4
    LEFT_KNEE = 2
    RIGHT_KNEE = 5
    LEFT_ANKLE = 3
    RIGHT_ANKLE = 6

    # Additional joints
    SPINE = 9
    SPINE_TOP = 10

    # Hand extremities
    LEFT_HAND_TIP = 21
    LEFT_HAND_THUMB = 22
    RIGHT_HAND_TIP = 23
    RIGHT_HAND_THUMB = 24

    # Foot extremities
    LEFT_FOOT_TIP = 17
    LEFT_FOOT_THUMB = 18
    RIGHT_FOOT_TIP = 19
    RIGHT_FOOT_THUMB = 20


class GTJointsHumanEVa(JointEnum):
    PELVIS = 0
    SHOULDER_CENTER = (1, 2)
    LEFT_HIP = 11
    RIGHT_HIP = 15
    LEFT_KNEE = (12, 13)
    RIGHT_KNEE = (16, 17)
    LEFT_ANKLE = 14
    RIGHT_ANKLE = 18
    HEAD = 19
    LEFT_SHOULDER = 3
    RIGHT_SHOULDER = 7
    LEFT_ELBOW = (5, 4)
    RIGHT_ELBOW = (8, 9)
    LEFT_WRIST = 6
    RIGHT_WRIST = 10
    MID_SHOULDER = (3, 7)  # Midpoint of both shoulders


class GTJointsMoVi(JointEnum):
    HEAD = 15
    NECK = 14
    LEFT_SHOULDER = 16
    RIGHT_SHOULDER = 17
    LEFT_ELBOW = 18
    RIGHT_ELBOW = 19
    LEFT_WRIST = 20
    RIGHT_WRIST = 21
    LEFT_HIP = 1
    RIGHT_HIP = 2
    LEFT_KNEE = 4
    RIGHT_KNEE = 5
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    # LEFT_HEEL = 44
    # RIGHT_HEEL = 45
    LEFT_TOE = 10  # 2nd metatarsal
    RIGHT_TOE = 11


class PredJointsCOCOWholebody(JointEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    # Foot Keypoints (17-22)
    LEFT_BIG_TOE = 17
    LEFT_SMALL_TOE = 22
    LEFT_HEEL = 18
    RIGHT_BIG_TOE = 19
    RIGHT_SMALL_TOE = 21
    RIGHT_HEEL = 20

    # Note: Indices 23-90 are Face landmarks
    # Indices 91-111 are Left Hand landmarks
    # Indices 112-132 are Right Hand landmarks


class PredJointsCOCO17(JointEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Mediapipe(JointEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_BIG_TOE = 31
    RIGHT_BIG_TOE = 32


class OpenPoseBODY25(JointEnum):
    NOSE = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    MID_HIP = 8
    RIGHT_HIP = 9
    RIGHT_KNEE = 10
    RIGHT_ANKLE = 11
    LEFT_HIP = 12
    LEFT_KNEE = 13
    LEFT_ANKLE = 14
    RIGHT_EYE = 15
    LEFT_EYE = 16
    RIGHT_EAR = 17
    LEFT_EAR = 18
    LEFT_TOE = 19
    LEFT_SMALL_TOE = 20
    LEFT_HEEL = 21
    RIGHT_TOE = 22
    RIGHT_SMALL_TOE = 23
    RIGHT_HEEL = 24


class PredJointsHALPE26(JointEnum):
    Nose = 0
    Left_Eye = 1
    Right_Eye = 2
    Left_Ear = 3
    Right_Ear = 4

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16

    HEAD = 17
    NECK = 18
    HIP = 19

    LEFT_BIG_TOE = 20
    RIGHT_BIG_TOE = 21
    LEFT_SMALL_TOE = 22
    RIGHT_SMALL_TOE = 23
    LEFT_HEEL = 24
    RIGHT_HEEL = 25


"""
Joint schema mapping layer.

Bridges StandardJoint <-> any prediction/GT schema.
RTMPose/RTMw outputs COCO Wholebody (23 body+foot joints, plus face/hand).
This module handles the mismatch cleanly.
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAPPING TABLES
# Each entry: StandardJoint -> index in the source schema.
# None = joint does not exist in that schema (must be derived or skipped).
# ─────────────────────────────────────────────────────────────────────────────

# COCO Wholebody (RTMPose / RTMw output)
# Joints 0-16  : body  (same as COCO17)
# Joints 17-22 : foot
# Joints 23-90 : face  (not mapped to StandardJoint)
# Joints 91-132: hands (not mapped to StandardJoint)
COCO_WHOLEBODY_TO_STANDARD: dict[StandardJoint, Optional[int]] = {
    StandardJoint.NOSE: PredJointsCOCOWholebody.NOSE.value,
    StandardJoint.LEFT_EYE: PredJointsCOCOWholebody.LEFT_EYE.value,
    StandardJoint.RIGHT_EYE: PredJointsCOCOWholebody.RIGHT_EYE.value,
    StandardJoint.LEFT_EAR: PredJointsCOCOWholebody.LEFT_EAR.value,
    StandardJoint.RIGHT_EAR: PredJointsCOCOWholebody.RIGHT_EAR.value,
    StandardJoint.LEFT_SHOULDER: PredJointsCOCOWholebody.LEFT_SHOULDER.value,
    StandardJoint.RIGHT_SHOULDER: PredJointsCOCOWholebody.RIGHT_SHOULDER.value,
    StandardJoint.LEFT_ELBOW: PredJointsCOCOWholebody.LEFT_ELBOW.value,
    StandardJoint.RIGHT_ELBOW: PredJointsCOCOWholebody.RIGHT_ELBOW.value,
    StandardJoint.LEFT_WRIST: PredJointsCOCOWholebody.LEFT_WRIST.value,
    StandardJoint.RIGHT_WRIST: PredJointsCOCOWholebody.RIGHT_WRIST.value,
    StandardJoint.LEFT_HIP: PredJointsCOCOWholebody.LEFT_HIP.value,
    StandardJoint.RIGHT_HIP: PredJointsCOCOWholebody.RIGHT_HIP.value,
    StandardJoint.LEFT_KNEE: PredJointsCOCOWholebody.LEFT_KNEE.value,
    StandardJoint.RIGHT_KNEE: PredJointsCOCOWholebody.RIGHT_KNEE.value,
    StandardJoint.LEFT_ANKLE: PredJointsCOCOWholebody.LEFT_ANKLE.value,
    StandardJoint.RIGHT_ANKLE: PredJointsCOCOWholebody.RIGHT_ANKLE.value,
    StandardJoint.LEFT_BIG_TOE: PredJointsCOCOWholebody.LEFT_BIG_TOE.value,
    StandardJoint.RIGHT_BIG_TOE: PredJointsCOCOWholebody.RIGHT_BIG_TOE.value,
    StandardJoint.LEFT_HEEL: PredJointsCOCOWholebody.LEFT_HEEL.value,
    StandardJoint.RIGHT_HEEL: PredJointsCOCOWholebody.RIGHT_HEEL.value,
    # ── Derived joints: not directly in COCO Wholebody ───────────────────────
    # These must be computed (see DerivedJointStrategy below).
    StandardJoint.NECK: None,  # midpoint of shoulders (5, 6)
    StandardJoint.HEAD: None,  # midpoint of ears (3, 4)
    StandardJoint.MID_HIP: None,  # midpoint of hips (11, 12)
    StandardJoint.PELVIS: None,  # same as MID_HIP in 2-D context
    StandardJoint.SPINE: None,  # midpoint of NECK and MID_HIP (derived)
}


COCO17_TO_STANDARD: dict[StandardJoint, Optional[int]] = {
    StandardJoint.NOSE: PredJointsCOCO17.NOSE.value,
    StandardJoint.LEFT_EYE: PredJointsCOCO17.LEFT_EYE.value,
    StandardJoint.RIGHT_EYE: PredJointsCOCO17.RIGHT_EYE.value,
    StandardJoint.LEFT_EAR: PredJointsCOCO17.LEFT_EAR.value,
    StandardJoint.RIGHT_EAR: PredJointsCOCO17.RIGHT_EAR.value,
    StandardJoint.LEFT_SHOULDER: PredJointsCOCO17.LEFT_SHOULDER.value,
    StandardJoint.RIGHT_SHOULDER: PredJointsCOCO17.RIGHT_SHOULDER.value,
    StandardJoint.LEFT_ELBOW: PredJointsCOCO17.LEFT_ELBOW.value,
    StandardJoint.RIGHT_ELBOW: PredJointsCOCO17.RIGHT_ELBOW.value,
    StandardJoint.LEFT_WRIST: PredJointsCOCO17.LEFT_WRIST.value,
    StandardJoint.RIGHT_WRIST: PredJointsCOCO17.RIGHT_WRIST.value,
    StandardJoint.LEFT_HIP: PredJointsCOCO17.LEFT_HIP.value,
    StandardJoint.RIGHT_HIP: PredJointsCOCO17.RIGHT_HIP.value,
    StandardJoint.LEFT_KNEE: PredJointsCOCO17.LEFT_KNEE.value,
    StandardJoint.RIGHT_KNEE: PredJointsCOCO17.RIGHT_KNEE.value,
    StandardJoint.LEFT_ANKLE: PredJointsCOCO17.LEFT_ANKLE.value,
    StandardJoint.RIGHT_ANKLE: PredJointsCOCO17.RIGHT_ANKLE.value,
    # Not present in COCO17
    StandardJoint.NECK: None,
    StandardJoint.HEAD: None,
    StandardJoint.MID_HIP: None,
    StandardJoint.PELVIS: None,
    StandardJoint.SPINE: None,
    StandardJoint.LEFT_BIG_TOE: None,
    StandardJoint.RIGHT_BIG_TOE: None,
    StandardJoint.LEFT_HEEL: None,
    StandardJoint.RIGHT_HEEL: None,
}


MEDIAPIPE_TO_STANDARD: dict[StandardJoint, Optional[int]] = {
    StandardJoint.NOSE: Mediapipe.NOSE.value,
    StandardJoint.LEFT_EYE: Mediapipe.LEFT_EYE.value,
    StandardJoint.RIGHT_EYE: Mediapipe.RIGHT_EYE.value,
    StandardJoint.LEFT_EAR: Mediapipe.LEFT_EAR.value,
    StandardJoint.RIGHT_EAR: Mediapipe.RIGHT_EAR.value,
    StandardJoint.LEFT_SHOULDER: Mediapipe.LEFT_SHOULDER.value,
    StandardJoint.RIGHT_SHOULDER: Mediapipe.RIGHT_SHOULDER.value,
    StandardJoint.LEFT_ELBOW: Mediapipe.LEFT_ELBOW.value,
    StandardJoint.RIGHT_ELBOW: Mediapipe.RIGHT_ELBOW.value,
    StandardJoint.LEFT_WRIST: Mediapipe.LEFT_WRIST.value,
    StandardJoint.RIGHT_WRIST: Mediapipe.RIGHT_WRIST.value,
    StandardJoint.LEFT_HIP: Mediapipe.LEFT_HIP.value,
    StandardJoint.RIGHT_HIP: Mediapipe.RIGHT_HIP.value,
    StandardJoint.LEFT_KNEE: Mediapipe.LEFT_KNEE.value,
    StandardJoint.RIGHT_KNEE: Mediapipe.RIGHT_KNEE.value,
    StandardJoint.LEFT_ANKLE: Mediapipe.LEFT_ANKLE.value,
    StandardJoint.RIGHT_ANKLE: Mediapipe.RIGHT_ANKLE.value,
    StandardJoint.LEFT_HEEL: Mediapipe.LEFT_HEEL.value,
    StandardJoint.RIGHT_HEEL: Mediapipe.RIGHT_HEEL.value,
    StandardJoint.LEFT_BIG_TOE: Mediapipe.LEFT_BIG_TOE.value,
    StandardJoint.RIGHT_BIG_TOE: Mediapipe.RIGHT_BIG_TOE.value,
    StandardJoint.NECK: None,
    StandardJoint.HEAD: None,
    StandardJoint.MID_HIP: None,
    StandardJoint.PELVIS: None,
    StandardJoint.SPINE: None,
}


OPENPOSE_BODY25_TO_STANDARD: dict[StandardJoint, Optional[int]] = {
    StandardJoint.NOSE: OpenPoseBODY25.NOSE.value,
    StandardJoint.LEFT_EYE: OpenPoseBODY25.LEFT_EYE.value,
    StandardJoint.RIGHT_EYE: OpenPoseBODY25.RIGHT_EYE.value,
    StandardJoint.LEFT_EAR: OpenPoseBODY25.LEFT_EAR.value,
    StandardJoint.RIGHT_EAR: OpenPoseBODY25.RIGHT_EAR.value,
    StandardJoint.NECK: OpenPoseBODY25.NECK.value,
    StandardJoint.MID_HIP: OpenPoseBODY25.MID_HIP.value,
    StandardJoint.LEFT_SHOULDER: OpenPoseBODY25.LEFT_SHOULDER.value,
    StandardJoint.RIGHT_SHOULDER: OpenPoseBODY25.RIGHT_SHOULDER.value,
    StandardJoint.LEFT_ELBOW: OpenPoseBODY25.LEFT_ELBOW.value,
    StandardJoint.RIGHT_ELBOW: OpenPoseBODY25.RIGHT_ELBOW.value,
    StandardJoint.LEFT_WRIST: OpenPoseBODY25.LEFT_WRIST.value,
    StandardJoint.RIGHT_WRIST: OpenPoseBODY25.RIGHT_WRIST.value,
    StandardJoint.LEFT_HIP: OpenPoseBODY25.LEFT_HIP.value,
    StandardJoint.RIGHT_HIP: OpenPoseBODY25.RIGHT_HIP.value,
    StandardJoint.LEFT_KNEE: OpenPoseBODY25.LEFT_KNEE.value,
    StandardJoint.RIGHT_KNEE: OpenPoseBODY25.RIGHT_KNEE.value,
    StandardJoint.LEFT_ANKLE: OpenPoseBODY25.LEFT_ANKLE.value,
    StandardJoint.RIGHT_ANKLE: OpenPoseBODY25.RIGHT_ANKLE.value,
    StandardJoint.LEFT_HEEL: OpenPoseBODY25.LEFT_HEEL.value,
    StandardJoint.RIGHT_HEEL: OpenPoseBODY25.RIGHT_HEEL.value,
    StandardJoint.HEAD: None,
    StandardJoint.LEFT_BIG_TOE: None,
    StandardJoint.RIGHT_BIG_TOE: None,
    StandardJoint.PELVIS: None,
    StandardJoint.SPINE: None,
}


HALPE26_TO_STANDARD: dict[StandardJoint, Optional[int]] = {
    StandardJoint.NOSE: PredJointsHALPE26.Nose.value,
    StandardJoint.LEFT_EYE: PredJointsHALPE26.Left_Eye.value,
    StandardJoint.RIGHT_EYE: PredJointsHALPE26.Right_Eye.value,
    StandardJoint.LEFT_EAR: PredJointsHALPE26.Left_Ear.value,
    StandardJoint.RIGHT_EAR: PredJointsHALPE26.Right_Ear.value,
    StandardJoint.HEAD: PredJointsHALPE26.HEAD.value,
    StandardJoint.NECK: PredJointsHALPE26.NECK.value,
    StandardJoint.LEFT_SHOULDER: PredJointsHALPE26.LEFT_SHOULDER.value,
    StandardJoint.RIGHT_SHOULDER: PredJointsHALPE26.RIGHT_SHOULDER.value,
    StandardJoint.LEFT_ELBOW: PredJointsHALPE26.LEFT_ELBOW.value,
    StandardJoint.RIGHT_ELBOW: PredJointsHALPE26.RIGHT_ELBOW.value,
    StandardJoint.LEFT_WRIST: PredJointsHALPE26.LEFT_WRIST.value,
    StandardJoint.RIGHT_WRIST: PredJointsHALPE26.RIGHT_WRIST.value,
    StandardJoint.LEFT_HIP: PredJointsHALPE26.LEFT_HIP.value,
    StandardJoint.RIGHT_HIP: PredJointsHALPE26.RIGHT_HIP.value,
    StandardJoint.LEFT_KNEE: PredJointsHALPE26.LEFT_KNEE.value,
    StandardJoint.RIGHT_KNEE: PredJointsHALPE26.RIGHT_KNEE.value,
    StandardJoint.LEFT_ANKLE: PredJointsHALPE26.LEFT_ANKLE.value,
    StandardJoint.RIGHT_ANKLE: PredJointsHALPE26.RIGHT_ANKLE.value,
    StandardJoint.LEFT_BIG_TOE: PredJointsHALPE26.LEFT_BIG_TOE.value,
    StandardJoint.RIGHT_BIG_TOE: PredJointsHALPE26.RIGHT_BIG_TOE.value,
    StandardJoint.LEFT_HEEL: PredJointsHALPE26.LEFT_HEEL.value,
    StandardJoint.RIGHT_HEEL: PredJointsHALPE26.RIGHT_HEEL.value,
    StandardJoint.MID_HIP: None,
    StandardJoint.PELVIS: None,
    StandardJoint.SPINE: None,
}


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED JOINT STRATEGIES
# Tells the converter how to compute joints that don't exist in a schema.
# Each strategy is (source_indices, weights) -> weighted average.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DerivedJoint:
    """Compute a virtual joint as a weighted average of real source indices."""

    source_indices: list[int]
    weights: list[float] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.source_indices)
        if not self.weights:
            self.weights = [1.0 / n] * n
        assert len(self.weights) == n, "weights must match source_indices length"
        assert abs(sum(self.weights) - 1.0) < 1e-6, "weights must sum to 1"


# Derived joints for COCO Wholebody / COCO17
# (same shoulder/hip indices in both schemas)
COCO_DERIVED: dict[StandardJoint, DerivedJoint] = {
    # NECK  = midpoint of left/right shoulder
    StandardJoint.NECK: DerivedJoint(
        source_indices=[
            PredJointsCOCOWholebody.LEFT_SHOULDER.value,  # 5
            PredJointsCOCOWholebody.RIGHT_SHOULDER.value,  # 6
        ]
    ),
    # HEAD  = midpoint of ears
    StandardJoint.HEAD: DerivedJoint(
        source_indices=[
            PredJointsCOCOWholebody.LEFT_EAR.value,  # 3
            PredJointsCOCOWholebody.RIGHT_EAR.value,  # 4
        ]
    ),
    # MID_HIP / PELVIS = midpoint of hips
    StandardJoint.MID_HIP: DerivedJoint(
        source_indices=[
            PredJointsCOCOWholebody.LEFT_HIP.value,  # 11
            PredJointsCOCOWholebody.RIGHT_HIP.value,  # 12
        ]
    ),
    StandardJoint.PELVIS: DerivedJoint(
        source_indices=[
            PredJointsCOCOWholebody.LEFT_HIP.value,
            PredJointsCOCOWholebody.RIGHT_HIP.value,
        ]
    ),
    # SPINE = midpoint of derived NECK (avg shoulders) and MID_HIP (avg hips)
    # Expressed as average of all four joints with equal weight
    StandardJoint.SPINE: DerivedJoint(
        source_indices=[
            PredJointsCOCOWholebody.LEFT_SHOULDER.value,  # 5
            PredJointsCOCOWholebody.RIGHT_SHOULDER.value,  # 6
            PredJointsCOCOWholebody.LEFT_HIP.value,  # 11
            PredJointsCOCOWholebody.RIGHT_HIP.value,  # 12
        ]
    ),
}

# Mediapipe shares the same shoulder/hip indices as COCO for derived joints
MEDIAPIPE_DERIVED: dict[StandardJoint, DerivedJoint] = {
    StandardJoint.NECK: DerivedJoint(
        source_indices=[Mediapipe.LEFT_SHOULDER.value, Mediapipe.RIGHT_SHOULDER.value]
    ),
    StandardJoint.HEAD: DerivedJoint(
        source_indices=[Mediapipe.LEFT_EAR.value, Mediapipe.RIGHT_EAR.value]
    ),
    StandardJoint.MID_HIP: DerivedJoint(
        source_indices=[Mediapipe.LEFT_HIP.value, Mediapipe.RIGHT_HIP.value]
    ),
    StandardJoint.PELVIS: DerivedJoint(
        source_indices=[Mediapipe.LEFT_HIP.value, Mediapipe.RIGHT_HIP.value]
    ),
    StandardJoint.SPINE: DerivedJoint(
        source_indices=[
            Mediapipe.LEFT_SHOULDER.value,
            Mediapipe.RIGHT_SHOULDER.value,
            Mediapipe.LEFT_HIP.value,
            Mediapipe.RIGHT_HIP.value,
        ]
    ),
}

# OpenPose BODY25 has NECK and MID_HIP natively; only SPINE/PELVIS/HEAD need derivation
OPENPOSE_DERIVED: dict[StandardJoint, DerivedJoint] = {
    StandardJoint.HEAD: DerivedJoint(
        source_indices=[OpenPoseBODY25.LEFT_EAR.value, OpenPoseBODY25.RIGHT_EAR.value]
    ),
    StandardJoint.PELVIS: DerivedJoint(
        source_indices=[OpenPoseBODY25.LEFT_HIP.value, OpenPoseBODY25.RIGHT_HIP.value]
    ),
    StandardJoint.SPINE: DerivedJoint(
        source_indices=[OpenPoseBODY25.NECK.value, OpenPoseBODY25.MID_HIP.value]
    ),
    StandardJoint.LEFT_BIG_TOE: DerivedJoint(
        source_indices=[OpenPoseBODY25.LEFT_ANKLE.value]
    ),
    StandardJoint.RIGHT_BIG_TOE: DerivedJoint(
        source_indices=[OpenPoseBODY25.RIGHT_ANKLE.value]
    ),
}

# HALPE26 has HEAD, NECK natively; only SPINE/PELVIS/MID_HIP need derivation
HALPE26_DERIVED: dict[StandardJoint, DerivedJoint] = {
    StandardJoint.MID_HIP: DerivedJoint(
        source_indices=[
            PredJointsHALPE26.LEFT_HIP.value,
            PredJointsHALPE26.RIGHT_HIP.value,
        ]
    ),
    StandardJoint.PELVIS: DerivedJoint(
        source_indices=[
            PredJointsHALPE26.LEFT_HIP.value,
            PredJointsHALPE26.RIGHT_HIP.value,
        ]
    ),
    StandardJoint.SPINE: DerivedJoint(
        source_indices=[
            PredJointsHALPE26.LEFT_SHOULDER.value,
            PredJointsHALPE26.RIGHT_SHOULDER.value,
            PredJointsHALPE26.LEFT_HIP.value,
            PredJointsHALPE26.RIGHT_HIP.value,
        ]
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA REGISTRY
# One place to look up mappings + derived strategies by schema name.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SchemaConfig:
    """Everything needed to convert one schema -> StandardJoint."""

    direct: dict[StandardJoint, Optional[int]]
    derived: dict[StandardJoint, DerivedJoint] = field(default_factory=dict)
    # total number of joints in raw model output (for validation)
    num_raw_joints: Optional[int] = None


SCHEMA_REGISTRY: dict[str, SchemaConfig] = {
    "coco_wholebody": SchemaConfig(
        direct=COCO_WHOLEBODY_TO_STANDARD,
        derived=COCO_DERIVED,
        num_raw_joints=133,
    ),
    "coco17": SchemaConfig(
        direct=COCO17_TO_STANDARD,
        derived=COCO_DERIVED,  # shoulder/hip indices are identical
        num_raw_joints=17,
    ),
    "mediapipe": SchemaConfig(
        direct=MEDIAPIPE_TO_STANDARD,
        derived=MEDIAPIPE_DERIVED,
        num_raw_joints=33,
    ),
    "openpose_body25": SchemaConfig(
        direct=OPENPOSE_BODY25_TO_STANDARD,
        derived=OPENPOSE_DERIVED,
        num_raw_joints=25,
    ),
    "halpe26": SchemaConfig(
        direct=HALPE26_TO_STANDARD,
        derived=HALPE26_DERIVED,
        num_raw_joints=26,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# CONVERTER
# ─────────────────────────────────────────────────────────────────────────────


class JointConverter:
    """
    Convert raw keypoint arrays from any registered schema to StandardJoint order.

    Usage:
        converter = JointConverter("coco_wholebody")
        standard_kps = converter.to_standard(raw_kps)   # shape: (26, D)
    """

    def __init__(self, source_schema: str):
        if source_schema not in SCHEMA_REGISTRY:
            raise ValueError(
                f"Unknown schema '{source_schema}'. "
                f"Available: {list(SCHEMA_REGISTRY.keys())}"
            )
        self.schema_name = source_schema
        self.config = SCHEMA_REGISTRY[source_schema]
        self._standard_joints = list(StandardJoint)

    def to_standard(
        self,
        keypoints: np.ndarray,
        missing_value: float = 0.0,
    ) -> np.ndarray:
        """
        Map raw keypoints to StandardJoint order.

        Args:
            keypoints:     Raw array of shape (N, D) where N = num raw joints,
                           D = 2 (xy) or 3 (xyz) or 4 (xy + conf + vis).
            missing_value: Fill value for joints absent from source schema
                           and not derivable.

        Returns:
            Array of shape (len(StandardJoint), D).
        """
        if self.config.num_raw_joints is not None:
            if keypoints.shape[0] != self.config.num_raw_joints:
                raise ValueError(
                    f"Schema '{self.schema_name}' expects {self.config.num_raw_joints} "
                    f"joints, got {keypoints.shape[0]}."
                )

        n_standard = len(self._standard_joints)
        D = keypoints.shape[1] if keypoints.ndim == 2 else 1
        out = np.full((n_standard, D), missing_value, dtype=keypoints.dtype)

        for i, joint in enumerate(self._standard_joints):
            direct_idx = self.config.direct.get(joint)

            if direct_idx is not None:
                # Direct copy
                out[i] = keypoints[direct_idx]

            elif joint in self.config.derived:
                # Weighted average of source joints
                strategy = self.config.derived[joint]
                src = keypoints[strategy.source_indices]  # (k, D)
                w = np.array(strategy.weights, dtype=float)  # (k,)
                out[i] = (src * w[:, None]).sum(axis=0)

            # else: leave as missing_value

        return out

    def available_joints(self) -> list[StandardJoint]:
        """
        Return which StandardJoints are available (direct or derived) in this schema.
        """
        available = []
        for joint in self._standard_joints:
            direct_idx = self.config.direct.get(joint)
            if direct_idx is not None or joint in self.config.derived:
                available.append(joint)
        return available

    def missing_joints(self) -> list[StandardJoint]:
        """
        Return which StandardJoints cannot be produced from this schema.
        """
        available = set(self.available_joints())
        return [j for j in self._standard_joints if j not in available]

    def to_direct_only(self, keypoints: np.ndarray) -> list[dict]:
        """
        Return only StandardJoints with a DIRECT raw-index mapping (no derived
        joints), ordered by ascending raw index -- i.e. the same relative
        order as the raw schema itself.

        Returns:
            List of {"name": StandardJoint, "keypoint": np.ndarray} dicts.
            Length = number of direct mappings (21 for coco_wholebody).
        """
        if self.config.num_raw_joints is not None:
            if keypoints.shape[0] != self.config.num_raw_joints:
                raise ValueError(
                    f"Schema '{self.schema_name}' expects {self.config.num_raw_joints} "
                    f"joints, got {keypoints.shape[0]}."
                )

        direct_pairs = [
            (raw_idx, joint)
            for joint, raw_idx in self.config.direct.items()
            if raw_idx is not None
        ]
        direct_pairs.sort(key=lambda pair: pair[0])  # ascending raw index

        return [
            {"name": joint, "keypoint": keypoints[raw_idx]}
            for raw_idx, joint in direct_pairs
        ]
