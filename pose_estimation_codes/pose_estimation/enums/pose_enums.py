from enum import Enum, IntEnum


class ObjectLabel(IntEnum):
    PERSON = 0
    UNKNOWN = 1  # extensible for other detectors


class PoseModel(str, Enum):
    RTMW = "rtmw"
    RTMPose = "rtmpose"
    OpenPose = "openpose"
    MediaPipe = "mediapipe"
    AlphaPose = "alphapose"
    Unknown = "unknown"


class SkeletonAxis(IntEnum):
    FRAME = 0
    JOINT = 1
    COORD = 2


class CoordinateAxis(IntEnum):
    X = 0
    Y = 1
    Z = 2
