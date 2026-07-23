from enum import Enum


class NormalizerType(str, Enum):
    HEAD_BASED = "head_based"
    SHOULDER_BASED = "shoulder_based"
    HIP_BASED = "hip_based"
