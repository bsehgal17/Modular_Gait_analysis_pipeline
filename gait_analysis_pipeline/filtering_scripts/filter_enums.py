from enum import Enum


class FilterType(str, Enum):
    BUTTERWORTH = "butterworth"
    SAVGOL = "savgol"
