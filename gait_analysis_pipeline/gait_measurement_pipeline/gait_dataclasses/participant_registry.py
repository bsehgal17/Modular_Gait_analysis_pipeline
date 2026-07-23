from typing import Optional


class ParticipantRegistry:
    """
    Central registry for participant metadata (e.g., height).
    Can be extended later (age, weight, condition, etc.).
    """

    _HEIGHTS = {
        "MS_22_01_25_06": 165,
        "MH_02_12_24_07": 186,
        "AM_02_12_24_10": 172,
        "BP_15_01_25_11": 164,
        "CT_15_01_25_12": 157,
        "DK_17_07_25_13": 182,
        "SH_17_07_25_14": 154,
        "IB_24_07_25_15": 193,
        "IS_24_07_25_16": 175,
        "SF_24_07_25_17": 176,
    }

    @classmethod
    def get_height(cls, participant_id: str) -> Optional[float]:
        """
        Returns height in centimeters.
        """
        h_cm = cls._HEIGHTS.get(participant_id.upper())
        return h_cm
