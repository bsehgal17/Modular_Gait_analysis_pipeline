from pydantic import BaseModel, Field
from typing import List
from .gait_results_dataclass import GaitAnalysisResult


# ======================
# Pass / Trial / Participant Models
# ======================


class PassResult(BaseModel):
    """
    Represents the result of a single walking pass.

    A "pass" is typically one traversal (e.g., walking across a walkway).
    """

    pass_id: str = Field(..., description="Unique identifier for this pass")

    pass_number: int = Field(..., ge=1, description="Sequential pass number (>= 1)")

    result: GaitAnalysisResult  # Full gait analysis output for this pass


class Trial(BaseModel):
    """
    Groups multiple passes under a single trial.

    A trial may contain repeated walking passes under the same condition.
    """

    trial_id: int = Field(..., description="Unique identifier for the trial")

    num_passes: int = Field(
        ..., ge=0, description="Expected number of passes in this trial"
    )

    passes: List[PassResult] = Field(
        ..., description="List of pass results belonging to this trial"
    )


class ParticipantData(BaseModel):
    """
    Top-level container for all gait data of a participant.

    Organizes trials, which in turn contain multiple passes.
    """

    participant_id: str = Field(
        ..., description="Unique identifier for the participant"
    )

    trials: List[Trial] = Field(
        ..., description="All trials recorded for this participant"
    )
