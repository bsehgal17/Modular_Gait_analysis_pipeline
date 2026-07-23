from pydantic import (
    BaseModel,
    ConfigDict,
)
from typing import Any


class ProcessingStep(BaseModel):
    """
    Generic record of any pose processing operation:
    filtering, smoothing, normalization, scaling, etc.
    """

    model_config = ConfigDict(frozen=True)

    step_name: str  # or Enum if you want strict control
    params: dict[str, Any]
