from pydantic import BaseModel, Field
from typing import Optional
from .temporal_metric_dataclass import TemporalMetrics
from .spatial_metric_dataclass import SpatialMetrics
from .derived_metric_dataclass import DerivedMetrics


class GaitMetadata(BaseModel):
    """
    Metadata describing the context of a gait recording/session.
    """

    participant_id: str = Field(
        ..., description="Unique identifier for the participant"
    )

    pass_number: int = Field(
        ..., ge=1, description="Sequential pass number (must be >= 1)"
    )

    # Optional contextual information
    trial_id: Optional[int] = Field(None, description="Optional trial identifier")

    fps: Optional[float] = Field(
        None, gt=0, description="Frames per second of the recording"
    )

    file: Optional[str] = Field(None, description="Source file name or path")


class GaitAnalysisResult(BaseModel):
    """
    Main container for all gait analysis outputs for a single trial/pass.

    Combines:
    - Temporal metrics (timing-related)
    - Spatial metrics (distance-related)
    - Derived metrics (overall trial metrics like cadence and velocity)
    - Metadata (context about the recording)
    """

    # Core gait metrics
    temporal: TemporalMetrics
    spatial: SpatialMetrics

    # Optional derived metrics (may not always be computed)
    derived: Optional[DerivedMetrics] = None

    # Metadata describing the data source
    metadata: GaitMetadata
