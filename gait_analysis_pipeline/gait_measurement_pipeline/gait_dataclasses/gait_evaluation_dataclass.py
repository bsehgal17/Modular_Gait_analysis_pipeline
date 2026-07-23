from dataclasses import dataclass
from typing import Dict


@dataclass
class MetricStats:
    """
    Stores statistical comparison metrics between two signals
    (predicted vs ground truth gait data).
    """

    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    percent_error: float  # Mean signed error (systematic offset)
    std: float  # Standard deviation of the error
    r: float  # Correlation coefficient (e.g., Pearson r)
    bias: float  # Systematic error
    icc: float  # Intraclass correlation coefficient
    loa_lower: float  # Lower limit of agreement
    loa_upper: float  # Upper limit of agreement


@dataclass
class ComparisonResult:
    """
    Stores comparison results for a single participant.

    Metrics are grouped into categories:
    - temporal: timing-related metrics (e.g., step time)
    - spatial: distance-related metrics (e.g., step length)
    - derived: higher-level metrics (e.g., cadence, velocity)
    """

    participant_id: str  # Unique identifier for the participant

    # Each dictionary maps a metric name (e.g., "step_time")
    # to its corresponding statistical comparison results.
    temporal_stats: Dict[str, MetricStats]
    spatial_stats: Dict[str, MetricStats]
    derived_stats: Dict[str, MetricStats]
