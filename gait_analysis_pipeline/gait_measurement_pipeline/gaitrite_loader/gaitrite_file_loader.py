from pathlib import Path
from typing import List, Optional
import pandas as pd

from gait_measurement_pipeline.gait_dataclasses.gait_results_dataclass import (
    GaitAnalysisResult,
    GaitMetadata,
)
from gait_measurement_pipeline.gait_dataclasses.temporal_metric_dataclass import (
    TemporalMetrics,
)
from gait_measurement_pipeline.gait_dataclasses.spatial_metric_dataclass import (
    SpatialMetrics,
)
from gait_measurement_pipeline.gait_dataclasses.derived_metric_dataclass import (
    DerivedMetrics,
)

from gait_measurement_pipeline.gaitrite_loader.pass_detector import PassDetector
from gait_measurement_pipeline.gaitrite_loader.temporal_event_builder import (
    TemporalEventBuilder,
)
from gait_measurement_pipeline.gaitrite_loader.spatial_event_builder import (
    SpatialEventBuilder,
)
from gait_measurement_pipeline.gait_dataclasses.participant_registry import (
    ParticipantRegistry,
)


class GaitRiteFileLoader:
    """
    Loader for a single GaitRite Excel file.

    Responsibilities:
    - Read and clean raw Excel data
    - Detect and assign pass numbers
    - Build temporal, spatial, and derived gait metrics
    - Package results into GaitAnalysisResult objects
    - Maintain consistent pass numbering across files
    """

    def __init__(self, file_path: Path, start_pass: int = 1) -> None:
        """
        Args:
            file_path: Path to the GaitRite Excel file
            start_pass: Starting pass number (used for global sequencing)
        """
        self.file_path: Path = Path(file_path)
        self.start_pass: int = start_pass

        # Pass detector assigns pass numbers based on temporal gaps
        self.pass_detector: PassDetector = PassDetector()

    # ---------------------------
    # Helper: Extract participant ID
    # ---------------------------
    def _extract_participant_id(self) -> str:
        """
        Extract participant ID from the parent folder name.

        Returns:
            Uppercase participant ID string.
        """
        pid: str = self.file_path.parent.name
        return pid.strip().upper()

    # ---------------------------
    # Main API: Load GaitRite file
    # ---------------------------
    def load(self) -> List[GaitAnalysisResult]:
        """
        Load and process the Excel file into structured gait results.

        Pipeline:
            1. Read Excel
            2. Clean and filter rows
            3. Assign pass numbers
            4. Split into passes
            5. Compute temporal, spatial, and derived metrics
            6. Return structured results

        Returns:
            List of GaitAnalysisResult objects (one per detected pass)
        """

        # ---------------------------
        # Step 1: Read and clean Excel
        # ---------------------------
        df: pd.DataFrame = pd.read_excel(self.file_path)

        # Normalize column names (remove trailing spaces)
        df.columns = [str(c).strip() for c in df.columns]

        # Keep only valid gait events
        df = df[df["First Contact Time"].notna()].copy()

        # ---------------------------
        # Step 2: Assign pass numbers
        # ---------------------------
        df = self.pass_detector.assign_pass_numbers(df)

        # Extract unique passes (order depends on data appearance)
        passes = df["Computed Pass"].dropna().unique()

        results: List[GaitAnalysisResult] = []

        # Initialize global pass counter
        pass_counter: int = self.start_pass

        # Extract participant metadata
        pid: str = self._extract_participant_id()

        # Retrieve participant height (in meters)
        height: Optional[float] = ParticipantRegistry.get_height(pid)

        # ---------------------------
        # Step 3: Process each pass
        # ---------------------------
        for p in passes:
            # Subset DataFrame for this pass
            pass_df: pd.DataFrame = df[df["Computed Pass"] == p]

            # Initialize builders (stateless per pass)
            temporal_builder: TemporalEventBuilder = TemporalEventBuilder()
            spatial_builder: SpatialEventBuilder = SpatialEventBuilder(height=height)

            # ---------------------------
            # Compute temporal metrics
            # ---------------------------
            temporal: TemporalMetrics = temporal_builder.build_temporal_metrics(pass_df)

            # ---------------------------
            # Compute spatial metrics
            # ---------------------------
            spatial: SpatialMetrics = spatial_builder.build_spatial_metrics(pass_df)

            # ---------------------------
            # Compute derived metrics
            # ---------------------------
            cadence_series: pd.Series = pass_df["Cadence"].dropna()
            velocity_series: pd.Series = pass_df["Velocity"].dropna()

            derived: DerivedMetrics = DerivedMetrics(
                avg_cadence=float(cadence_series.iloc[0])
                if not cadence_series.empty
                else None,
                avg_velocity=float(velocity_series.iloc[0])
                if not velocity_series.empty
                else None,
            )

            # ---------------------------
            # Assemble final result
            # ---------------------------
            result: GaitAnalysisResult = GaitAnalysisResult(
                temporal=temporal,
                spatial=spatial,
                derived=derived,
                metadata=GaitMetadata(
                    participant_id=pid,
                    file=self.file_path.name,  # traceability
                    pass_number=pass_counter,
                ),
            )

            results.append(result)

            # Increment global pass counter
            pass_counter += 1

        return results
