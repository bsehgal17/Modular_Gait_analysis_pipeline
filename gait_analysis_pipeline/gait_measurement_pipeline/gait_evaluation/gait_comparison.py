from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd

from gait_measurement_pipeline.gait_dataclasses.gait_evaluation_dataclass import (
    ComparisonResult,
)
from gait_measurement_pipeline.gait_evaluation.metrics import (
    compute_stats,
    generate_bland_altman_per_metric,
)
from gait_measurement_pipeline.gait_dataclasses.gait_results_dataclass import (
    GaitAnalysisResult,
)


# Type aliases for clarity
MetricValues = Tuple[List[float], List[float]]  # (pipeline_values, gaitrite_values)
MetricDict = Dict[str, MetricValues]


class GaitMetricsComparator:
    """
    Compare gait metrics from pipeline output vs. GaitRite reference.

    Responsibilities:
    -----------------
    1. Align pipeline and GaitRite results per participant and pass
    2. Export raw per-pass Excel files for manual correction/alignment
    3. Re-load Excel files and compute statistical comparisons

    Metrics:
    --------
    - Temporal: stance, stride, step, swing times
    - Spatial: stride length, step length
    - Derived: cadence, velocity
    """

    def __init__(
        self,
        pipeline_results: List[GaitAnalysisResult],
        gaitrite_results: List[GaitAnalysisResult],
    ) -> None:
        """
        Args:
            pipeline_results: Pipeline-generated gait results
            gaitrite_results: Reference GaitRite results
        """
        self.pipeline: List[GaitAnalysisResult] = pipeline_results
        self.gaitrite: List[GaitAnalysisResult] = gaitrite_results

    # -------------------------
    # Utility: index by pass number
    # -------------------------
    def _index_by_pass(
        self, data: List[GaitAnalysisResult]
    ) -> Dict[int, GaitAnalysisResult]:
        """
        Create mapping: pass_number → GaitAnalysisResult.

        Assumption:
            pass_number is unique per participant.

        Returns:
            Dict[int, GaitAnalysisResult]
        """
        return {r.metadata.pass_number: r for r in data}

    # -------------------------
    # Utility: group by participant
    # -------------------------
    def _group_by_participant(
        self, data: List[GaitAnalysisResult]
    ) -> Dict[str, List[GaitAnalysisResult]]:
        """
        Group results by participant_id.

        Returns:
            Dict[participant_id, List[GaitAnalysisResult]]
        """
        grouped: Dict[str, List[GaitAnalysisResult]] = {}

        for r in data:
            pid: str = r.metadata.participant_id
            grouped.setdefault(pid, []).append(r)

        return grouped

    # -------------------------
    # Export raw Excel per pass
    # -------------------------
    def export_raw_per_pass(self, output_dir: Path) -> None:
        """
        Export combined pipeline vs GaitRite values into Excel files (per pass).
        Focuses on chronological 'all_x' metrics rather than splitting L/R.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_group = self._group_by_participant(self.pipeline)
        gaitrite_group = self._group_by_participant(self.gaitrite)

        for pid in pipeline_group:
            if pid not in gaitrite_group:
                continue

            participant_dir: Path = output_dir / pid
            participant_dir.mkdir(parents=True, exist_ok=True)

            p_index = self._index_by_pass(pipeline_group[pid])
            g_index = self._index_by_pass(gaitrite_group[pid])

            common_passes = p_index.keys() & g_index.keys()

            for pass_number in common_passes:
                p = p_index[pass_number]
                g = g_index[pass_number]

                # ---------------- TEMPORAL (Combined) ----------------
                # Mapping the Excel Column Name to the Dataclass Attribute Name
                temporal_map = {
                    "stance_times": "stance_times",
                    "stride_times": "stride_times",
                    "step_times": "step_times",
                    "swing_times": "swing_times",
                }

                temporal_data = {}
                for col_name, attr in temporal_map.items():
                    # Extract values from the combined lists established in previous steps
                    p_vals = [e.value for e in getattr(p.temporal, attr, [])]
                    g_vals = [e.value for e in getattr(g.temporal, attr, [])]

                    temporal_data[f"gaitrite_{col_name}"] = g_vals
                    temporal_data[f"pipeline_{col_name}"] = p_vals

                df_temporal = pd.DataFrame({
                    k: pd.Series(v) for k, v in temporal_data.items()
                })

                # ---------------- SPATIAL (Combined) ----------------
                spatial_map = {
                    "Stride_lengths": "stride_lengths",
                    "Step_lengths": "step_lengths",
                }

                spatial_data = {}
                for col_name, attr in spatial_map.items():
                    p_vals = [e.value for e in getattr(p.spatial, attr, [])]
                    g_vals = [e.value for e in getattr(g.spatial, attr, [])]

                    spatial_data[f"gaitrite_{col_name}"] = g_vals
                    spatial_data[f"pipeline_{col_name}"] = p_vals

                df_spatial = pd.DataFrame({
                    k: pd.Series(v) for k, v in spatial_data.items()
                })

                # ---------------- DERIVED ----------------
                # Derived metrics remain as single summary values
                derived_data = {}
                if p.derived and g.derived:
                    derived_data["gaitrite_cadence"] = [g.derived.avg_cadence]
                    derived_data["pipeline_cadence"] = [p.derived.avg_cadence]
                    derived_data["gaitrite_velocity"] = [g.derived.avg_velocity]
                    derived_data["pipeline_velocity"] = [p.derived.avg_velocity]

                df_derived = pd.DataFrame(derived_data)

                # ---------------- SAVE ----------------
                output_file: Path = participant_dir / f"{pid}_pass_{pass_number}.xlsx"

                with pd.ExcelWriter(output_file) as writer:
                    df_temporal.to_excel(
                        writer, sheet_name="temporal_combined", index=False
                    )
                    df_spatial.to_excel(
                        writer, sheet_name="spatial_combined", index=False
                    )
                    df_derived.to_excel(writer, sheet_name="derived", index=False)

    # -------------------------
    # Comparison using Excel
    # -------------------------
    def compare(self, excel_root: Path) -> List[ComparisonResult]:
        """
        Compare metrics using Excel files (after manual alignment).

        Args:
            excel_root (Path): Root folder containing participant subfolders

        Returns:
            List[ComparisonResult]: Statistical comparison per participant
        """

        results: List[ComparisonResult] = []

        for participant_dir in excel_root.iterdir():
            if not participant_dir.is_dir():
                continue

            pid: str = participant_dir.name
            excel_files: List[Path] = list(participant_dir.glob("*.xlsx"))

            temporal_data: MetricDict = {}
            spatial_data: MetricDict = {}
            derived_data: MetricDict = {}

            for file in excel_files:
                xls = pd.ExcelFile(file)

                # ---------------- TEMPORAL ----------------
                if "temporal_combined" in xls.sheet_names:
                    df = pd.read_excel(xls, "temporal_combined")

                    for col in df.columns:
                        if col.startswith("gaitrite_"):
                            metric = col.replace("gaitrite_", "")

                            g_vals = df[col].dropna().tolist()
                            p_vals = df[f"pipeline_{metric}"].dropna().tolist()

                            temporal_data.setdefault(metric, ([], []))
                            temporal_data[metric][0].extend(p_vals)
                            temporal_data[metric][1].extend(g_vals)

                # ---------------- SPATIAL ----------------
                if "spatial_combined" in xls.sheet_names:
                    df = pd.read_excel(xls, "spatial_combined")

                    for col in df.columns:
                        if col.startswith("gaitrite_"):
                            metric = col.replace("gaitrite_", "")

                            g_vals = df[col].dropna().tolist()
                            p_vals = df[f"pipeline_{metric}"].dropna().tolist()

                            spatial_data.setdefault(metric, ([], []))
                            spatial_data[metric][0].extend(p_vals)
                            spatial_data[metric][1].extend(g_vals)

                # ---------------- DERIVED ----------------
                # if "derived" in xls.sheet_names:
                #     df = pd.read_excel(xls, "derived")

                #     for col in df.columns:
                #         if col.startswith("gaitrite_"):
                #             metric = col.replace("gaitrite_", "")

                #             g_vals = df[col].dropna().tolist()
                #             p_vals = df[f"pipeline_{metric}"].dropna().tolist()

                #             derived_data.setdefault(metric, ([], []))
                #             derived_data[metric][0].extend(p_vals)
                #             derived_data[metric][1].extend(g_vals)

            # ---------------- COMPUTE STATS ----------------
            result: ComparisonResult = ComparisonResult(
                participant_id=pid,
                temporal_stats={
                    k: compute_stats(v[0], v[1]) for k, v in temporal_data.items()
                },
                spatial_stats={
                    k: compute_stats(v[0], v[1]) for k, v in spatial_data.items()
                },
                derived_stats={
                    k: compute_stats(v[0], v[1]) for k, v in derived_data.items()
                },
            )

            results.append(result)

            generate_bland_altman_per_metric(
                temporal_data, result.temporal_stats, "temporal", pid
            )
            generate_bland_altman_per_metric(
                spatial_data, result.spatial_stats, "spatial", pid
            )
            generate_bland_altman_per_metric(
                derived_data, result.derived_stats, "derived", pid
            )

        return results
