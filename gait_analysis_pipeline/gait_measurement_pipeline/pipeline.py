from pathlib import Path
from typing import List, Any, Optional, Tuple, Dict
import json
from collections import defaultdict
import re
import numpy as np

from filtering_scripts.filters.butterworth import ButterworthConfig
from filtering_scripts.filters.savgol import SavgolConfig
from filtering_scripts.filter_enums import FilterType
from gait_measurement_pipeline.gait_dataclasses.phase_detection_dataclass import (
    PhaseDetectionConfig,
)
from pose_estimation.pose_estimation_dataclasses.pose_estimation_dataclass import (
    PoseEstimationResult,
)
from pose_estimation.configs.pose_config import PoseConfig
from gait_measurement_pipeline.gait_dataclasses.gait_results_dataclass import (
    GaitAnalysisResult,
    GaitMetadata,
)
from gait_measurement_pipeline.gait_dataclasses.participant_gait_results_dataclass import (
    ParticipantData,
    Trial,
    PassResult,
)

from skeleton_normalizer.normalized_skel_points import PoseNormalizer
from skeleton_normalizer.height_signal_visualizer import HeightSignalVisualizer

from gait_measurement_pipeline.phase_detection.velocity_based_phase_detection import (
    VelocityPhaseDetector,
)
from filtering_scripts.skel_points_filter_processor import (
    FramePoseFilterProcessor,
)
from pose_estimation.processors.tracked_pose_resolver import TrackedPoseResolver
from skeleton_normalizer.height_estimator_resolver import HeightEstimatorResolver
from filtering_scripts.smoother_methods import (
    MedianSmoother,
    SavgolSmoother,
)
from gait_measurement_pipeline.gait_dataclasses.phase_detection_dataclass import (
    VisualizationConfig,
)
from pose_estimation.utils.body_connections import (
    BODY25_CONNECTION_NAMES,
    COCO_WHOLEBODY_CONNECTION_NAMES,
)
from pose_estimation.utils.create_body_connections import build_connections
from gait_measurement_pipeline.event_detection.compute_contact_times import (
    GaitEventDetector,
)
from gait_measurement_pipeline.gait_metric_calculator.temporal_metrics import (
    TemporalMetricsComputer,
)
from gait_measurement_pipeline.gait_metric_calculator.spatial_metrics import (
    SpatialMetricsComputer,
)
from gait_measurement_pipeline.gait_metric_calculator.gait_derived_metrics import (
    GaitDerivedMetricsComputer,
)
from skeleton_normalizer.height_calculator import HeightPipeline
from gait_measurement_pipeline.utils.compute_foot_velocities import (
    FootVelocityExtractor,
)
from skeleton_normalizer.height_interpolator import HeightOutlierInterpolator
from filtering_scripts.skel_points_interpolator import ConfidenceInterpolationProcessor
from gait_measurement_pipeline.utils.velocity_linear_interpolator import (
    VelocityOutlierInterpolator,
)
# -----------------------
# Helper Functions
# -----------------------


def serialize(obj: Any) -> Any:
    """
    Recursively convert Pydantic models into dicts/lists.
    """
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()

    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [serialize(v) for v in obj]

    return obj


def find_files(
    input_path: Path,
    json_ext: str = ".json",
    video_exts: tuple = (".mp4", ".mov", ".avi"),
) -> Tuple[List[Path], Dict[str, Path]]:
    """
    Recursively find JSON and video files.
    Returns:
        - list of JSON paths
        - dict mapping video stem (lowercase) -> Path
    """
    input_path = Path(input_path)

    if input_path.is_file():
        return [input_path], {}

    json_files = list(input_path.rglob(f"*{json_ext}"))

    video_files: dict[str, Path] = {}
    for ext in video_exts:
        for v in input_path.rglob(f"*{ext}"):
            video_files[v.stem.lower()] = v

    return json_files, video_files


def extract_participant_id(path: Path) -> str:
    """
    Extract participant ID from file name.
    Removes "_pass#" or "-pass#" suffixes.
    """
    match = re.split(r"pass[_\-]?\d+", path.stem.lower())
    pid = match[0].rstrip("_-") if match else path.stem
    return pid.upper()


def extract_pass_number(path: Path) -> Optional[int]:
    """
    Extract numeric pass number from filename. Returns None if not found.
    """
    match = re.search(r"pass[_\-]?(\d+)", path.stem.lower())
    return int(match.group(1)) if match else None


# -----------------------
# Main Pipeline Class
# -----------------------


class GaitMeasurementPipeline:
    def __init__(self, config: dict):
        self.config = config

    def run(
        self,
        input_path: Path,
        output_dir: Path,
    ) -> List[ParticipantData]:
        """
        Run full gait measurement pipeline:
        1. Load JSON & video files.
        2. Extract joint trajectories.
        3. Normalize trajectories.
        4. Detect gait phases.
        5. Compute temporal, spatial, and derived metrics.
        6. Save per-participant JSON output.
        """

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        json_files, video_map = find_files(input_path)

        print(f"\nFound {len(json_files)} JSON files")

        # Group files by participant ID
        participants: dict[str, list[Path]] = defaultdict(list)
        for f in json_files:
            pid = extract_participant_id(f)
            participants[pid].append(f)

        all_participants_results: list[ParticipantData] = []

        # Process each participant
        for pid, files in participants.items():
            print(f"\nParticipant: {pid}")

            files = sorted(files, key=lambda x: extract_pass_number(x) or 1e9)
            trials: dict[int, list[Path]] = defaultdict(list)

            # Group passes into trials
            for f in files:
                pass_num = extract_pass_number(f)
                if pass_num is None:
                    continue
                trial_id = (pass_num - 1) // 2 + 1
                trials[trial_id].append(f)

            participant_data = ParticipantData(participant_id=pid, trials=[])

            # Process trials
            for trial_id, trial_files in sorted(trials.items()):
                trial_data = Trial(
                    trial_id=trial_id, num_passes=len(trial_files), passes=[]
                )

                # Process each pass
                for json_file in trial_files:
                    video_file = video_map.get(json_file.stem.lower(), None)
                    result = self._run_single(
                        json_file=json_file,
                        video_file=video_file,
                        joint_enum=self.config["joint_enum"],
                    )
                    pass_num = extract_pass_number(json_file)

                    trial_data.passes.append(
                        PassResult(
                            pass_id=json_file.stem,
                            pass_number=pass_num,
                            result=result,
                        )
                    )

                participant_data.trials.append(trial_data)

            # Save participant JSON
            output_dir.mkdir(parents=True, exist_ok=True)
            participant_filename = f"{pid.upper()}.json"
            output_path = output_dir / participant_filename
            with open(output_path, "w") as f:
                json.dump(participant_data.model_dump(), f, indent=4)

            print(f"Saved: {output_path}")
            all_participants_results.append(participant_data)

        print("\nBatch processing complete!")
        return all_participants_results

    def _run_single(
        self,
        json_file: Path,
        video_file: Optional[Path],
        joint_enum: dict,
    ) -> GaitAnalysisResult:
        """
        Process a single pass:
        1. Load pose estimation JSON.
        2. Extract and normalize joint trajectories.
        3. Detect gait phases using velocity-based method.
        4. Compute temporal, spatial, and derived gait metrics.
        5. Return GaitAnalysisResult object.
        """

        with open(json_file) as file:
            data = json.load(file)

        # =========================================================
        # LOAD POSE ESTIMATION DATA
        # =========================================================

        pose_data = PoseEstimationResult.from_dict(data)

        # =========================================================
        # TRAJECTORY CONFIGURATION
        # =========================================================

        pose_config = PoseConfig(
            frames_per_second=self.config.get(
                "fps",
                59.91706750704926,
            ),
            joint_schema=joint_enum,
            rolling_median_window=3,
        )

        # =========================================================
        # FILTER CONFIGURATION
        # =========================================================

        # filter_config = ButterworthConfig(
        #     filter_type=FilterType.BUTTERWORTH,
        #     cutoff_frequency=2,
        #     filter_order=2,
        # )

        # Optional Savitzky-Golay configuration
        filter_config = SavgolConfig(
            filter_type=FilterType.SAVGOL,
            window_length=int(
                pose_config.frames_per_second * 0.5
            ),  # try 31 #make it agonstic to fps
            polynomial_order=3,
        )

        # =========================================================
        # BUILD FILTER DIRECTLY
        # =========================================================

        filter_impl = filter_config.build(
            fps=pose_config.frames_per_second,
        )

        # =========================================================
        # INITIALIZE PROCESSOR
        # =========================================================

        processor = FramePoseFilterProcessor(
            filter_impl=filter_impl, filter_config=filter_config
        )

        # =========================================================
        # RESOLVE TRACKED POSE SEQUENCE
        # =========================================================
        tracked_pose_resolver = TrackedPoseResolver()

        processing_steps = []

        pose_sequence, person_id = tracked_pose_resolver.resolve(pose_data)

        conf_processor = ConfidenceInterpolationProcessor(
            confidence_threshold=3, visibility_threshold=0.9
        )
        pose_sequence, conf_step = conf_processor.filter_low_scores(
            pose_sequence
        )  # chnage from filter to interpolation processor
        processing_steps.append(conf_step)

        # =========================================================
        # APPLY FILTERING to pose sequence
        # =========================================================

        filtered_pose_sequence, filter_step = processor.filter(pose_sequence)
        processing_steps.append(filter_step)

        # Below steps estimate the heights using above pose sequence,
        # drops the sudden spiking values, linearly interpolated and the smooth then either using svagol or median filter

        height_estimator = HeightEstimatorResolver().resolve(joint_enum)
        # smoother = MedianSmoother(window_size=pose_config.rolling_median_window)
        smoother = SavgolSmoother(
            window_length=int(pose_config.frames_per_second * 0.9), polyorder=3
        )
        pipeline = HeightPipeline(estimator=height_estimator, smoother=smoother)

        height_cleaner = HeightOutlierInterpolator(
            window_size=5,  # frames of context around each point
            z_thresh=1.0,  # lower = stricter (removes more), higher = looser
        )

        raw = height_estimator.estimate(filtered_pose_sequence)

        raw = height_cleaner.filter(raw)

        smoothed = smoother.smooth(raw)

        # Step 2: visualize (caller's choice — no longer buried in normalization)
        # HeightSignalVisualizer.plot(
        #     raw=np.array(raw.values),
        #     smoothed=np.array(smoothed.values),
        #     title=f"Height Signal — {height_estimator.__class__.name}",
        #     fps=pose_config.frames_per_second,
        # )

        # Then using above smoothes values each joint from pose sequence
        # for a frame is normalized from the height calculated from that particular frame

        normalizer = PoseNormalizer()
        # Step 3: normalize poses against the smoothed signal
        normalized_pose_sequence = normalizer.normalize(
            filtered_pose_sequence, smoothed
        )

        processing_steps.append(pipeline.to_processing_step())

        filtered_norm_skel_points, norm_filter_step = processor.filter(
            normalized_pose_sequence
        )
        processing_steps.append(norm_filter_step)
        pose_data = pose_data.add_processing_step(processing_steps)
        pose_data = pose_data.replace_poses(
            filtered_norm_skel_points, person_id=person_id
        )
        # pose_data.save_json(path)

        ## Phase detection
        phase_config = PhaseDetectionConfig(
            fps=self.config.get("fps", 59.91706750704926),
            prominence_ratio_peaks_min=self.config.get("prominence_ratio_peaks", 0.1),
            prominence_ratio_valleys_min=self.config.get(
                "prominence_ratio_valleys", 0.005
            ),
            visualization=VisualizationConfig(
                show_velocity_viewer=True,
                plot=False,
                show_comparison=False,
                save_svg=True,
            ),
        )

        connections = build_connections(joint_enum, COCO_WHOLEBODY_CONNECTION_NAMES)

        # Step 1: extract keypoints + compute raw velocities
        left_vel, right_vel, left_kp, right_kp = FootVelocityExtractor(
            joint_enum
        ).extract(filtered_norm_skel_points)

        velocity_cleaner = VelocityOutlierInterpolator(window_size=5, z_thresh=1.0)
        left_vel = velocity_cleaner.filter(left_vel)
        right_vel = velocity_cleaner.filter(right_vel)

        # Step 2: smooth velocities (on the signal, not the positions)
        left_vel = smoother.smooth(left_vel)
        right_vel = smoother.smooth(right_vel)

        # Step 3: detect phases + visualize
        phases = VelocityPhaseDetector(
            config=phase_config,
            joint_enum=joint_enum,
            connections=connections,
            video_path=video_file,
        ).run(
            left_vel=left_vel,
            right_vel=right_vel,
            left_kp=left_kp,
            right_kp=right_kp,
            raw_poses=filtered_pose_sequence,
        )

        # Gait event detection
        event_detector = GaitEventDetector(joint_enum=joint_enum)
        events = event_detector.detect(
            video_file=str(video_file) if video_file else None,
            phases=phases,
            poses=normalized_pose_sequence,
            do_plot=False,
            do_video=False,
        )

        # # Compute metrics
        temporal = TemporalMetricsComputer(events).compute()
        spatial = SpatialMetricsComputer(
            events, left_keypoints=left_kp, right_keypoints=right_kp
        ).compute()
        derived = GaitDerivedMetricsComputer(
            events,
            time=phases.time,
            fps=pose_config.frames_per_second,
            poses=pose_sequence,
            joint_enum=joint_enum,
        ).compute()

        return GaitAnalysisResult(
            temporal=temporal,
            spatial=spatial,
            derived=derived,
            metadata=GaitMetadata(
                participant_id=extract_participant_id(json_file),
                pass_number=extract_pass_number(json_file),
                trial_id=None,
                fps=self.config.get("fps", 60),
                file=json_file.stem,
            ),
        )
