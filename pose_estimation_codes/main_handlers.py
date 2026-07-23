import logging
import os
from pathlib import Path
from config.pipeline_config import PipelineConfig
from config.global_config import GlobalConfig
from utils.run_utils import make_run_dir, get_pipeline_io_paths

logger = logging.getLogger(__name__)


def _handle_detect_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    run_detection_pipeline = _get_detection_pipeline_fn(pipeline_config.models.detector)

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    video_folder = args.video_folder if args.video_folder else input_dir
    if args.video_folder:
        logger.info(f"Overriding video folder: {video_folder}")

    run_detection_pipeline(
        pipeline_config=pipeline_config,
        global_config=global_config,
        input_dir=video_folder,
        output_dir=step_out,
    )


def _handle_noise_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from noise_simulator import NoiseSimulator

    # Step 0: Set up base input/output paths
    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    # Step 1: Add noise to input videos
    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )
    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    input_folder = args.input_folder or input_dir
    output_folder = args.output_folder or step_out

    logger.info(f"Applying noise to videos from: {input_folder}")
    logger.info(f"Noisy videos will be saved to: {output_folder}")

    simulator = NoiseSimulator(pipeline_config, global_config)
    simulator.process_all_videos(str(input_folder), str(output_folder))

    # Step 2: Conditionally run detection only if models and detector are configured
    if (
        pipeline_config.models is not None
        and getattr(pipeline_config.models, "detector", None) is not None
    ):
        run_detection_pipeline = _get_detection_pipeline_fn(
            pipeline_config.models.detector
        )

        detect_out_dir = Path(output_folder) / "detect"
        detect_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running detection on noisy videos from: {output_folder}")
        logger.info(f"Detection results will be saved to: {detect_out_dir}")

        run_detection_pipeline(
            pipeline_config=pipeline_config,
            global_config=global_config,
            input_dir=str(output_folder),
            output_dir=str(detect_out_dir),
        )
    else:
        logger.info("No detection model configured; skipping detection step.")


def _handle_filter_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from filter_pipeline import run_keypoint_filtering_from_config

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    pipeline_config.paths.output_dir = str(step_out)

    if not pipeline_config.filter.input_dir:
        noise_dir = os.path.join(base_pipeline_out, args.pipeline_name, "noise")
        detect_dir = os.path.join(base_pipeline_out, args.pipeline_name, "detect")

        if os.path.exists(noise_dir):
            pipeline_config.filter.input_dir = noise_dir
            logger.info(f"Auto-selected input_dir from noise step: {noise_dir}")
        elif os.path.exists(detect_dir):
            pipeline_config.filter.input_dir = detect_dir
            logger.info(f"Auto-selected input_dir from detect step: {detect_dir}")
        else:
            logger.warning("Could not auto-resolve input_dir for filter.")

    run_keypoint_filtering_from_config(
        pipeline_config,
        global_config,
        output_dir=step_out,
    )


def _handle_assess_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    from evaluation_pipeline import run_pose_assessment_pipeline

    input_dir, base_pipeline_out = get_pipeline_io_paths(
        global_config.paths, pipeline_config.paths.dataset
    )

    run_dir = make_run_dir(
        base_out=base_pipeline_out,
        pipeline_name=args.pipeline_name,
        step_name=args.command,
        cfg_path=args.pipeline_config,
        global_config_obj=global_config,
        pipeline_config_obj=pipeline_config,
    )

    step_out = run_dir
    step_out.mkdir(parents=True, exist_ok=True)

    # Get confidence filtering parameters - prioritize config file, then command line
    # Removed unused variables for confidence filtering

    # For HumanEva, disable confidence filtering by passing zero thresholds
    min_bbox_confidence = 0.0
    min_keypoint_confidence = 0.0
    logger.info(
        "Disabling confidence filtering for HumanEva evaluation (all skeletons considered)"
    )

    if pipeline_config.evaluation.input_dir:
        logger.info(
            f"Using manually specified input_dir: {pipeline_config.evaluation.input_dir}"
        )
        run_pose_assessment_pipeline(
            pipeline_config,
            global_config,
            output_dir=step_out,
            input_dir=pipeline_config.evaluation.input_dir,
            min_bbox_confidence=min_bbox_confidence,
            min_keypoint_confidence=min_keypoint_confidence,
        )
        return

    step_candidates = ["detect", "noise", "filter"]

    # Regular step evaluation
    for step in step_candidates:
        step_dir = os.path.join(base_pipeline_out, args.pipeline_name, step)
        if os.path.exists(step_dir):
            step_eval_dir = step_out / step
            step_eval_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Running evaluation for step: {step}, using input_dir: {step_dir}"
            )
            pipeline_config.evaluation.input_dir = step_dir
            run_pose_assessment_pipeline(
                pipeline_config,
                global_config,
                output_dir=step_eval_dir,
                input_dir=step_dir,
                min_bbox_confidence=min_bbox_confidence,
                min_keypoint_confidence=min_keypoint_confidence,
            )
        else:
            logger.warning(
                f"Step output folder not found: {step_dir}, skipping evaluation."
            )


# Gait Measurement Pipeline handler
def _handle_gait_measurement_command(
    args, pipeline_config: PipelineConfig, global_config: GlobalConfig
):
    import os
    from gait_measurement_pipeline.pipeline import GaitMeasurementPipeline

    pipeline_name = (
        getattr(pipeline_config, "pipeline_name", None)
        or args.pipeline_name
        or "default_pipeline"
    )
    base_out = getattr(pipeline_config.paths, "output_dir", None) or "."

    # 1. Check if detection step is run
    detect_dir = os.path.join(base_out, pipeline_name, "detect")
    if os.path.exists(detect_dir):
        logger.info(f"Detection step found: {detect_dir}")
        detection_input_dir = detect_dir
    else:
        input_dir = args.input_dir or pipeline_config.paths.dataset
        if not input_dir or not os.path.exists(input_dir):
            raise ValueError(
                "Detection step not found. Please provide --input_dir with all JSON files."
            )
        logger.info(f"Using user-provided input_dir for gait measurement: {input_dir}")
        detection_input_dir = input_dir

    # 2. Check if filter step is run
    filter_dir = os.path.join(base_out, pipeline_name, "filter")
    if os.path.exists(filter_dir):
        logger.info(f"Filter step found: {filter_dir}")
        filter_input_dir = filter_dir
    else:
        logger.info("Filter step not found. Skipping filter.")
        filter_input_dir = None

    output_dir = args.output_dir or pipeline_config.paths.output_dir
    runner = GaitMeasurementPipeline(pipeline_config, global_config)
    # Pass both detection_input_dir and filter_input_dir to pipeline
    runner.run(detection_input_dir, output_dir, filter_input_dir=filter_input_dir)


def _get_detection_pipeline_fn(detector_name: str):
    detector_name = detector_name.lower()
    if detector_name == "dwpose":
        from detection_pipeline_DWPose import run_detection_pipeline

        logger.info("Using DWPose detection pipeline.")
    else:
        from detection_pipeline_rtmw import run_detection_pipeline

        logger.info("Using RTMW/MMpose detection pipeline.")
    return run_detection_pipeline
