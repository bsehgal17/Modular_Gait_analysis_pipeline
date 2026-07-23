"""
Main script to run the Gait Measurement Pipeline as a standalone module.
"""

from gait_measurement_pipeline.pipeline import GaitMeasurementPipeline
from gait_measurement_pipeline.gaitrite_loader.gaitrite_main_runner import (
    run_gaitrite_folder,
)
from config.gait_config import load_pipeline_config
from gait_measurement_pipeline.gait_evaluation.gait_evaluation_runner import (
    GaitEvaluationRunner,
)
from pathlib import Path


def main():
    input_dir = Path(
        r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\pipeline_processing\rtmw\test"
    )
    output_dir = Path(
        r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\gait_output"
    )
    pipeline_config = Path(r"gait_analysis_pipeline\config_yamls\gait_pipeline.yaml")
    pipeline_config = load_pipeline_config(pipeline_config)

    runner = GaitMeasurementPipeline(config=pipeline_config)
    runner.run(
        input_dir,
        output_dir,
    )
    ###########################################################
    # input_folder = Path(
    #     r"C:\Users\BhavyaSehgal\Downloads\OneDrive_2026-05-27\Validation data - GaitRite 60 FPS"
    # )
    # output_folder = Path(
    #     r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\gaitrite_results"
    # )

    # run_gaitrite_folder(input_folder, output_folder)
    ###########################################################
    pipeline_folder = Path(
        r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\gait_output"
    )
    gaitrite_folder = Path(
        r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\gaitrite_results"
    )
    output_file = Path("comparison_results.xlsx")

    runner = GaitEvaluationRunner(
        pipeline_folder=pipeline_folder,
        gaitrite_folder=gaitrite_folder,
        output_file=output_file,
    )
    runner.run()


if __name__ == "__main__":
    main()
