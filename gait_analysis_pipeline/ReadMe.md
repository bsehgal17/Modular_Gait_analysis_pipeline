# Gait Analysis Pipeline

Computes spatiotemporal gait metrics from pose estimation keypoint data, and evaluates pipeline output against GAITRite ground truth.

## Overview

This module has three main workflows, run from `main.py`:

1. **Gait metric extraction** — computes gait metrics (temporal and spatial) from pose estimation JSON files.
2. **GAITRite conversion** — converts raw GAITRite CSV files into the standard dataclass format used for comparison.
3. **Evaluation** — compares pipeline-extracted gait metrics against GAITRite ground truth and outputs accuracy results (MAE, RMSE, etc.).

## Requirements

- Pose estimation JSON files in the pipeline's **standard dataclass format** (see [`Pose_estimation_codes/`](./Pose_estimation_codes) for generating these from video). Format specification: [docs/pose_estimation_dataclass.md](../docs/pose_estimation_dataclass.md).
- GAITRite ground truth CSV files, needed for the evaluation workflow.
## Data Formats

- **Pose estimation dataclass** — the standardized format that pose estimation JSON files must follow before being passed into the pipeline. See [docs/pose_estimation_dataclass.md](../docs/pose_estimation_dataclass.md).
- **Gait results dataclass** — the standardized format used for both pipeline-extracted gait metrics and converted GAITRite results, required for the evaluation step. See [docs/gait_results_dataclass.md](../docs/gait_results_dataclass.md).

## Installation

```bash
cd Gait_Analysis_pipeline
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

- `config_yamls/gait_measurement/gait_pipeline.yaml` — gait measurement pipeline settings. This file contains key runtime parameters, including:

  - `fps` — video frame rate used for filtering and gait event detection.
  - `joint_enum` — specifies the Python module and class defining the joint/keypoint layout corresponding to the pose estimation model output. The `module` field specifies the location of the joint enumeration class, and the `class` field specifies the required joint layout (e.g., `PredJointsCOCOWholebody` for RTMW or `OpenPoseBODY25` for OpenPose).

Example `gait_pipeline.yaml`:

```yaml
fps: 59.91706750704926

joint_enum:
  module: "pose_estimation.enums.joint_enum"
  class: "PredJointsCOCOWholebody"
```

## Usage

Edit the paths and settings in `gait_metric_pipeline.py`, then run:

```bash
python gait_metric_pipeline.py
```

### 1. Extract gait metrics from pose estimation data

```python
from pathlib import Path

from gait_measurement_pipeline.pipeline import GaitMeasurementPipeline
from gait_measurement_pipeline.config.gait_config import load_pipeline_config


config = load_pipeline_config(Path("config_yamls/gait_pipeline.yaml"))

pipeline = GaitMeasurementPipeline(config=config)

pipeline.run(
    input_dir=Path("path/to/pose_estimation_json_folder"),
    output_dir=Path("path/to/gait_output"),
)
```

- **Input:** folder of pose estimation JSON files in standard dataclass format.
- **Output:** extracted gait metrics written to `output_dir`.

### 2. Convert GAITRite CSVs to standard format

```python
from gait_measurement_pipeline.gaitrite_loader.gaitrite_main_runner import (
    run_gaitrite_folder,
)
from pathlib import Path

run_gaitrite_folder(
    input_folder=Path("path/to/gaitrite_csv_folder"),
    output_folder=Path("path/to/gaitrite_results"),
)
```

- **Input:** folder of raw GAITRite CSV files (one per participant).
- **Output:** GAITRite data converted into standard dataclass format, saved to `output_folder`.

### 3. Evaluate pipeline output against GAITRite ground truth

```python
from gait_measurement_pipeline.gait_evaluation.gait_evaluation_runner import (
    GaitEvaluationRunner,
)
from pathlib import Path

runner = GaitEvaluationRunner(
    pipeline_folder=Path("path/to/gait_output"),
    gaitrite_folder=Path("path/to/gaitrite_results"),
    output_file=Path("comparison_results.xlsx"),
)
runner.run()
```

- **Input:** pipeline gait metrics (from step 1) and converted GAITRite results (from step 2), both in standard dataclass format.
- **Output:** comparison results (MAE, RMSE, percent error) written to an Excel file.

## Notes

- Make sure the [`joint_enum`](pose_estimation\enums\joint_enum.py)  selected matches the pose estimation model used to generate your input JSON files — mismatched joint layouts will produce incorrect results.
- The evaluation step requires both pipeline output and GAITRite data to already be in standard dataclass format before comparison.