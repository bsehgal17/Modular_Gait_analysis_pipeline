# RTMW Detection Pipeline

## Overview

The RTMW Detection Pipeline performs whole-body human pose estimation on videos using **MMDetection** for person detection and **RTMW (RTMPose WholeBody)** from **MMPose** for keypoint estimation.

The pipeline is fully configuration-driven through YAML files and requires no code changes for standard use cases.

For each input video, the pipeline automatically:

- discovers videos recursively within a dataset directory
- detects people in every frame
- estimates whole-body keypoints
- tracks people across frames
- converts predictions into the project's standardized pose representation
- saves standardized pose outputs, raw RTMW output, and an optional visualization video

> **Model source:** RTMW model configs, checkpoints, and documentation are maintained upstream in the [MMPose GitHub repository](https://github.com/open-mmlab/mmpose). This project uses **RTMW-X**. See [Model Source & Configs](#model-source--configs) below for details and for how to switch to other MMPose models.

> **Data format reference:** All standardized pose outputs produced by this pipeline follow the format described in [`pose_estimation_dataclass.md`](../../docs/pose_estimation_dataclass.md ). Refer to that document for the exact schema of the saved JSON/pickle files.

---

## Pipeline Structure

Configuration is fully separated from execution:

```
              +--------------------+
              | global_config.yaml |
              +--------------------+
                        │
                        ▼
              +-----------------------+
              | pipeline_config.yaml  |
              +-----------------------+
                        │
                        ▼
                      main.py
                        │
                        ▼
                 pipeline_runner.py
                        │
                        ▼
              main_handler._handle_detect_command
                        │
                        ▼
              detection_pipeline_rtmw.run_detection_pipeline
                        │
                        ▼
              Standardized Output Files
```

There are two configuration files:

- `global_config.yaml` — shared, environment-level settings (dataset root, output root, supported video extensions)
- `pipeline_config.yaml` — settings specific to one pipeline run (dataset, model, processing/tracking/filter parameters)

---

## Directory Structure

```
pose_estimation_codes/

├── main.py                  # Entry point — the only way to run the pipeline
├── pipeline_runner.py       # Invoked by main.py for each step (detect / noise / filter / ...)
├── main_handler.py          # Command handlers, dispatched to by pipeline_runner.py
├── cli.py                   # Argument parsing

├── requirements.txt         # All Python dependencies for this project

├── config/
│   ├── global_config.py
│   └── pipeline_config.py

├── config_yamls/
│   ├── global_config.yaml
│   ├── gait_pipeline.yaml   # List of pipelines/steps run by main.py
│   └── detector/
│       └── detect_rtmw.yaml

├── detection_pipeline_rtmw.py   # RTMW/MMPose detection implementation
├── pose_estimation/
├── video_processing/
└── utils/
```

> **Dependencies:** All required Python packages are listed in [`requirements.txt`](../requirements.txt). Install them before running the pipeline, e.g. `pip install -r pose_estimation_codes/requirements.txt`.

---

## Configuration Files

### 1. Global Configuration (`global_config.yaml`)

```yaml
paths:
  input_dir: /path/to/datasets
  output_dir: /path/to/pipeline_results

video:
  extensions:
    - .mp4
    - .avi
    - .mov
    - .mkv
```

| Field | Description |
|---|---|
| `paths.input_dir` | Root directory where all datasets live. |
| `paths.output_dir` | Root directory where all pipeline outputs are written. Each run creates its own timestamped subfolder here. |
| `video.extensions` | File extensions searched for recursively when scanning a dataset. |

**Sample dataset layout**

```
/path/to/datasets/
├── cropped_uvic_pass_videos/
│   ├── subject01/
│   │   ├── walk01.mp4
│   │   └── walk02.mp4
│   └── subject02/
└── another_dataset/
```

---

### 2. Pipeline Configuration (`pipeline_config.yaml`)

#### Dataset path

```yaml
paths:
  dataset: cropped_uvic_pass_videos
```

`dataset` is resolved **relative to** `global_config.paths.input_dir`, so the example above resolves to:

```
/path/to/datasets/cropped_uvic_pass_videos
```

This directory is searched recursively for supported video files, and the same subdirectory structure is preserved in the output.

#### Models

```yaml
models:
  detector: rtmw

  det_config: /path/to/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py
  det_checkpoint: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

  pose_config: /path/to/mmpose/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-x_8xb320-270e_cocktail14-384x288.py
  pose_checkpoint: /path/to/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-*.pth
```

| Field | Description |
|---|---|
| `detector` | Selects the detection backend. `rtmw` selects the MMDetection + RTMPose pipeline. |
| `det_config` | MMDetection person-detector config file. |
| `det_checkpoint` | Checkpoint for the person detector. Can be a local path or a remote URL — if remote, it is downloaded on first use. |
| `pose_config` | RTMW pose model config file, from the MMPose repo. |
| `pose_checkpoint` | Pretrained RTMW weights, from the MMPose repo. |

#### Processing parameters

```yaml
processing:
  device: cuda:0
  nms_threshold: 0.5
  detection_threshold: 0.3
  kpt_threshold: 0.3
```

| Field | Description |
|---|---|
| `device` | Inference device, e.g. `cuda:0`, `cuda:1`, or `cpu`. |
| `nms_threshold` | Non-Maximum Suppression threshold used during person detection. |
| `detection_threshold` | Minimum bounding-box confidence to keep a detection. Lower values detect more people but increase false positives. |
| `kpt_threshold` | Minimum keypoint confidence used for visualization/display. |

---

## Model Source & Configs

RTMW is part of the [MMPose](https://github.com/open-mmlab/mmpose) model zoo and is not vendored into this repository — `pose_config` and `pose_checkpoint` in `pipeline_config.yaml` must point at files sourced from the MMPose repo.

- **MMPose repository:** https://github.com/open-mmlab/mmpose
- **RTMW model page (configs, variants, checkpoints):** https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md

This project uses the **RTMW-X** variant. If you need a different accuracy/speed trade-off, MMPose also publishes smaller RTMW variants (e.g. RTMW-L, RTMW-M) on the same page.

**Running a different MMPose pose estimation model:** the pipeline is not hard-wired to RTMW specifically — any other MMPose 2D pose estimation model can be run through the same `detect` step by updating `pipeline_config.yaml`:

- point `pose_config` and `pose_checkpoint` at the config/checkpoint files for the desired model (from the corresponding page in the MMPose repo), and
- update `det_config` / `det_checkpoint` if a different person detector is required for that model.

No code changes are required — only the paths and model names in the config.

---

## Running the Pipeline

The entry point for getting the pose estimation body keypoint results is:
```bash
python main.py
```

### Environment Setup

Before running the pipeline, create a dedicated conda environment and install dependencies:

```bash
conda create -n pose_estimation python=3.9 -y
conda activate pose_estimation
pip install -r pose_estimation_codes/requirements.txt
```

See [`requirements.txt`](../../pose_estimation_codes/requirements.txt) for the full dependency list.

`main.py` reads `config_yamls/gait_pipeline.yaml` and executes every configured step, in order, as a subprocess call to `pipeline_runner.py`. If a step fails (non-zero exit code), execution stops and the error is logged.

Example `gait_pipeline.yaml`:

```yaml
global_config_file: config_yamls/global_config.yaml

pipelines:
  - name: rtmw_pipeline
    steps:
      - command: detect
        config_file: config_yamls/pipelines/rtmw_detection.yaml
```

- `global_config_file` at the top level is used as the default global config for every pipeline/step unless overridden.
- Each entry under `steps` maps directly to a `pipeline_runner.py <command>` call, using that step's `config_file` as the pipeline config.

---

## What Happens During Detection

For every discovered input video, `detection_pipeline_rtmw.run_detection_pipeline` does the following:

1. Initializes the detector, pose estimator, visualizer, and frame processor once for the whole run.
2. Opens the video and creates a matching output subfolder that mirrors the input's relative path.
3. Reads the video frame by frame.
4. Detects people in the frame.
5. Estimates RTMW whole-body keypoints for each detected person.
6. Writes the (optionally annotated) frame to the output video.
7. After all frames are processed:
   - saves the **raw**, pre-standardization RTMW output,
   - converts and saves the **standardized** pose result (JSON + pickle) — see [`pose_estimation_dataclass.md`](../../docs/pose_estimation_dataclass.md) for the schema,
   - logs where each output was written.

---

## Output Directory Structure

Each run is written to its own directory under `output_dir/<pipeline_name>/detect/`, using a timestamped run folder:

```
pipeline_results/
    rtmw_pipeline/
        detect/
            run_2026-07-23_15-30-10/
                config_snapshot/
                cropped_uvic_pass_videos/
                    subject01/
                        walk01/
                            walk01.json
                            walk01.pkl
                            walk01_raw_rtmw.json
                            walk01.mp4
```

The `config_snapshot/` folder stores a copy of the exact config used for that run, for reproducibility. The exact run-folder name depends on the run timestamp.

---

## Output Files

| File | Description |
|---|---|
| `<video>.json` | Standardized pose representation used by all downstream pipeline stages (metadata, per-frame tracked people, keypoints, confidence scores). Schema documented in [`../docs/pose_estimation_dataclass.md`](../docs/pose_estimation_dataclass.md). |
| `<video>.pkl` | Pickled version of the same standardized pose data, for faster loading in Python. |
| `<video>_raw_rtmw.json` | Original, unconverted RTMW predictions. Useful for debugging or reproducing results outside the standard pipeline. |
| `<video>.mp4` | Output video with pose skeleton overlays (only generated if visualization is enabled). |

---

## Troubleshooting

**No videos found**
- Verify `global_config.paths.input_dir` points to the correct root directory.
- Verify `pipeline_config.paths.dataset` exists under that input directory.
- Confirm the video files use one of the extensions listed in `global_config.video.extensions`.

**CUDA errors**
- Confirm NVIDIA drivers are installed and CUDA is available.
- Confirm the configured `processing.device` (e.g. `cuda:0`) exists on the machine.
- Confirm the required model checkpoints are accessible.
- To run on CPU instead, set:
  ```yaml
  processing:
    device: cpu
  ```

**Missing model checkpoints**
- Ensure `det_config`, `det_checkpoint`, `pose_config`, and `pose_checkpoint` all point to valid, reachable paths.
- `pose_config` / `pose_checkpoint` for RTMW come from the [MMPose RTMW model page](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md) — confirm you're using the RTMW-X variant used by this project.
- If `det_checkpoint` is a remote URL, make sure internet access is available on first run so weights can be downloaded.

**Missing/incompatible dependencies**
- Reinstall from [`pose_estimation_codes/requirements.txt`](../pose_estimation_codes/requirements.txt): `pip install -r pose_estimation_codes/requirements.txt`.

---

## Typical Workflow

1. Install dependencies from [`pose_estimation_codes/requirements.txt`](../pose_estimation_codes/requirements.txt).
2. Configure `global_config.yaml` with your input and output directories.
3. Configure a pipeline config (e.g. `config_yamls/detector/rtmw_detection.yaml`) with the dataset name and RTMW-X model settings, using config/checkpoint files from the [MMPose RTMW page](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md).
4. Run detection with `python main.py`.
5. Review the generated JSON, PKL, raw RTMW output, and visualization video under the run's output folder — see [`pose_estimation_dataclass.md`](../../docs/pose_estimation_dataclass.md) for how to read the standardized JSON/pickle files.