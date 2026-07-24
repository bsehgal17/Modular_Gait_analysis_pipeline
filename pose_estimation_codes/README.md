# Pose Estimation

## Overview

This repository supports multiple human pose estimation frameworks under a common project structure. Each model has its own installation requirements, configuration files, and execution procedure.

To avoid duplicating documentation, each implementation is documented separately.

---

## Available Pose Estimation Models

| Model | Description | Documentation |
|--------|-------------|---------------|
| RTMW (MMPose) | Whole-body pose estimation using RTMW | [RTMW README](\pose_estimation_docs\rtmw_detection.md) |
| OpenPose | OpenPose-based human pose estimation | [OpenPose README](\pose_estimation_docs\openpose.md) |

---

## Choosing a Pose Estimation Model

Each pose estimation framework differs in terms of:

- Installation requirements
- Model checkpoints
- Configuration files
- Runtime dependencies
- Output formats
- Supported keypoint schemas

Please refer to the model-specific documentation before running any pipeline.

---

## Output Compatibility

Although each detector has its own implementation, all supported models are converted into the project's standardized pose estimation format.

This allows downstream modules, such as:

- Keypoint filtering
- Evaluation
- Gait analysis
- Visualization

to operate independently of the underlying detector.

---

## Directory Structure

```
README.md                          # Project root

pose_estimation_codes/
├── README.md                      # Entry point for all pose estimation documentation
│
├── pose_estimation_docs/
│   ├── rtmw.md                    # RTMW documentation
│   ├── openpose.md                # OpenPose documentation

```

---

## Model Documentation

### RTMW (MMPose)

Documentation includes:

- Installation
- Dependencies
- Model checkpoints
- Configuration files
- Running inference
- Output structure
- Generated files

See:

```
pose_estimation_docs/rtmw.md
```

---

### OpenPose

Documentation will include:

- Building OpenPose
- Downloading models
- Running inference
- Output conversion
- Integration with the pipeline

See:

```
pose_estimation_docs/openpose.md
```

---

## Adding New Pose Estimation Models

New pose estimation backends should provide their own documentation covering:

- Installation
- Dependencies
- Configuration
- Execution
- Output format
- Known limitations

This keeps implementation-specific details isolated while maintaining a consistent project structure.