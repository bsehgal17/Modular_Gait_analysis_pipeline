# Modular Gait Pipeline

A modular, model-agnostic gait analysis pipeline for extracting spatiotemporal gait metrics from video-based pose estimation data.

## Repository Structure

```
Modular_Gait_Pipeline/
├── README.md
├── Gait_Analysis_pipeline/
│   ├── README.md
│   └── requirements.txt
└── Pose_estimation_codes/
    ├── README.md
    └── requirements.txt
```

- **`Gait_Analysis_pipeline/`** — the core pipeline that computes gait metrics from keypoint data. See its [README](https://github.com/bsehgal17/Modular_Gait_pipeline/tree/main/Gait_Analysis_Pipeline) for setup and usage.
- **`Pose_estimation_codes/`** — generates body keypoints from video using pose estimation models (currently RTMW and OpenPose), producing output as pose estimation JSON files in the pipeline's standard dataclass format. Code for converting body keypoints into a standardized dataclass format to facilitate the integration and evaluation of additional pose estimation models will be added over time. See its [README](https://github.com/bsehgal17/Modular_Gait_pipeline/tree/main/Pose_estimation_codes) for setup and usage.

Each folder is self-contained with its own README and `requirements.txt`.

## Getting Started

Each folder has its own dependencies and should be set up in its **own separate virtual environment** to avoid conflicts.

```bash
git clone https://github.com/<your-username>/Modular_Gait_Pipeline.git
cd Modular_Gait_Pipeline
```

### Gait Analysis Pipeline

```bash
cd Gait_Analysis_pipeline
python -m venv gait_env
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Pose Estimation

```bash
cd Pose_estimation_codes
python -m venv pose_env
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Refer to the README inside each folder for input/output formats, configuration, and example usage.

## Contact

For questions, please open an issue on this repository or contact bsehgal1@ualberta.ca