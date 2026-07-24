# OpenPose Detection Pipeline (Docker-based)

## Overview

This document describes how to generate OpenPose **BODY_25** keypoints from a video dataset using Docker, and how to convert that raw output into the project's standard pose format for use in the downstream pipeline (`filter`, `assess`, `gait_measurement`).

Unlike RTMW, OpenPose in this project runs inside a **Docker container** rather than as a native Python step in `main.py`. The workflow is:

```
Video dataset
      │
      ▼
Docker + OpenPose  →  one JSON per frame (BODY_25 keypoints)
      │
      ▼
openpose_combined_json.py           (combine per-frame JSONs into one file per video)
      │
      ▼
openpose_standard_format_conversion.py   (convert to the project's standard pose format)
      │
      ▼
Standard pose JSON/PKL — usable by filter / assess / gait_measurement
```

> **Data format reference:** The final output of this workflow follows the same schema as the RTMW pipeline, documented in [`pose_estimation_dataclass.md`](../../docs/pose_estimation_dataclass.md).

---
## Why Docker?

Installing OpenPose natively means matching its exact CUDA, cuDNN, and library versions to whatever is already installed on the host. In practice this is fragile:

- **Version conflicts are common.** OpenPose (and most GPU-accelerated tools) are built against a specific CUDA/cuDNN version, which often doesn't match what's already installed for other projects on the same machine.
- **Changing the host's CUDA version is risky.** CUDA is a shared, system-wide dependency — downgrading or swapping it to satisfy one tool can silently break other GPU-accelerated applications that rely on the version already installed.
- **Docker avoids the conflict entirely.** A Docker image bundles the exact CUDA/cuDNN runtime OpenPose needs inside an isolated container. OpenPose runs against that bundled runtime instead of the host's, so it can be installed and run without touching host-level drivers or existing CUDA setups — and removed cleanly afterward with no leftover system changes.
- **Reproducibility.** The same image behaves identically across machines, so setup isn't re-solved by every new user who tries to build OpenPose from source on their own system.

---

## Prerequisites

- Docker installed on the host.
- NVIDIA GPU + NVIDIA Container Toolkit installed, so Docker can be run with `--gpus all`.
- Your user account able to run `docker` commands (see [Docker Permissions](#docker-permissions) below).

---

## Docker Permissions

If `docker` commands fail with a `permission denied` error on `/var/run/docker.sock`, the CLI cannot talk to the Docker engine as your current user.

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply the group change immediately, without a full logout/login
newgrp docker
```

---

## Choosing a Docker Image (GPU Compatibility)

Not every published OpenPose image supports every GPU. If you see:

```
no kernel image is available for execution on the device   (CUDA Error 48)
```

it means the image was compiled for older GPU architectures that don't cover your card's Compute Capability.

| Image | Result |
|---|---|
| `cwaffles/openpose` | Failed on modern NVIDIA GPUs — compiled for older architectures only. |
| `stanfordnmbl/openpose-gpu` | **Works** — supports a broader range of Compute Capabilities. |

This project uses **`stanfordnmbl/openpose-gpu`**.

---

## Running OpenPose on a Single Video

```bash
docker run --gpus all --rm \
  -v /path/to/input_folder:/input \
  -v /path/to/output_folder:/output \
  stanfordnmbl/openpose-gpu \
  ./build/examples/openpose/openpose.bin \
  --video /input/video_name.mov \
  --write_json /output/ \
  --display 0 \
  --render_pose 0
```

Key points:

- `--gpus all` gives the container access to the GPU.
- `--rm` removes the container after it exits (the container itself is disposable — only the mapped volumes persist data).
- `-v host_path:container_path` maps a folder on the host into the container, so JSON output written to `/output` inside the container ends up on your physical drive at `host_path`.
- `--display 0 --render_pose 0` disables the GUI window and pose-rendering overlay, since this is a headless batch run.

**Before running:** create the output folder on the host with `mkdir -p` first — see [Output Folder Ownership](#output-folder-ownership) below for why.

---

## Batch Processing Nested Folders

Datasets are usually organized in nested subfolders. The following script recursively finds every `.mov` file under a root folder, mirrors that folder structure under the output root, and runs OpenPose on each video:

```bash
find /storage/Projects/Gaitly/bsehgal/cropped_uvic_pass_videos/AM_02_12_24_10/ -name "*.mov" | while read v; do
  rel_path=$(dirname "${v#/storage/Projects/Gaitly/bsehgal/cropped_uvic_pass_videos/AM_02_12_24_10/}")
  vid_name=$(basename "$v" .mov)
  out_dir="/storage/Projects/Gaitly/bsehgal/OpenPose_json/$rel_path/$vid_name"
  mkdir -p "$out_dir"
  docker run --gpus all --rm \
    -v "$(dirname "$v")":/input \
    -v "$out_dir":/output \
    stanfordnmbl/openpose-gpu \
    ./build/examples/openpose/openpose.bin \
    --video "/input/$(basename "$v")" \
    --write_json /output/ \
    --display 0 \
    --render_pose 0
done
```

What it does:

1. `find ... -name "*.mov"` recursively lists every video under the dataset root.
2. For each video, `rel_path` reconstructs its subfolder path relative to the dataset root, so the output mirrors the input's structure.
3. `mkdir -p "$out_dir"` creates the output folder **on the host, as the host user**, before Docker ever touches it — see below.
4. The same single-video `docker run` command from above is executed per video, with `/input` and `/output` mapped to that video's folder and its matching output folder.

Adjust the root input path, `.mov` extension (if your dataset uses `.mp4`/`.avi`/etc.), and output root to match your dataset.

---

## Output Folder Ownership

Docker containers run as `root` by default, so **any folder Docker creates on the host is owned by `root`** — which your normal user then can't modify or delete.

Two ways this is handled here:

- **Prevention (recommended):** always `mkdir -p` the output folder on the host *before* running the container (as done in the batch script above). Since the folder already exists and is owned by your user, Docker writes into it without changing ownership.
- **Recovery:** if a folder does end up root-owned, reclaim it with:
  ```bash
  sudo chown -R $USER:$USER /path/to/folder
  ```

---

## Output Format (Per-Frame JSON — BODY_25)

OpenPose writes **one JSON file per frame** to the mapped output folder. Each file contains 2D keypoints for the **BODY_25** model: 25 body parts (e.g. Neck, Hip, Ankle, ...), each with `(X, Y, Confidence)`.

> `"Empty frame detected"` warnings at the end of a video are **non-critical** — they're expected as OpenPose reaches the end of the stream, and don't indicate a failed run.

---

## Converting to the Project's Standard Format

The per-frame JSONs from OpenPose are not yet in the format the rest of the pipeline (`filter`, `assess`, `gait_measurement`) expects. Two conversion scripts handle this, in order:

### Step 1 — Combine per-frame JSONs into one file per video

```bash
python pose_estimation_codes/pose_estimation/utils/openpose_combined_json.py
```

Merges the many per-frame JSON files produced for a video into a single combined JSON for that video.

### Step 2 — Convert to the standard pose format

```bash
python pose_estimation_codes/pose_estimation/utils/openpose_standard_format_conversion.py
```

Converts the combined per-video JSON into the project's standardized pose representation (matching what the RTMW pipeline produces), described in [`pose_estimation_dataclass.md`](../../docs/pose_estimation_dataclass.md), so it can be consumed by `filter`, `assess`, and `gait_measurement`.

> **Note:** these two scripts are plain Python and use this project's dependencies — activate the project's conda environment first (see [Environment Setup](#environment-setup)) before running them.

### Environment Setup

```bash
conda create -n pose_estimation python=3.9 -y
conda activate pose_estimation
pip install -r pose_estimation_codes/requirements.txt
```

---

## Troubleshooting

**`permission denied` on `/var/run/docker.sock`**
Your user isn't in the `docker` group yet — see [Docker Permissions](#docker-permissions).

**`no kernel image is available for execution on the device` (CUDA Error 48)**
The Docker image doesn't support your GPU's architecture. Switch to `stanfordnmbl/openpose-gpu` (see [Choosing a Docker Image](#choosing-a-docker-image-gpu-compatibility)).

**Output folders/files you can't modify or delete**
They were created by Docker running as root. Either `mkdir -p` the output folder yourself before running Docker next time, or reclaim existing folders with `sudo chown -R $USER:$USER <folder>` — see [Output Folder Ownership](#output-folder-ownership).

**`"Empty frame detected"` at the end of a run**
Expected and non-critical — the run has still completed successfully.

---

## Typical Workflow

1. Ensure Docker + NVIDIA Container Toolkit are installed and your user is in the `docker` group.
2. Run OpenPose via Docker on your dataset — single video ([Running OpenPose on a Single Video](#running-openpose-on-a-single-video)) or full dataset ([Batch Processing Nested Folders](#batch-processing-nested-folders)) — using the `stanfordnmbl/openpose-gpu` image.
3. Confirm you get one BODY_25 JSON per frame per video.
4. Activate the project's conda environment and combine per-frame JSONs into one file per video with `openpose_combined_json.py`.
5. Convert the combined JSON into the project's standard pose format with `openpose_standard_format_conversion.py`.
