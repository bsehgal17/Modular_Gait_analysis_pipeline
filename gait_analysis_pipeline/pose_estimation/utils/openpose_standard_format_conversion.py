from pathlib import Path
import json
import numpy as np

# -------- PATHS --------
INPUT_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\OpenPose_uvic"
)
OUTPUT_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\OpenPose_uvic_datacalass_format"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.1

# OpenPose 25-keypoint joint names (index → name)
# https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html
OPENPOSE_JOINT_NAMES = [
    "NOSE",  # 0
    "NECK",  # 1
    "RIGHT_SHOULDER",  # 2
    "RIGHT_ELBOW",  # 3
    "RIGHT_WRIST",  # 4
    "LEFT_SHOULDER",  # 5
    "LEFT_ELBOW",  # 6
    "LEFT_WRIST",  # 7
    "MID_HIP",  # 8
    "RIGHT_HIP",  # 9
    "RIGHT_KNEE",  # 10
    "RIGHT_ANKLE",  # 11
    "LEFT_HIP",  # 12
    "LEFT_KNEE",  # 13
    "LEFT_ANKLE",  # 14
    "RIGHT_EYE",  # 15
    "LEFT_EYE",  # 16
    "RIGHT_EAR",  # 17
    "LEFT_EAR",  # 18
    "LEFT_TOE",  # 19  (big toe)
    "LEFT_SMALL_TOE",  # 20
    "LEFT_HEEL",  # 21
    "RIGHT_TOE",  # 22  (big toe)
    "RIGHT_SMALL_TOE",  # 23
    "RIGHT_HEEL",  # 24
]


def process_keypoints(pose_keypoints):
    """
    Convert flat OpenPose keypoints into the new standard format.

    Returns:
        joints      - list of joint dicts with name, keypoint {x,y,z,confidence}, visibility
        bbox        - dict {x1, y1, x2, y2}
        bbox_score  - float
    """
    kp = np.array(pose_keypoints).reshape(-1, 3)  # shape (N, 3): x, y, confidence

    # Build joints list
    joints = []
    for idx, (x, y, conf) in enumerate(kp):
        name = OPENPOSE_JOINT_NAMES[idx] if idx < len(OPENPOSE_JOINT_NAMES) else idx
        joints.append({
            "name": name,
            "keypoint": {
                "x": float(x),
                "y": float(y),
                "z": 0.0,
                "confidence": None,  # OpenPose doesn't provide separate confidence
            },
            "visibility": float(conf),
        })

    # Compute bbox from valid keypoints only
    valid = kp[kp[:, 2] > CONF_THRESHOLD]
    if len(valid) == 0:
        bbox = {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}
        bbox_score = 0.0
    else:
        bbox = {
            "x1": float(np.min(valid[:, 0])),
            "y1": float(np.min(valid[:, 1])),
            "x2": float(np.max(valid[:, 0])),
            "y2": float(np.max(valid[:, 1])),
        }
        bbox_score = float(np.mean(valid[:, 2]))

    return joints, bbox, bbox_score


# -------- MAIN --------
for subject_folder in INPUT_ROOT.iterdir():
    if not subject_folder.is_dir():
        continue

    subject_name = subject_folder.name
    print(f"\nProcessing SUBJECT: {subject_name}")

    out_subject_folder = OUTPUT_ROOT / subject_name
    out_subject_folder.mkdir(parents=True, exist_ok=True)

    for combined_file in subject_folder.glob("*.json"):
        print(f"  Converting: {combined_file.name}")

        with open(combined_file, "r") as f:
            data = json.load(f)

        video_name = data["pass_name"]

        detections = []
        poses = []

        for frame in data["frames"]:
            frame_idx = frame["frame_index"]

            if not frame["people"]:
                continue

            pose_keypoints = frame["people"][0]["pose_keypoints_2d"]

            joints, bbox, bbox_score = process_keypoints(pose_keypoints)

            # Detection — flat x1/y1/x2/y2 fields, label as string
            detections.append({
                "frame_idx": frame_idx,
                "x1": bbox["x1"],
                "y1": bbox["y1"],
                "x2": bbox["x2"],
                "y2": bbox["y2"],
                "score": bbox_score,
                "label": "PERSON",
            })

            # Pose — joints list, bbox as dict, scalar bbox_score, model tag
            poses.append({
                "frame_idx": frame_idx,
                "joints": joints,
                "bbox": bbox,
                "bbox_score": bbox_score,
                "model": "OpenPose",
            })

        output_data = {
            "video_name": video_name,
            "fps": None,
            "width": None,
            "height": None,
            "persons": [
                {
                    "person_id": 0,
                    "detections": detections,
                    "poses": poses,
                }
            ],
        }

        output_file = out_subject_folder / combined_file.name.replace(
            "_combined", "_pose"
        )

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"  Saved: {output_file.name}")

print("\nALL DONE")
