"""
migrate_pose_json.py

Converts the old pose estimation JSON format to the schema expected by
PoseEstimationResult.load_json().

Recursively scans INPUT_ROOT for all .json files, mirrors the folder
structure under OUTPUT_ROOT, and writes converted files there.

Usage (batch):
    Set INPUT_ROOT and OUTPUT_ROOT below, then run:
        python migrate_pose_json.py
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pose_estimation.enums.joint_enum import PredJointsCOCOWholebody


# ---------------------------------------------------------------------------
# Paths  —  edit these
# ---------------------------------------------------------------------------

INPUT_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\jsons\rtmw_json"
)

OUTPUT_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\jsons\rtmw_json_converted"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def json_enum_serializer(obj):
    """
    Allows json.dump() to serialize Enum objects.

    Example:
        <PredJointsCOCOWholebody.LEFT_HEEL: 19>
            -> "LEFT_HEEL"
    """
    if isinstance(obj, Enum):
        return obj.name

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def convert_detection(det: dict) -> dict:
    x1, y1, x2, y2 = det["bbox"]

    return {
        "frame_idx": det["frame_idx"],
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": det["score"],
        "label": "PERSON",  # ObjectLabel.PERSON
    }


def convert_pose(pose: dict) -> dict:
    """
    Converts pose format.

    Keeps known joints as enum objects:
        <PredJointsCOCOWholebody.LEFT_HEEL: 19>

    Unknown joints (face / hands) remain integers.
    """

    # keypoints and keypoints_visible are batched:
    # [[kp0, kp1, ...]]
    kps = pose["keypoints"][0]
    vis = pose["keypoints_visible"][0]

    joints = []

    for j, kp in enumerate(kps):
        # Convert known joints to enum objects
        try:
            joint_name = PredJointsCOCOWholebody(j)

        # Face / hand landmarks remain raw ints
        except ValueError:
            joint_name = j

        joints.append({
            "name": joint_name,
            "keypoint": {
                "x": kp[0],
                "y": kp[1],
                "z": kp[2] if len(kp) > 2 else 0.0,
                "confidence": None,
            },
            "visibility": (vis[j] if j < len(vis) else 1.0),
        })

    x1, y1, x2, y2 = pose["bbox"]

    return {
        "frame_idx": pose["frame_idx"],
        "joints": joints,
        "bbox": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "bbox_score": float(pose["bbox_scores"][0]),
        "model": pose.get("pose_model", "RTMW").upper(),
    }


def convert_person(person: dict) -> dict:
    return {
        "person_id": person["person_id"],
        "detections": [convert_detection(d) for d in person["detections"]],
        "poses": [convert_pose(p) for p in person["poses"]],
    }


# ---------------------------------------------------------------------------
# Single-file migration
# ---------------------------------------------------------------------------


def migrate(
    input_path: str | Path,
    output_path: str | Path,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        old = json.load(f)

    new = {
        "video_name": old.get("video_name", input_path.stem),
        "fps": old.get("fps", None),
        "width": old.get("width", None),
        "height": old.get("height", None),
        "persons": [convert_person(p) for p in old.get("persons", [])],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2, default=json_enum_serializer)

    n_persons = len(new["persons"])
    print(f"  Saved : {output_path}")
    print(f"          {n_persons} person(s)", end="")

    for p in new["persons"]:
        print(
            f"  |  person_id={p['person_id']}: "
            f"{len(p['detections'])} detections, "
            f"{len(p['poses'])} poses",
            end="",
        )
    print()


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def migrate_all(input_root: Path, output_root: Path):
    json_files = sorted(input_root.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found under {input_root}")
        return

    print(f"Found {len(json_files)} JSON file(s) under {input_root}\n")

    ok = 0
    failed = 0

    for input_path in json_files:
        # Mirror the subfolder structure inside OUTPUT_ROOT
        relative = input_path.relative_to(input_root)
        output_path = output_root / relative

        print(f"Converting: {relative}")

        try:
            migrate(input_path, output_path)
            ok += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\nDone.  {ok} converted, {failed} failed.")


if __name__ == "__main__":
    migrate_all(INPUT_ROOT, OUTPUT_ROOT)
