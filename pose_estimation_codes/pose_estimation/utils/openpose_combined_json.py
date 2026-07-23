from pathlib import Path
import json
import re

# -------- PATHS --------
INPUT_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\OpenPose_json_uvic"
)
OUTPUT_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\OpenPose_uvic"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# -------- Extract pass number --------
def get_pass_number(path: Path):
    match = re.search(r"pass[_-]?(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 999999


# -------- Extract frame number --------
def get_frame_number(path: Path):
    match = re.search(r"_(\d+)_keypoints\.json$", path.name)
    return int(match.group(1)) if match else -1


# -------- MAIN LOOP --------
for subject_folder in INPUT_ROOT.iterdir():
    if not subject_folder.is_dir():
        continue

    subject_name = subject_folder.name
    print(f"\n========================")
    print(f"Processing SUBJECT: {subject_name}")

    # Create output subject folder
    out_subject_folder = OUTPUT_ROOT / subject_name
    out_subject_folder.mkdir(parents=True, exist_ok=True)

    # Sort pass folders numerically
    pass_folders = sorted(
        [p for p in subject_folder.iterdir() if p.is_dir()], key=get_pass_number
    )

    for pass_folder in pass_folders:
        pass_name = pass_folder.name
        print(f"\n  Processing PASS: {pass_name}")

        # Get all frame json files sorted
        json_files = sorted(pass_folder.glob("*_keypoints.json"), key=get_frame_number)

        if not json_files:
            print("  No JSON files found")
            continue

        combined_data = {
            "subject": subject_name,
            "pass_name": pass_name.replace(".MOV", ""),
            "frames": [],
        }

        for file in json_files:
            frame_index = get_frame_number(file)

            with open(file, "r") as f:
                data = json.load(f)

            frame_data = {"frame_index": frame_index, "people": data.get("people", [])}

            combined_data["frames"].append(frame_data)

        # Save combined JSON
        output_file = out_subject_folder / f"{pass_name.replace('.MOV', '')}.json"

        with open(output_file, "w") as f:
            json.dump(combined_data, f, indent=4)

        print(f"  Saved: {output_file.name}")

print("\nALL SUBJECTS DONE ")
