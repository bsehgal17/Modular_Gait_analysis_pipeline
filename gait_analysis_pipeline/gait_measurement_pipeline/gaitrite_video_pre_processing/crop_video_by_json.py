# ----- PyAV VIDEO CROPPING BASED ON JSON TIMESTAMPS ----- #
from pathlib import Path
import json
import re
import av  # PyAV library for video reading/writing
import numpy as np

# -------- CONFIGURATION --------
VIDEO_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\video_files - Copy"
)
JSON_ROOT = Path(r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\json_files")
OUTPUT_ROOT = Path(
    r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\cropped_uvic_pass_videos"
)

MIN_DURATION = 3.0  # minimum clip duration in seconds

# Ensure output directory exists
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# -------- Utility: Extract pass number from filename --------
def get_pass_number(path: Path) -> int:
    """
    Extract numeric pass number from a filename for sorting.
    Example: 'pass_12.json' → 12
    If no pass number found, return a large number to sort last.
    """
    match = re.search(r"pass[_-]?(\d+)", path.name, re.IGNORECASE)
    return int(match.group(1)) if match else 999999


# -------- Utility: Extract timestamps from JSON --------
def extract_timestamps(json_file: Path):
    """
    Reads the JSON file and extracts assessment timestamps.
    Assumes nested dictionary structure:
    data["UVIC"]["UVIC_VALIDATION"]["UVIC_VALIDATION_LAT"]["assessment_timestamps"]
    """
    data = json.loads(json_file.read_text())
    return data["UVIC"]["UVIC_VALIDATION"]["UVIC_VALIDATION_LAT"][
        "assessment_timestamps"
    ]


# -------- Core function: Crop video --------
def crop_video(input_video: str, start_sec: float, end_sec: float, output_path: str):
    """
    Crop a video using PyAV between start_sec and end_sec.
    Handles rotation, re-encoding, and timestamp alignment.
    """

    container = av.open(input_video)
    stream = container.streams.video[0]

    # Create output container
    out = av.open(output_path, mode="w", format="mov")

    # Add output video stream (re-encode)
    out_stream = out.add_stream("libx265", rate=stream.average_rate)
    out_stream.width = stream.width
    out_stream.height = stream.height
    out_stream.pix_fmt = "yuv420p10le"
    out_stream.color_primaries = stream.color_primaries
    out_stream.color_trc = stream.color_trc
    out_stream.colorspace = stream.colorspace
    out_stream.color_range = stream.color_range
    out_stream.options = {"crf": "10", "preset": "slow"}

    # Seek to closest keyframe before start
    container.seek(int(start_sec * av.time_base))

    first_pts = None
    break_flag = False

    for packet in container.demux(stream):
        for frame in packet.decode():
            if frame.time is None:
                continue

            # Skip frames before start time
            if frame.time < start_sec:
                continue

            # Stop after end time
            if frame.time > end_sec:
                break_flag = True
                break

            # Adjust timestamps so output starts at 0
            if first_pts is None:
                first_pts = frame.pts
            frame.pts -= first_pts
            frame.dts = frame.pts

            # Encode and mux
            for pkt in out_stream.encode(frame):
                out.mux(pkt)

        if break_flag:
            break

    # Flush encoder
    for packet in out_stream.encode():
        out.mux(packet)

    # Close containers
    out.close()
    container.close()


# -------- MAIN PIPELINE --------
for subject_folder in VIDEO_ROOT.iterdir():
    if not subject_folder.is_dir():
        continue

    subject_name = subject_folder.name
    print(f"\n====================")
    print(f"Processing SUBJECT: {subject_name}")

    # Locate Lat video folder
    lat_folder = subject_folder / "Lat"
    if not lat_folder.exists():
        print("No Lat folder")
        continue

    video_files = list(lat_folder.glob("*.MOV"))
    if not video_files:
        print("No video found")
        continue

    video_path = video_files[0]
    print(f"Video: {video_path.name}")

    # JSON folder for this subject
    json_folder = JSON_ROOT / subject_name
    if not json_folder.exists():
        print("No JSON folder")
        continue

    # Create output folder for cropped clips
    out_subject_folder = OUTPUT_ROOT / subject_name
    out_subject_folder.mkdir(parents=True, exist_ok=True)

    # Sort JSON files by pass number
    json_files = sorted(json_folder.rglob("*.json"), key=get_pass_number)
    pass_counter = 1

    for json_file in json_files:
        print(f"JSON: {json_file.name}")

        try:
            timestamps = extract_timestamps(json_file)
        except Exception as e:
            print(f"Missing timestamps: {e}")
            continue

        # Process timestamps in pairs (start, end)
        for i in range(0, len(timestamps), 2):
            start = float(timestamps[i])
            end = float(timestamps[i + 1])
            duration = end - start

            if duration < MIN_DURATION:
                print(f"Skipping short clip {start:.2f}-{end:.2f}")
                continue

            out_name = f"{subject_name}_pass_{pass_counter}.MOV"
            out_path = out_subject_folder / out_name
            pass_counter += 1

            print(f"  Cropping {start:.2f} → {end:.2f} ({duration:.2f}s)")
            crop_video(str(video_path), start, end, str(out_path))

    print(f"Done subject {subject_name}")

print("\nALL DONE")

# # ---------------------------------------FFMPEG VIDEO CROPPING BASED ON JSON TIMESTAMPS---------------------------------------
# from pathlib import Path
# import json
# import subprocess
# import re

# # -------- CONFIG --------
# VIDEO_ROOT = Path(r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\Video_files")
# JSON_ROOT = Path(r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\json_files")
# OUTPUT_ROOT = Path(
#     r"C:\Users\BhavyaSehgal\Downloads\Video_pre_preocessing\cropped_pass_videos"
# )

# MIN_DURATION = 3.0  # seconds
# # ------------------------

# OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# # -------- Helper: Extract pass number from JSON filename --------
# def get_pass_number(path: Path):
#     """
#     Extract number after 'pass' in filename.
#     Example: Pass_12_trial.json -> 12
#     If not found, return large number so it sorts last.
#     """
#     match = re.search(r"pass\s*[_-]?\s*(\d+)", path.name, re.IGNORECASE)
#     return int(match.group(1)) if match else 999999


# # Loop over subject folders
# for subject_folder in VIDEO_ROOT.iterdir():
#     if not subject_folder.is_dir():
#         continue

#     subject_name = subject_folder.name  # Maps to JSON folder name
#     print("\n============================")
#     print(f"Processing SUBJECT: {subject_name}")

#     # Find Lat video folder
#     lat_folder = subject_folder / "Lat"
#     if not lat_folder.exists():
#         print(f"No Lat folder for {subject_name}")
#         continue

#     # Find video inside Lat folder (first .MOV found)
#     video_files = list(lat_folder.glob("*.MOV"))
#     if not video_files:
#         print(f"No video found in {lat_folder}")
#         continue

#     video_path = video_files[0]  # take first video
#     print(f"Using video: {video_path.name}")

#     # Find matching JSON folder
#     json_folder = JSON_ROOT / subject_name
#     if not json_folder.exists():
#         print(f"No JSON folder for {subject_name}")
#         continue

#     print(f"JSON folder: {json_folder}")

#     # Output folder per subject
#     output_folder = OUTPUT_ROOT / subject_name
#     output_folder.mkdir(parents=True, exist_ok=True)

#     pass_counter = 1

#     # ---- Sort JSON files by pass number ----
#     json_files = sorted(json_folder.rglob("*.json"), key=get_pass_number)

#     for json_file in json_files:
#         print(f"Processing JSON: {json_file.name}")

#         data = json.loads(json_file.read_text())

#         try:
#             timestamps = data["UVIC"]["UVIC_VALIDATION"]["UVIC_VALIDATION_LAT"][
#                 "assessment_timestamps"
#             ]
#         except KeyError:
#             print(f"Missing timestamps in {json_file.name}")
#             continue

#         # Process timestamp pairs
#         for i in range(0, len(timestamps), 2):
#             start = float(timestamps[i])
#             end = float(timestamps[i + 1])
#             duration = end - start

#             # Skip short clips
#             if duration < MIN_DURATION:
#                 print(
#                     f"Skipping {subject_name} [{start:.2f}-{end:.2f}] duration={duration:.2f}s"
#                 )
#                 continue

#             # Output name
#             out_name = f"{subject_name}_pass_{pass_counter}.mov"
#             out_path = output_folder / out_name
#             pass_counter += 1

#             cmd = [
#                 "ffmpeg",
#                 "-y",
#                 "-ss",
#                 str(start),
#                 "-to",
#                 str(end),
#                 "-i",
#                 str(video_path),
#                 "-c",
#                 "copy",  # copy video and audio streams
#                 str(out_path),
#             ]

#             subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print(f"Saved {out_name} (duration={duration:.2f}s)")
