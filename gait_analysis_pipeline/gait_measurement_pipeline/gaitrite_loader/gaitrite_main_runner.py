from pathlib import Path
import json
import re
from typing import List

from gait_measurement_pipeline.gaitrite_loader.gaitrite_file_loader import (
    GaitRiteFileLoader,
)
from gait_measurement_pipeline.gait_dataclasses.participant_gait_results_dataclass import (
    ParticipantData,
    Trial,
    PassResult,
)

# ---------------------------
# Helpers
# ---------------------------


def extract_participant_id(path: Path) -> str:
    """
    Extracts participant ID from the folder name.
    """
    return path.name


def extract_trial_number(path: Path) -> int:
    """
    Extracts trial number from filename using regex.
    Example: 'File2.xlsx' → 2
    Defaults to 1 if no number found.
    """
    match = re.search(r"File(\d+)", path.stem)
    return int(match.group(1)) if match else 1


# ---------------------------
# Main folder runner
# ---------------------------


def run_gaitrite_folder(
    main_folder: Path, output_folder: Path
) -> List[ParticipantData]:
    """
    Processes an entire GaitRite dataset folder.

    Steps:
    1. Loop over participant folders
    2. Sort Excel files (trials) by trial number
    3. Load each Excel file using GaitRiteFileLoader
    4. Convert results into structured ParticipantData/Trial/PassResult objects
    5. Save each participant as JSON
    """

    main_folder = Path(main_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_results: List[ParticipantData] = []

    # ---------------------------
    # Loop over participants
    # ---------------------------
    for participant_folder in main_folder.iterdir():
        if not participant_folder.is_dir():
            continue

        pid = extract_participant_id(participant_folder)
        print(f"\nProcessing Participant: {pid}")

        # Get Excel files for participant, sorted by trial number
        excel_files = sorted(
            participant_folder.glob("*.xlsx"),
            key=lambda f: extract_trial_number(f),
        )

        if not excel_files:
            print(f"  No Excel files found in {participant_folder}")
            continue

        # Initialize participant object
        participant_data = ParticipantData(participant_id=pid, trials=[])
        pass_counter = 1  # global pass counter across all trials

        # ---------------------------
        # Loop over trials (Excel files)
        # ---------------------------
        for trial_file in excel_files:
            trial_num = extract_trial_number(trial_file)
            print(f"  Trial {trial_num}: {trial_file.name}")

            # Load GaitAnalysisResults for this trial
            file_loader = GaitRiteFileLoader(
                trial_file,
                start_pass=pass_counter,  # maintain global pass continuity
            )

            trial_results = file_loader.load()  # List[GaitAnalysisResult]

            # Create trial object
            trial_obj = Trial(
                trial_id=trial_num,
                num_passes=len(trial_results),
                passes=[],
            )

            # Populate PassResult objects for each pass
            for r in trial_results:
                trial_obj.passes.append(
                    PassResult(
                        pass_id=r.metadata.file,
                        pass_number=r.metadata.pass_number,  # already sequential
                        result=r,
                    )
                )

            # Increment global pass counter
            pass_counter += len(trial_results)

            # Add trial to participant
            participant_data.trials.append(trial_obj)

        # ---------------------------
        # Save participant data as JSON
        # ---------------------------
        participant_file = output_folder / f"{pid.upper()}.json"
        with open(participant_file, "w") as f:
            json.dump(participant_data.model_dump(), f, indent=2)

        print(f"Saved {participant_file}")
        all_results.append(participant_data)

    print("\nAll participants processed!")
    return all_results
