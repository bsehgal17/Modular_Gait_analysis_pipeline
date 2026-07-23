from pathlib import Path
from typing import List
import json

from gait_measurement_pipeline.gait_dataclasses.gait_results_dataclass import (
    GaitAnalysisResult,
)


class JSONResultsLoader:
    """
    Load GaitAnalysisResult objects from a folder of JSON files.

    Handles two formats:
    1. A list of GaitAnalysisResult dictionaries (flattened).
    2. Participant-style JSON:
       {
         "participant_id": "...",
         "trials": [
             {
                 "trial_id": ...,
                 "passes": [
                     {"pass_id": "...", "pass_number": ..., "result": {...}}
                 ]
             }
         ]
       }

    Returns a flat list of GaitAnalysisResult objects.
    """

    def __init__(self, folder_path: Path):
        self.folder_path: Path = Path(folder_path)

    def load(self) -> List[GaitAnalysisResult]:
        results: List[GaitAnalysisResult] = []

        for json_file in self.folder_path.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for r in data:
                        results.append(GaitAnalysisResult.model_validate(r))

                else:
                    # Participant-style structure
                    participant_id = data.get("participant_id", "unknown")

                    for trial in data.get("trials", []):
                        trial_id = trial.get("trial_id")

                        for p in trial.get("passes", []):
                            res_dict = p.get("result", {})

                            # Injection: Ensure metadata exists so Pydantic doesn't complain
                            if "metadata" not in res_dict:
                                res_dict["metadata"] = {
                                    "participant_id": participant_id,
                                    "pass_number": p.get("pass_number", 1),
                                    "trial_id": trial_id,
                                    "file": str(json_file.name),
                                }

                            results.append(GaitAnalysisResult.model_validate(res_dict))
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")

        return results
