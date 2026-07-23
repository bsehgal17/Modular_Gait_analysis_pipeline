from pathlib import Path
from typing import Dict, Any, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt


def visualize_gait_events(
    video_path: Union[str, Path],
    events: Dict[str, np.ndarray],
    output_dir: Union[str, Path] = None,
    pause: bool = True,
) -> None:
    """
    Visualize detected gait events on video frames and optionally save frames.

    Parameters
    ----------
    video_path : str or Path
        Path to the input walking video.
    events : dict
        Dictionary containing frame indices for detected gait events.
        Keys should include: "left_heel_strike", "right_heel_strike",
        "left_toe_off", "right_toe_off".
    output_dir : str or Path, optional
        Directory to save event frames as images. If None, frames are not saved.
    pause : bool
        If True, pause at each frame until a key is pressed. Otherwise, playback automatically.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Combine events into a sorted list of (frame_index, label)
    event_frames = []
    for label_key, label_name in [
        ("left_heel_strike", "LEFT_HS"),
        ("right_heel_strike", "RIGHT_HS"),
        ("left_toe_off", "LEFT_TO"),
        ("right_toe_off", "RIGHT_TO"),
    ]:
        for frame_idx in events.get(label_key, []):
            event_frames.append((frame_idx, label_name))

    event_frames.sort(key=lambda x: x[0])

    # Loop through each event frame
    for frame_id, label in event_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        # Overlay text labels
        cv2.putText(frame, label, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"Frame: {frame_id}",
            (50, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Save frame if output directory is provided
        if output_dir is not None:
            filename = output_dir / f"{label}_frame_{frame_id}.png"
            cv2.imwrite(str(filename), frame)

        # Display frame
        cv2.imshow("Gait Event Validation", frame)
        cv2.waitKey(0 if pause else 100)

    cap.release()
    cv2.destroyAllWindows()


def visualize_gait_phases(
    video_path: Union[str, Path],
    trajectories: Any,  # object with left_toe, right_toe, etc.
    phases: Any,  # object with left_stance, right_stance boolean arrays
    step: int = 1,
    delay: float = 0.1,
) -> None:
    """
    Visualize gait phases (stance vs swing) overlaid on video frames.

    Parameters
    ----------
    video_path : str or Path
        Path to input walking video.
    trajectories : object
        Object containing joint trajectories, e.g., trajectories.left_toe.
    phases : object
        Object containing boolean stance/swing arrays: phases.left_stance, phases.right_stance.
    step : int
        Process every `step` frames to speed up visualization.
    delay : float
        Pause between frames (seconds), controls playback speed.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    left_stance = phases.left_stance
    right_stance = phases.right_stance

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(total_frames, len(left_stance))

    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(6, 4))

    frame_idx = 0
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ax.clear()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)
        ax.set_title(
            f"Frame {frame_idx} | "
            f"L: {'STANCE' if left_stance[frame_idx] else 'SWING'} | "
            f"R: {'STANCE' if right_stance[frame_idx] else 'SWING'}"
        )
        ax.axis("off")
        plt.draw()
        plt.pause(delay)

        frame_idx += 1

    cap.release()
    plt.ioff()
    plt.show()
