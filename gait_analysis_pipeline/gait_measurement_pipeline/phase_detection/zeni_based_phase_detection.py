from typing import Dict
import numpy as np


def detect_stance_swing_zeni(
    trajectories: Dict[str, np.ndarray], fps: float, plot: bool = False
) -> Dict[str, np.ndarray]:
    """
    Detect stance and swing phases using Zeni et al.'s method
    based on relative heel-to-pelvis positions.

    Args:
        trajectories: Dictionary containing joint trajectories with keys:
                      - "frame_ids": array of frame numbers
                      - "left_heel": array of shape (N, 2) [x, y]
                      - "right_heel": array of shape (N, 2) [x, y]
                      - "mid_hip": array of shape (N, 2) [x, y]
        fps: Frames per second of the motion capture/video.
        plot: If True, generate a plot of relative positions and detected events.

    Returns:
        Dictionary containing:
            - "time": array of time points corresponding to each frame
            - "left_stance": boolean array, True if left foot is in stance
            - "left_swing": boolean array, True if left foot is in swing
            - "right_stance": boolean array, True if right foot is in stance
            - "right_swing": boolean array, True if right foot is in swing
    """

    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    # -------------------------------
    # Convert frame IDs to time array
    # -------------------------------
    frame_ids = np.array(trajectories["frame_ids"])
    time = frame_ids / fps

    # -------------------------------
    # Extract joints
    # -------------------------------
    left_heel = np.array(trajectories["left_heel"])
    right_heel = np.array(trajectories["right_heel"])
    pelvis = np.array(trajectories["mid_hip"])  # important reference joint

    # -------------------------------
    # Relative position (heel to pelvis)
    # -------------------------------
    left_rel = left_heel[:, 0] - pelvis[:, 0]  # X-axis difference
    right_rel = right_heel[:, 0] - pelvis[:, 0]

    # -------------------------------
    # Detect Heel Strike (HS) and Toe Off (TO) using peaks
    # -------------------------------
    # HS = local maxima in relative position
    left_hs, _ = find_peaks(left_rel, distance=fps // 2, prominence=0.9)
    right_hs, _ = find_peaks(right_rel, distance=fps // 2, prominence=0.9)

    # TO = local minima in relative position
    left_to, _ = find_peaks(-left_rel, distance=fps // 2)
    right_to, _ = find_peaks(-right_rel, distance=fps // 2)

    # -------------------------------
    # Convert events to phase masks
    # -------------------------------
    def build_phase_mask(length: int, hs: np.ndarray, to: np.ndarray) -> np.ndarray:
        """
        Build boolean array marking stance phase between HS → TO.

        Args:
            length: total number of frames
            hs: frame indices of Heel Strikes
            to: frame indices of Toe Offs

        Returns:
            Boolean array, True for stance, False for swing
        """
        stance = np.zeros(length, dtype=bool)
        events = np.sort(np.concatenate([hs, to]))

        for i in range(len(events) - 1):
            start = events[i]
            end = events[i + 1]
            if events[i] in hs:
                stance[start:end] = True  # Stance between HS and TO

        return stance

    left_stance = build_phase_mask(len(time), left_hs, left_to)
    right_stance = build_phase_mask(len(time), right_hs, right_to)

    left_swing = ~left_stance
    right_swing = ~right_stance

    # -------------------------------
    # Optional plotting for debugging
    # -------------------------------
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(time, left_rel, label="Left Rel Pos")
        plt.plot(time, right_rel, label="Right Rel Pos")
        plt.scatter(time[left_hs], left_rel[left_hs], label="L HS")
        plt.scatter(time[left_to], left_rel[left_to], label="L TO")
        plt.title("Zeni-based Phase Detection")
        plt.xlabel("Time [s]")
        plt.ylabel("Relative X Position")
        plt.legend()
        plt.grid()
        plt.show()

    # -------------------------------
    # Return all phases
    # -------------------------------
    return {
        "time": time,
        "left_stance": left_stance,
        "left_swing": left_swing,
        "right_stance": right_stance,
        "right_swing": right_swing,
    }
