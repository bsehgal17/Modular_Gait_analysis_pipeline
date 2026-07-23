import numpy as np


def compute_joint_speed(positions):
    """
    Compute speed of a 2D joint trajectory in pixels/frame
    using forward difference.

    Args:
        positions: (T, 2) array of (x, y) positions per frame

    Returns:
        speed: (T,) array of speed magnitudes in pixels/frame
    """
    dx = np.diff(positions[:, 0])
    dy = np.diff(positions[:, 1])

    speed = np.sqrt(dx**2 + dy**2)
    speed = np.append(speed, speed[-1])

    return speed
