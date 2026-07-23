from scipy.signal import find_peaks
import numpy as np


def detect_peaks(
    signal,
    fps,
    prominence_ratio,
    min_step_time=None,  # seconds
    height=None,
    width=None,
):
    """
    Generic peak detector for gait signals.

    Args:
        signal: 1D array
        fps: frames per second
        prominence_ratio: float OR (min_ratio, max_ratio)
        min_step_time: minimum time between steps (sec)
        height: optional absolute height threshold
        width: optional peak width
    """

    # -------------------------
    # Compute IQR
    # -------------------------
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25

    # -------------------------
    # Handle prominence input
    # -------------------------
    if isinstance(prominence_ratio, (tuple, list, np.ndarray)):
        lower_limit = prominence_ratio[0] * iqr
        upper_limit = prominence_ratio[1] * iqr
        prominence = (lower_limit, upper_limit)

    elif isinstance(prominence_ratio, (float, int)):
        lower_limit = prominence_ratio * iqr
        prominence = lower_limit  # only minimum

    else:
        raise ValueError("prominence_ratio must be float or tuple/list")

    # -------------------------
    # Optional min distance
    # -------------------------
    distance = None
    if min_step_time is not None:
        distance = int(min_step_time * fps)

    # -------------------------
    # Peak detection
    # -------------------------
    peaks, properties = find_peaks(
        signal,
        prominence=prominence,
        height=height,
        width=width,
        distance=distance,
    )

    return peaks, properties
