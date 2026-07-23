import numpy as np


def compute_ap_displacement(joint, mid_hip):
    """
    Compute anterior-posterior displacement relative to pelvis.
    """
    return joint[:, 0] - mid_hip[:, 0]
