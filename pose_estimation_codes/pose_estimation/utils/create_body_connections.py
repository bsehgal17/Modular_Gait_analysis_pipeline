from __future__ import annotations

from typing import NamedTuple

from pose_estimation.enums.joint_enum import JointEnum


class JointConnection(NamedTuple):
    start: int
    end: int


JointConnections = list[JointConnection]


def build_connections(
    enum_cls: type[JointEnum],
    connection_names: list[tuple[str, str]],
) -> JointConnections:
    """
    Convert named joint pairs to index-based JointConnection objects.

    Parameters
    ----------
    enum_cls : type[JointEnum]
        The joint enum for the model (e.g. Body25JointEnum).
    connection_names : list[tuple[str, str]]
        Pairs of joint names, e.g. [("LEFT_HIP", "LEFT_KNEE"), ...].
        Pairs where either name is absent from the enum are silently skipped,
        allowing the same connection list to be reused across models.

    Returns
    -------
    JointConnections
        List of JointConnection(start, end) with resolved integer indices.

    Examples
    --------
    >>> conns = build_connections(Body25JointEnum, BODY25_CONNECTION_NAMES)
    >>> conns[0].start, conns[0].end
    (0, 1)
    """
    name_to_idx: dict[str, int] = {e.name: e.value for e in enum_cls}
    return [
        JointConnection(start=name_to_idx[a], end=name_to_idx[b])
        for a, b in connection_names
        if a in name_to_idx and b in name_to_idx
    ]
