from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from filtering_scripts.filter_enums import FilterType


# =========================================================
# FILTER INTERFACE
# =========================================================


class SkelPointsFilter(ABC):
    """
    Abstract base for all skeleton point filters.
    """

    @abstractmethod
    def filter_data(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: shape (num_frames, 2)

        Returns:
            filtered array of same shape
        """
        ...


# =========================================================
# BASE CONFIG
# =========================================================


class FilterConfig(BaseModel):
    """
    Base config for all filters.

    Each subclass should implement:
        create_filter(self, fps) -> SkelPointsFilter
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    filter_type: FilterType

    def to_processing_step(self) -> "FilteringStep":
        return FilteringStep(
            step_name=self.filter_type,
            params=self.model_dump(exclude={"filter_type"}),
        )

    def create_filter(self, fps: float) -> SkelPointsFilter:
        """
        Override in subclasses to construct the actual filter.
        """
        raise NotImplementedError("Each filter config must implement create_filter()")

    def build(self, fps: float) -> SkelPointsFilter:
        """
        Instantiate filter from config.
        """
        return self.create_filter(fps)


# =========================================================
# METADATA
# =========================================================


class FilteringStep(BaseModel):
    """
    Immutable record of one filtering operation.
    """

    model_config = ConfigDict(frozen=True)

    step_name: FilterType
    params: dict[str, Any]


# =========================================================
# FACTORY (optional convenience layer)
# =========================================================


class FilterFactory:
    """
    Optional factory if you still want centralized creation.
    """

    @staticmethod
    def create(config: FilterConfig, fps: float) -> SkelPointsFilter:
        return config.create_filter(fps)
