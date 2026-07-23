from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class HeightSignal(BaseModel, frozen=True):
    """One height estimate per frame."""

    values: list[float]

    def __len__(self) -> int:
        return len(self.values)

    # --- add these two methods ---
    def to_signal_map(self) -> dict[str, list[float]]:
        return {"values": self.values}

    @classmethod
    def from_signal_map(cls, signals: dict[str, list[float]]) -> "HeightSignal":
        return cls(values=signals["values"])


class HeightSignalSmoother(BaseModel, ABC):
    model_config = {"frozen": True}

    @abstractmethod
    def smooth(self, signal: HeightSignal) -> HeightSignal: ...

    @abstractmethod
    def get_params(self) -> dict: ...
