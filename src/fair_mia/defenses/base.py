"""Defense interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fair_mia.types import TextSample


class Defense(ABC):
    name: str

    @abstractmethod
    def apply(self, sample: TextSample) -> TextSample:
        """Return the sample after defense processing."""

