"""No-op defense for baseline runs."""

from __future__ import annotations

from fair_mia.defenses.base import Defense
from fair_mia.types import TextSample


class NoOpDefense(Defense):
    name = "noop"

    def apply(self, sample: TextSample) -> TextSample:
        return sample

