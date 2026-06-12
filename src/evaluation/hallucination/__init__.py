"""Hallucination-detection (groundedness) package.

Importing the package registers the bundled HD methods so they are resolvable
via :func:`get_detector` / ``available_detectors``.
"""

from src.evaluation.hallucination.base import (
    HallucinationDetector,
    SummaryScore,
    available_detectors,
    get_detector,
    register_detector,
)

# Import concrete detectors for their registration side effects.
from src.evaluation.hallucination import trustcard_detector  # noqa: F401

DEFAULT_HD_METHOD = "trustcard"

__all__ = [
    "HallucinationDetector",
    "SummaryScore",
    "available_detectors",
    "get_detector",
    "register_detector",
    "DEFAULT_HD_METHOD",
]
