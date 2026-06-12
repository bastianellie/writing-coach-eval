"""Hallucination-detection (HD) interface and method registry.

Groundedness is measured by running a *hallucination detector* over the claims
found in a Writing-Coach ``response`` / ``suggestion`` and the references those
claims cite. Different HD methods may be shipped over time; they are kept behind
the :class:`HallucinationDetector` interface and selected by name through the
registry so ``run_full_eval`` can swap implementations without changing.

Per the agreed design, each HD method owns its *whole pipeline* — claim
extraction, claim/reference pairing, and per-claim classification — inside
:meth:`HallucinationDetector.score_summary`. The base class only fixes the
contract and the return shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


# Canonical per-claim decision labels.
ENTAILMENT = "Entailment"
CONTRADICTION = "Contradiction"
UNVERIFIABLE = "Unverifiable"


@dataclass
class SummaryScore:
    """Result of scoring a single response / suggestion ("summary").

    Attributes
    ----------
    claim_distribution_per_summary:
        Percentage of cited claims in this summary that fall into each label,
        e.g. ``{"entailment(%)": 90.0, "unverifiable(%)": 10.0,
        "contradiction(%)": 0.0}``. This is the dict consumed downstream by the
        grounded-ratio aggregation.
    claims:
        Per-claim records: ``{"claim", "claim_decision", "reasoning",
        "reference"}`` — one entry per cited claim.
    n_cited_claims:
        Number of cited claims scored (``len(claims)``).
    """

    claim_distribution_per_summary: dict[str, float]
    claims: list[dict] = field(default_factory=list)
    n_cited_claims: int = 0


class HallucinationDetector(ABC):
    """Interface every HD method must implement."""

    #: Human-readable method name; subclasses set this and it is the key used
    #: to select the method via the registry / ``--hd-method`` CLI argument.
    name: str = "base"

    @abstractmethod
    def score_summary(self, text: str, references: list[dict]) -> SummaryScore:
        """Score a single response/suggestion against its structured references.

        Implementations take the raw ``text`` (a response or a suggestion's
        transformed text) plus the list of structured reference dicts (each
        carrying at least a ``referenceId`` and an ``abstract``), and internally:
        extract claims, pair each claim with the reference text of the citations
        it carries, classify each pair, and return the label distribution.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry — maps method name -> factory callable
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., HallucinationDetector]] = {}


def register_detector(name: str) -> Callable[[Callable[..., HallucinationDetector]], Callable[..., HallucinationDetector]]:
    """Class/factory decorator that registers an HD method under *name*."""

    def _decorator(factory: Callable[..., HallucinationDetector]) -> Callable[..., HallucinationDetector]:
        if name in _REGISTRY:
            raise ValueError(f"HD method '{name}' is already registered")
        _REGISTRY[name] = factory
        return factory

    return _decorator


def get_detector(name: str, **kwargs) -> HallucinationDetector:
    """Instantiate the HD method registered under *name*."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown HD method '{name}'. Available: {', '.join(available_detectors()) or '(none)'}"
        )
    return _REGISTRY[name](**kwargs)


def available_detectors() -> list[str]:
    """Return the sorted list of registered HD method names."""
    return sorted(_REGISTRY)
