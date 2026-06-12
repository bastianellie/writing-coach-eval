"""Giskard scan orchestration and metric routing.

Metric name lists and the ``split_selected_metrics`` router live in
``src.constants.metrics_definitions`` (the single source of truth). This module
only owns the mapping from scan-metric names to their scan runner functions.
"""

from src.constants.metrics_definitions import SCAN_METRIC_NAMES


def _run_harmfulness_scan(*args, **kwargs):
    from src.evaluation.giskard_scans.scan_harmfulness import run_harmfulness_scan

    return run_harmfulness_scan(*args, **kwargs)


def _run_stereotypes_scan(*args, **kwargs):
    from src.evaluation.giskard_scans.scan_stereotypes import run_stereotypes_scan

    return run_stereotypes_scan(*args, **kwargs)


def _run_jailbreak_scan(*args, **kwargs):
    from src.evaluation.giskard_scans.scan_jailbreak import run_jailbreak_scan

    return run_jailbreak_scan(*args, **kwargs)


SCAN_METRIC_RUNNERS = {
    "potential_harm": _run_harmfulness_scan,
    "toxicity": _run_stereotypes_scan,
    "resilience": _run_jailbreak_scan,
}

# Keep the runner mapping aligned with the canonical scan-metric registry.
assert set(SCAN_METRIC_RUNNERS) == set(SCAN_METRIC_NAMES), (
    "SCAN_METRIC_RUNNERS keys must match SCAN_METRIC_NAMES "
    f"({sorted(SCAN_METRIC_RUNNERS)} != {sorted(SCAN_METRIC_NAMES)})"
)

