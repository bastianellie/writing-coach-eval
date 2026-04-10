"""Run Giskard jailbreak scan for Writing Coach."""

from __future__ import annotations

from typing import Optional

from giskard.core.core import SupportedModelTypes
from giskard.scanner.report import ScanReport

from src.evaluation.giskard_scans.common import run_scan_for_detector, validate_required_env_vars

DETECTOR = "jailbreak"


def run_jailbreak_scan(
    dataset_csv: Optional[str] = None,
    n_samples: int = 20,
    seed: int = 42,
    n_adversarial_samples: Optional[int] = None,
    n_requirements: Optional[int] = None,
    model_type: SupportedModelTypes = "text_generation",
) -> ScanReport:
    """Run jailbreak detector scan and return results.

    Args:
        dataset_csv: Path to CSV dataset file.
        n_samples: Number of samples to use from dataset.
        seed: Random seed for sampling.
        n_adversarial_samples: Number of adversarial samples per detector.
        n_requirements: Number of requirements per detector.
        model_type: Giskard model type.

    Returns:
        ScanReport object containing scan results.
    """
    validate_required_env_vars()
    return run_scan_for_detector(
        detector=DETECTOR,
        dataset_csv=dataset_csv,
        n_samples=n_samples,
        seed=seed,
        n_adversarial_samples=n_adversarial_samples,
        n_requirements=n_requirements,
        model_type=model_type,
        persist_output=False,
    )


