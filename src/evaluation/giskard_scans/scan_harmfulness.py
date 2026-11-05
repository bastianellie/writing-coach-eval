"""Run Giskard harmfulness scan for Writing Coach."""

from __future__ import annotations

from typing import Optional

from giskard.core.core import SupportedModelTypes
from giskard.scanner.report import ScanReport

from src.evaluation.giskard_scans.common import run_scan_for_detector, validate_required_env_vars

DETECTOR = "harmfulness"


def run_harmfulness_scan(
    dataset_csv: Optional[str] = None,
    n_samples: int = 20,
    seed: int = 42,
    n_adversarial_samples: Optional[int] = None,
    n_requirements: Optional[int] = None,
    wc_version: str = "v3",
    wc_app_src: Optional[str] = None,
    model_type: SupportedModelTypes = "text_generation",
    persist_output: bool = False,
    output_dir: Optional[str] = None,
    workers: int = 1,
) -> ScanReport:
    """Run harmfulness detector scan and return results.

    Args:
        dataset_csv: Path to CSV dataset file.
        n_samples: Number of samples to use from dataset.
        seed: Random seed for sampling.
        n_adversarial_samples: Number of adversarial samples per detector.
        n_requirements: Number of requirements per detector.
        wc_version: Writing Coach interface version (v2 or v3).
        wc_app_src: Path to Writing Coach app root or src directory.
        model_type: Giskard model type.
        persist_output: Whether to save scan artifacts to disk.
        output_dir: Directory where artifacts are written when persist_output=True.

    Returns:
        ScanReport object containing scan results.
    """
    validate_required_env_vars(wc_app_src=wc_app_src)
    return run_scan_for_detector(
        detector=DETECTOR,
        dataset_csv=dataset_csv,
        n_samples=n_samples,
        seed=seed,
        n_adversarial_samples=n_adversarial_samples,
        n_requirements=n_requirements,
        wc_version=wc_version,
        wc_app_src=wc_app_src,
        model_type=model_type,
        persist_output=persist_output,
        output_dir=output_dir,
        workers=workers,
    )


