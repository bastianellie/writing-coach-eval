"""Unit tests for Giskard detector scan functions.

Tests the three detector-specific scan functions:
  - run_harmfulness_scan
  - run_stereotypes_scan
  - run_jailbreak_scan

with minimal input (n_adversarial_samples=2, n_requirements=2) using the
all_results_domains.csv dataset.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from giskard.scanner.report import ScanReport

from src.evaluation.giskard_scans.scan_harmfulness import run_harmfulness_scan
from src.evaluation.giskard_scans.scan_stereotypes import run_stereotypes_scan
from src.evaluation.giskard_scans.scan_jailbreak import run_jailbreak_scan


class TestGiskardDetectorScans(unittest.TestCase):
    """Test Giskard detector scan functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test dataset path and validate environment."""
        cls.dataset_csv = (
            Path(__file__).parents[2] / "final_data" / "all_results_domains.csv"
        )

        # Verify dataset exists
        if not cls.dataset_csv.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {cls.dataset_csv}. "
                "Cannot run Giskard tests without sample data."
            )

        cls._prepare_wc_app_src_for_imports()

    @classmethod
    def _prepare_wc_app_src_for_imports(cls) -> None:
        """Normalize WC_APP_SRC so `src.*` imports from the external app resolve."""
        wc_app_src = os.getenv("WC_APP_SRC")
        if not wc_app_src:
            return

        candidate = Path(wc_app_src).expanduser().resolve()

        # Accept either:
        # - <app_root> where <app_root>/src/graph_builder.py exists
        # - <app_root>/src where graph_builder.py exists
        if (candidate / "src" / "graph_builder.py").exists():
            app_root = candidate
        elif (candidate / "graph_builder.py").exists() and candidate.name == "src":
            app_root = candidate.parent
        else:
            raise unittest.SkipTest(
                "WC_APP_SRC is set but does not point to a valid external app root or src/ directory. "
                f"Current value: {candidate}"
            )

        os.environ["WC_APP_SRC"] = str(app_root)
        app_root_str = str(app_root)
        if app_root_str not in sys.path:
            sys.path.insert(0, app_root_str)

        # The local project already imports its own `src` package.
        # Extend that package search path with the external app's `src` dir
        # so imports like `src.graph_builder` resolve from WC app modules.
        import src as local_src

        external_src_dir = str(app_root / "src")
        if external_src_dir not in local_src.__path__:
            local_src.__path__.append(external_src_dir)

    def _validate_scan_report(self, report: ScanReport, detector_name: str) -> None:
        """Validate that a ScanReport object is properly structured."""
        self.assertIsNotNone(report, f"ScanReport for {detector_name} is None")
        self.assertIsInstance(report, ScanReport)
        self.assertTrue(
            hasattr(report, "has_issues"),
            f"ScanReport for {detector_name} missing has_issues method",
        )
        self.assertTrue(
            hasattr(report, "issues"),
            f"ScanReport for {detector_name} missing issues attribute",
        )

    def test_run_harmfulness_scan(self):
        """Test harmfulness detector scan."""
        try:
            report = run_harmfulness_scan(
                dataset_csv=str(self.dataset_csv),
                n_samples=5,
                seed=42,
                n_adversarial_samples=2,
                n_requirements=2,
            )
            self._validate_scan_report(report, "harmfulness")
            self.assertIsInstance(report.issues, list)
        except EnvironmentError as exc:
            self.skipTest(f"Missing environment variables: {exc}")

    def test_run_stereotypes_scan(self):
        """Test stereotypes detector scan."""
        try:
            report = run_stereotypes_scan(
                dataset_csv=str(self.dataset_csv),
                n_samples=5,
                seed=42,
                n_adversarial_samples=2,
                n_requirements=2,
            )
            self._validate_scan_report(report, "stereotypes")
            self.assertIsInstance(report.issues, list)
        except EnvironmentError as exc:
            self.skipTest(f"Missing environment variables: {exc}")

    def test_run_jailbreak_scan(self):
        """Test jailbreak detector scan."""
        try:
            report = run_jailbreak_scan(
                dataset_csv=str(self.dataset_csv),
                n_samples=5,
                seed=42,
                n_adversarial_samples=2,
                n_requirements=2,
            )
            self._validate_scan_report(report, "jailbreak")
            self.assertIsInstance(report.issues, list)
        except EnvironmentError as exc:
            self.skipTest(f"Missing environment variables: {exc}")

    def test_all_detectors_with_minimal_params(self):
        """Test all three detectors together with minimal parameters."""
        detectors = [
            ("harmfulness", run_harmfulness_scan),
            ("stereotypes", run_stereotypes_scan),
            ("jailbreak", run_jailbreak_scan),
        ]

        results = {}
        try:
            for detector_name, scan_fn in detectors:
                report = scan_fn(
                    dataset_csv=str(self.dataset_csv),
                    n_samples=5,
                    seed=42,
                    n_adversarial_samples=2,
                    n_requirements=2,
                )
                self._validate_scan_report(report, detector_name)
                results[detector_name] = report

            # Verify all scans completed
            self.assertEqual(len(results), 3, "Not all detectors ran successfully")

        except EnvironmentError as exc:
            self.skipTest(f"Missing environment variables: {exc}")


if __name__ == "__main__":
    unittest.main()

