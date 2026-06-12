import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.constants.metrics_definitions import split_selected_metrics
from src.orchestration.eval_steps import step_giskard_scans


class TestSelectedMetricsSplit(unittest.TestCase):

    def test_none_defaults_to_scoring_pipeline_only(self):
        scoring, scans, groundedness = split_selected_metrics(None)
        self.assertIsNone(scoring)
        self.assertEqual(scans, [])
        self.assertEqual(groundedness, [])

    def test_mixed_metrics_are_normalized_and_partitioned(self):
        scoring, scans, groundedness = split_selected_metrics(
            ["Completeness", "potential harm", "correctness", "resilience", "correctness", "groundedness"]
        )
        self.assertEqual(scoring, ["completeness", "correctness"])
        self.assertEqual(scans, ["potential_harm", "resilience"])
        self.assertEqual(groundedness, ["groundedness"])


class TestGiskardScanStep(unittest.TestCase):

    def test_step_giskard_scans_uses_metric_specific_output_dirs(self):
        from src.evaluation.giskard_orchestrator import SCAN_METRIC_RUNNERS

        calls: list[tuple[str, dict]] = []

        def make_runner(name: str):
            def _runner(**kwargs):
                calls.append((name, kwargs))
            return _runner

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with patch.dict(
                SCAN_METRIC_RUNNERS,
                {
                    "potential_harm": make_runner("potential_harm"),
                    "toxicity": make_runner("toxicity"),
                    "resilience": make_runner("resilience"),
                },
                clear=True,
            ):
                step_giskard_scans(
                    input_path="final_data/all_results.csv",
                    output_dir=output_dir,
                    scan_metrics=["potential_harm", "toxicity"],
                    n_samples=7,
                    seed=11,
                    n_adversarial_samples=3,
                    n_requirements=2,
                    wc_version="v2",
                    wc_app_src="/tmp/external-wc",
                )

        self.assertEqual([name for name, _ in calls], ["potential_harm", "toxicity"])
        self.assertEqual(calls[0][1]["dataset_csv"], "final_data/all_results.csv")
        self.assertEqual(calls[0][1]["output_dir"], str(Path(output_dir) / "potential_harm"))
        self.assertEqual(calls[1][1]["output_dir"], str(Path(output_dir) / "toxicity"))
        self.assertTrue(all(kwargs["persist_output"] for _, kwargs in calls))
        self.assertTrue(all(kwargs["n_samples"] == 7 for _, kwargs in calls))
        self.assertTrue(all(kwargs["seed"] == 11 for _, kwargs in calls))
        self.assertTrue(all(kwargs["n_adversarial_samples"] == 3 for _, kwargs in calls))
        self.assertTrue(all(kwargs["n_requirements"] == 2 for _, kwargs in calls))
        self.assertTrue(all(kwargs["wc_version"] == "v2" for _, kwargs in calls))
        self.assertTrue(all(kwargs["wc_app_src"] == "/tmp/external-wc" for _, kwargs in calls))


if __name__ == "__main__":
    unittest.main()

