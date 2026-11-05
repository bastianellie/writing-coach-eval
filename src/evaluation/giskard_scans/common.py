"""Shared helpers for detector-specific Giskard security scans."""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import giskard
import litellm
import openai
from dotenv import load_dotenv
from giskard import Dataset, Model, scan
from giskard.core.core import SupportedModelTypes
from giskard.scanner.registry import DetectorRegistry
from giskard.scanner.report import ScanReport
from src.scripts.writing_coach_interfaces import create_writing_coach
from src.scripts.writing_coach_interfaces.utils import install_wc_app_path

load_dotenv()

logger = logging.getLogger("giskard_scan")
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
	datefmt="%H:%M:%S",
)

litellm._turn_on_debug()
litellm.drop_params = True

for noisy in ("langsmith", "langchain", "httpx", "httpcore", "openai"):
	logging.getLogger(noisy).setLevel(logging.WARNING)

_DEFAULT_DATASET_CSV = Path(__file__).parents[3] / "final_data" / "all_results.csv"

_wc_coach = None
_wc_version: str | None = None

# Thread-pool size for the model predict batch (see _writing_coach_predict_real).
# Giskard owns the predict call, so the worker count is passed via this module
# global, set by run_scan_for_detector before each scan.
_predict_workers: int = 1


def set_predict_workers(workers: int) -> None:
	"""Set the number of threads used to score the predict batch in parallel."""
	global _predict_workers  # noqa: PLW0603
	_predict_workers = max(1, workers)


# ---------------------------------------------------------------------------
# Writing Coach / LeapSpace log routing
#
# The external WC app (loaded from WC_APP_SRC) attaches its own console logging
# handlers (via basicConfig deep inside) and emits a lot of bare print() output
# when it runs the real model. Both flood the screen during Giskard scans. These
# helpers route the app's logs to a file and capture its stdout, so only the
# eval pipeline's own output stays on screen. Adapted from src/scripts/store_output.py.
# ---------------------------------------------------------------------------

_wc_log_file: str = os.getenv("WC_LOG_FILE", "wc_app.log")
_wc_log_verbose: bool = False


def set_wc_logging(log_file: Optional[str] = None, verbose: bool = False) -> None:
	"""Configure where Writing Coach / LeapSpace logs go.

	Call before the first scan. *log_file* overrides the default destination;
	*verbose* keeps the app's INFO logs and stdout on the console as well.
	"""
	global _wc_log_file, _wc_log_verbose  # noqa: PLW0603
	if log_file:
		_wc_log_file = log_file
	_wc_log_verbose = verbose


def _route_wc_logs_to_file() -> None:
	"""Send the app's (and root) logs to the log file; keep console quiet.

	Strips the console handlers the WC app attaches to the root logger so its
	INFO logs stop printing on screen, routes INFO+ to ``_wc_log_file``, and
	leaves only ERROR (or everything, when verbose) on the console. Must run
	*after* the app has been imported (i.e. after create_writing_coach) so the
	handlers it added are removed and this configuration wins.
	"""
	file_handler = logging.FileHandler(_wc_log_file, mode="a", encoding="utf-8")
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
	file_handler._wc_log_managed = True

	console_handler = logging.StreamHandler(sys.stderr)
	console_handler.setLevel(logging.INFO if _wc_log_verbose else logging.ERROR)
	console_handler.setFormatter(logging.Formatter("%(message)s"))
	console_handler._wc_log_managed = True

	root = logging.getLogger()
	root.setLevel(logging.INFO)
	for handler in list(root.handlers):
		# Drop the app's console handlers (StreamHandler to stdout/stderr) so its
		# INFO logs stop flooding the screen; keep FileHandlers. Also drop our
		# own handlers from a previous call so re-running is idempotent.
		is_console = (
			isinstance(handler, logging.StreamHandler)
			and not isinstance(handler, logging.FileHandler)
		)
		if is_console or getattr(handler, "_wc_log_managed", False):
			handler.close()
			root.removeHandler(handler)
	root.addHandler(file_handler)
	root.addHandler(console_handler)
	logger.info("Writing Coach / LeapSpace logs routed to %s (verbose=%s)", _wc_log_file, _wc_log_verbose)


@contextmanager
def _redirect_stdout_to_log():
	"""Redirect bare print() output to the log file while active.

	The WC app emits progress via print() (bypassing logging). Redirecting
	stdout to the log file captures it; stderr (tqdm bars, error logs) is left
	alone. A no-op in verbose mode so prints stay on screen.
	"""
	if _wc_log_verbose:
		yield
		return
	sink = open(_wc_log_file, "a", encoding="utf-8", buffering=1)
	original_stdout = sys.stdout
	sys.stdout = sink
	try:
		yield
	finally:
		sys.stdout = original_stdout
		sink.close()


def _get_wc_app_src() -> Optional[str]:
	return os.getenv("WC_APP_SRC")


def _resolve_wc_app_src(wc_app_src: Optional[str]) -> str:
	resolved = wc_app_src or _get_wc_app_src()
	if not resolved:
		raise RuntimeError(
			"WC_APP_SRC not set. Set WC_APP_SRC or pass --wc-app-src to the pipeline."
		)
	return resolved


def init_wc_interface(version: str = "v3", wc_app_src: Optional[str] = None) -> None:
	"""Initialize Writing Coach interface once and cache it."""
	global _wc_coach, _wc_version  # noqa: PLW0603

	if _wc_coach is not None and _wc_version == version:
		return

	resolved_wc_app_src = _resolve_wc_app_src(wc_app_src)
	install_wc_app_path(resolved_wc_app_src)
	logger.info("Initializing Writing Coach interface '%s' from %s", version, resolved_wc_app_src)
	_wc_coach = create_writing_coach(version)
	_wc_version = version

	# Strip the console handlers the app just attached and route its logs to a
	# file (must run after create_writing_coach so our configuration wins).
	_route_wc_logs_to_file()


def load_sample_documents(
	csv_path: Path | None = None,
	n_samples: int | None = None,
	seed: int | None = None,
) -> list[dict]:
	"""Load representative scan inputs from CSV."""
	import pandas as pd

	if csv_path is None:
		env_override = os.getenv("GISKARD_DATASET_CSV")
		csv_path = Path(env_override) if env_override else _DEFAULT_DATASET_CSV

	logger.info("Loading sample documents from %s", csv_path)
	df = pd.read_csv(csv_path, usecols=["query", "input"])
	df = df.dropna(subset=["query", "input"])
	df = df[df["query"].str.strip().astype(bool) & df["input"].str.strip().astype(bool)]

	if n_samples is not None and n_samples < len(df):
		df = df.sample(n=n_samples, random_state=seed)
		logger.info("Sampled %d rows (seed=%s)", n_samples, seed)

	records = [
		{"user_command": row["query"], "document": row["input"]}
		for _, row in df.iterrows()
	]
	logger.info("Loaded %d sample documents", len(records))
	return records


def writing_coach_predict(df) -> list[str]:
	"""Prediction function consumed by giskard.Model."""
	if _wc_coach is None:
		raise RuntimeError(
			"Writing Coach interface is not initialized. Call run_scan_for_detector first."
		)
	return _writing_coach_predict_real(df)


def _predict_single(idx: int, row) -> str:
	"""Run one Writing Coach query and render its response (one predict-batch item)."""
	try:
		result = _wc_coach.run_query(
			row_id=idx,
			query=row["user_command"],
			document_text=row["document"],
		)
		return build_response(result)
	except openai.BadRequestError as exc:
		return json.dumps(exc.body)
	except Exception as exc:  # noqa: BLE001
		logger.exception("Writing Coach prediction failed for scan row %s", idx)
		return json.dumps({"error": str(exc)})


def _writing_coach_predict_real(df) -> list[str]:
	if _wc_coach is None:
		raise RuntimeError("Writing Coach interface has not been initialized")

	rows = [row for _, row in df.iterrows()]

	# Sequential path keeps behaviour identical when no parallelism is requested.
	if _predict_workers <= 1 or len(rows) <= 1:
		return [_predict_single(idx, row) for idx, row in enumerate(rows)]

	# Parallel path: score the predict batch with a thread pool. executor.map
	# preserves input order, so responses stay aligned with the dataset rows.
	# NOTE: this assumes _wc_coach.run_query is safe to call concurrently
	# (the coach lives in the external WC app at WC_APP_SRC).
	workers = min(_predict_workers, len(rows))
	with ThreadPoolExecutor(max_workers=workers) as executor:
		return list(executor.map(lambda item: _predict_single(*item), enumerate(rows)))


def build_response(state: dict) -> str:
	response = state.get("response", "")
	suggestions = state.get("suggestions", [])
	if suggestions:
		response += "\n\n## SUGGESTIONS:\n" + "\n".join(f"- {s}" for s in suggestions)
	return response


def configure_giskard_llm_client() -> None:
	"""Configure Giskard evaluator LLM endpoint."""
	giskard.llm.set_llm_model(
		"azure/gpt-5-chat",
		api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
	)


def build_giskard_dataset(sample_docs: list[dict]) -> Dataset:
	import pandas as pd

	df = pd.DataFrame(sample_docs)
	return Dataset(
		df=df,
		name="Writing Coach Eval Dataset",
		column_types={
			"document": "text",
			"user_command": "text",
		},
	)


def build_giskard_model(model_type: SupportedModelTypes = "text_generation") -> Model:
	return Model(
		model=writing_coach_predict,
		model_type=model_type,
		name="Writing Coach AI Assistant",
		description=(
			"An AI assistant that helps users edit and improve academic texts. "
			"The system receives a document and a user command and returns suggested edits."
		),
		feature_names=["document", "user_command"],
	)


def build_detector_params(
	detector: str,
	n_adversarial_samples: Optional[int],
	n_requirements: Optional[int],
) -> Optional[dict]:
	if n_adversarial_samples is None and n_requirements is None:
		return None

	detector_params: dict = {}
	for label, cls in DetectorRegistry.get_detector_classes([detector]).items():
		try:
			init_params = set(inspect.signature(cls.__init__).parameters.keys())
		except (ValueError, TypeError) as exc:
			logger.warning(
				"Could not inspect __init__ for detector '%s' (%s): %s",
				label,
				cls.__name__,
				exc,
			)
			continue

		det_cfg: dict = {}
		if n_adversarial_samples is not None and "num_samples" in init_params:
			det_cfg["num_samples"] = n_adversarial_samples
		if n_requirements is not None and "num_requirements" in init_params:
			det_cfg["num_requirements"] = n_requirements

		if det_cfg:
			detector_params[label] = det_cfg

	return detector_params or None


def run_scan_for_detector(
    detector: str,
    dataset_csv: Optional[str] = None,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    n_adversarial_samples: Optional[int] = None,
    n_requirements: Optional[int] = None,
	wc_version: str = "v3",
	wc_app_src: Optional[str] = None,
    model_type: SupportedModelTypes = "text_generation",
    persist_output: bool = False,
    output_dir: Optional[str] = None,
    workers: int = 1,
) -> ScanReport:
    """Run scan for a single detector and optionally persist results.

    Args:
        detector: Detector name to run.
        dataset_csv: Path to CSV dataset file.
        n_samples: Number of samples to use from dataset.
        seed: Random seed for sampling.
        n_adversarial_samples: Number of adversarial samples per detector.
		n_requirements: Number of requirements per detector.
		wc_version: Writing Coach interface version to use (v2 or v3).
		wc_app_src: Optional path to external Writing Coach repo root or src dir.
        model_type: Giskard model type.
        persist_output: Whether to save results to disk.
        output_dir: Directory to save results (required if persist_output=True).
        workers: Threads used to score the model predict batch in parallel.

    Returns:
        ScanReport object from the Giskard scan.
    """
    configure_giskard_llm_client()
    set_predict_workers(workers)

    logger.info("Building Giskard model wrapper")
    gsk_model = build_giskard_model(model_type=model_type)

    sample_docs = load_sample_documents(
        Path(dataset_csv) if dataset_csv else None,
        n_samples=n_samples,
        seed=seed,
    )
    gsk_dataset = build_giskard_dataset(sample_docs)

    logger.info("Initializing Writing Coach interface")
    init_wc_interface(version=wc_version, wc_app_src=wc_app_src)

    logger.info("Running detector: %s", detector)
    detector_params = build_detector_params(detector, n_adversarial_samples, n_requirements)

    # Capture the WC app's bare print() flood (emitted while it runs the model
    # during the scan) to the log file instead of the screen.
    with _redirect_stdout_to_log():
        scan_results: ScanReport = scan(
            gsk_model,
            gsk_dataset,
            only=[detector],
            raise_exceptions=False,
            params=detector_params,
        )

    if persist_output:
        if output_dir is None:
            raise ValueError("output_dir is required when persist_output=True")
        persist_results(
            scan_results,
            output_dir,
            n_adversarial_samples=n_adversarial_samples,
            n_requirements=n_requirements,
        )
        print_summary(
            scan_results,
            detector=detector,
            n_adversarial_samples=n_adversarial_samples,
            n_requirements=n_requirements,
        )

    return scan_results


def persist_results(
	scan_results: ScanReport,
	output_dir: str,
	n_adversarial_samples: Optional[int] = None,
	n_requirements: Optional[int] = None,
) -> None:
	out_path = Path(output_dir)
	out_path.mkdir(parents=True, exist_ok=True)

	html_path = out_path / "scan_report.html"
	scan_results.to_html(str(html_path))
	logger.info("HTML report saved to %s", html_path)

	save_json_summary(
		scan_results,
		out_path,
		n_adversarial_samples=n_adversarial_samples,
		n_requirements=n_requirements,
	)

	try:
		test_suite = scan_results.generate_test_suite("Writing Coach Security Tests")
		suite_dir = out_path / "test_suite"
		test_suite.save(str(suite_dir))
		logger.info("Test suite saved to %s", suite_dir)
	except Exception as exc:  # noqa: BLE001
		logger.warning("Could not generate test suite: %s", exc)


def example_to_dict(example: dict | str) -> dict:
	import ast
	import re

	raw: str = example.get("input", "") if isinstance(example, dict) else str(example)

	user_match = re.search(r"USER:\s*", raw)
	agent_match = re.search(r"\n\nAGENT:\s*", raw)
	suggestions_match = re.search(r"\n\n## SUGGESTIONS:\s*\n", raw)

	user_parsed: dict | str = {}
	if user_match:
		start = user_match.end()
		end = agent_match.start() if agent_match else (suggestions_match.start() if suggestions_match else len(raw))
		user_str = raw[start:end].strip()
		try:
			user_parsed = ast.literal_eval(user_str)
		except (ValueError, SyntaxError):
			user_parsed = user_str

	agent_parsed = ""
	if agent_match:
		start = agent_match.end()
		end = suggestions_match.start() if suggestions_match else len(raw)
		agent_parsed = raw[start:end].strip()

	suggestions_parsed = []
	if suggestions_match:
		start = suggestions_match.end()
		suggestions_str = raw[start:].strip()
		for line in re.split(r"\n-\s*", suggestions_str):
			line = line.lstrip("- ").strip()
			if not line:
				continue
			try:
				suggestions_parsed.append(ast.literal_eval(line))
			except (ValueError, SyntaxError):
				suggestions_parsed.append(line)

	input_text = ""
	user_command = ""
	if isinstance(user_parsed, dict):
		input_text = str(user_parsed.get("document", ""))
		user_command = str(user_parsed.get("user_command", ""))

	return {
		"input_text": input_text,
		"user_command": user_command,
		"agent": agent_parsed,
		"suggestions": suggestions_parsed,
	}


def save_json_summary(
	scan_results: ScanReport,
	out_path: Path,
	n_adversarial_samples: Optional[int] = None,
	n_requirements: Optional[int] = None,
) -> None:
	issues = []
	for issue in scan_results.issues:
		issue_examples = []
		if issue.scan_examples is not None:
			examples = issue.scan_examples.examples.to_dict(orient="records")
			for example in examples:
				reason = example.get("Reason", "")
				conversation = example.get("Conversation", "")
				system = example_to_dict(conversation)
				issue_examples.append({"reason": reason, "system": system})

		issues.append(
			{
				"detector": getattr(issue, "detector_name", type(issue).__name__),
				"group": getattr(issue, "group", ""),
				"level": getattr(issue, "level", ""),
				"description": getattr(issue, "description", ""),
				"meta": getattr(issue, "meta", {}),
				"examples": issue_examples,
			}
		)

	summary = {
		"has_issues": scan_results.has_issues(),
		"total_issues": len(issues),
		"issues": issues,
	}
	if n_adversarial_samples is not None:
		summary["n_adversarial_samples"] = n_adversarial_samples
	if n_requirements is not None:
		summary["n_requirements"] = n_requirements

	json_path = out_path / "scan_summary.json"
	with open(json_path, "w", encoding="utf-8") as fh:
		json.dump(summary, fh, indent=2, default=str)
	logger.info("JSON summary saved to %s", json_path)


def print_summary(
	scan_results: ScanReport,
	detector: str,
	n_adversarial_samples: Optional[int] = None,
	n_requirements: Optional[int] = None,
) -> None:
	print("\n" + "=" * 64)
	print(f"  GISKARD SCAN SUMMARY - {detector}")
	print("=" * 64)
	print(" Number of adversarial samples per detector: %s" % (n_adversarial_samples or "default"))
	print(" Number of requirements per detector:        %s" % (n_requirements or "default"))
	print("-" * 64)

	if not scan_results.has_issues():
		print("No vulnerabilities detected.")
	else:
		n_issues = len(scan_results.issues)
		print(f"{n_issues} issue(s) detected:\n")
		for idx, issue in enumerate(scan_results.issues, start=1):
			detector_name = getattr(issue, "detector_name", type(issue).__name__)
			level = str(getattr(issue, "level", "unknown")).upper()
			description = getattr(issue, "description", "(no description)")
			print(f"  [{idx}] [{level}] {detector_name}")
			print(f"       {description}\n")

	print("=" * 64 + "\n")


def validate_required_env_vars(wc_app_src: Optional[str] = None) -> None:
	missing = [
		name
		for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
		if not os.getenv(name)
	]
	if not (wc_app_src or os.getenv("WC_APP_SRC")):
		missing.append("WC_APP_SRC")
	if missing:
		raise EnvironmentError(
			"Missing required environment variables: " + ", ".join(missing)
		)

