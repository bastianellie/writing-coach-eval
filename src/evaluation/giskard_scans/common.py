"""Shared helpers for detector-specific Giskard security scans."""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
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

_wc_graph = None
_wc_preset_config = None


def _get_wc_app_src() -> Optional[str]:
	return os.getenv("WC_APP_SRC")


def _ensure_wc_path_on_syspath() -> str:
	wc_app_src = _get_wc_app_src()
	if not wc_app_src:
		raise RuntimeError(
			"WC_APP_SRC not set. Set WC_APP_SRC to the root of the Writing Coach app codebase."
		)

	wc_root = str(Path(wc_app_src).resolve())
	if wc_root not in sys.path:
		sys.path.insert(0, wc_root)
		logger.info("Writing Coach app root added to sys.path: %s", wc_root)
	return wc_root


def _load_wc_dependencies():
	# Imports are intentionally lazy so importing this module does not require WC app code.
	from langgraph.graph import END, START  # type: ignore[import]
	from src.graph_builder import GraphBuilder  # type: ignore[import]
	from src.graph_nodes.writing_coach_nodes import (  # type: ignore[import]
		output_node,
		research_response_node,
		research_transform_node,
		revision_explanation_node,
		search_node,
		search_router,
		segment_analysis_node,
		segment_analysis_router,
		simple_transform_node,
		wc_orchestrator_node,
		wc_orchestrator_router,
		wc_respond_node,
	)
	from src.graph_presets import get_preset, register_preset  # type: ignore[import]
	from src.presets_config import get_preset_config  # type: ignore[import]
	from src.state_definitions import WritingCoachV2State  # type: ignore[import]

	return {
		"START": START,
		"END": END,
		"GraphBuilder": GraphBuilder,
		"WritingCoachV2State": WritingCoachV2State,
		"get_preset": get_preset,
		"register_preset": register_preset,
		"get_preset_config": get_preset_config,
		"wc_orchestrator_node": wc_orchestrator_node,
		"wc_orchestrator_router": wc_orchestrator_router,
		"segment_analysis_node": segment_analysis_node,
		"segment_analysis_router": segment_analysis_router,
		"search_node": search_node,
		"search_router": search_router,
		"research_response_node": research_response_node,
		"research_transform_node": research_transform_node,
		"simple_transform_node": simple_transform_node,
		"revision_explanation_node": revision_explanation_node,
		"wc_respond_node": wc_respond_node,
		"output_node": output_node,
	}


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
	if not _get_wc_app_src():
		raise RuntimeError(
			"WC_APP_SRC not set. Set WC_APP_SRC to the root of the Writing Coach app codebase."
		)
	return _writing_coach_predict_real(df)


def init_wc_graph() -> None:
	"""Initialize Writing Coach graph once and cache it."""
	global _wc_graph, _wc_preset_config  # noqa: PLW0603

	if _wc_graph is not None:
		return

	wc_root = _ensure_wc_path_on_syspath()
	deps = _load_wc_dependencies()

	logger.info("Initializing Writing Coach V2 graph from %s", wc_root)

	builder = deps["GraphBuilder"](deps["WritingCoachV2State"])
	builder.set_display_name("Writing Coach V2")
	builder.set_description("Conversational writing coach with hybrid graph architecture")

	for name, fn in [
		("wc2/orchestrator", deps["wc_orchestrator_node"]),
		("wc2/segment_analysis", deps["segment_analysis_node"]),
		("wc2/copilot_search", deps["search_node"]),
		("wc2/research_response", deps["research_response_node"]),
		("wc2/research_transform", deps["research_transform_node"]),
		("wc2/simple_transform", deps["simple_transform_node"]),
		("wc2/revision_explanation", deps["revision_explanation_node"]),
		("wc2/respond", deps["wc_respond_node"]),
		("wc2/output", deps["output_node"]),
	]:
		builder.register_node_function(name, fn)
		builder.add_node(name)

	builder.add_edge(deps["START"], "wc2/orchestrator")
	builder.add_conditional_edge("wc2/orchestrator", deps["wc_orchestrator_router"], {
		"wc2/segment_analysis": "wc2/segment_analysis",
		"wc2/respond": "wc2/respond",
	})
	builder.add_conditional_edge("wc2/segment_analysis", deps["segment_analysis_router"], {
		"wc2/copilot_search": "wc2/copilot_search",
		"wc2/simple_transform": "wc2/simple_transform",
		"wc2/revision_explanation": "wc2/revision_explanation",
		"wc2/respond": "wc2/respond",
	})
	builder.add_conditional_edge("wc2/copilot_search", deps["search_router"], {
		"wc2/research_response": "wc2/research_response",
		"wc2/research_transform": "wc2/research_transform",
	})
	builder.add_edge("wc2/research_response", "wc2/output")
	builder.add_edge("wc2/research_transform", "wc2/revision_explanation")
	builder.add_edge("wc2/simple_transform", "wc2/revision_explanation")
	builder.add_edge("wc2/revision_explanation", "wc2/output")
	builder.add_edge("wc2/respond", "wc2/output")
	builder.add_edge("wc2/output", deps["END"])

	deps["register_preset"]("writing_coach_v2", builder)
	_wc_graph = deps["get_preset"]("writing_coach_v2").build_without_checkpointing()
	_wc_preset_config = deps["get_preset_config"]("writing_coach_v2")
	logger.info("Writing Coach V2 graph ready")


def _writing_coach_predict_real(df) -> list[str]:
	responses: list[str] = []

	for idx, (_, row) in enumerate(df.iterrows()):
		initial_state = {
			"message": row["user_command"],
			"document_text": row["document"],
			"selected_text": None,
			"conversation_history": [],
			"prior_references": [],
			"conversation_id": f"giskard_scan_{idx}",
			"conversation_turn": 1,
			"streaming": False,
			"writer": None,
			"preset": "writing_coach_v2",
			"model_config": _wc_preset_config.get("models", {}),
			"prompt_versions": _wc_preset_config.get("prompts", {}),
			"parameters": _wc_preset_config.get("parameters", {}),
			"suggestions": [],
			"references": [],
			"research_papers": [],
		}
		try:
			final_state = _wc_graph.invoke(initial_state)
			responses.append(build_response(final_state))
		except openai.BadRequestError as exc:
			responses.append(json.dumps(exc.body))

	return responses


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
    model_type: SupportedModelTypes = "text_generation",
    persist_output: bool = False,
    output_dir: Optional[str] = None,
) -> ScanReport:
    """Run scan for a single detector and optionally persist results.

    Args:
        detector: Detector name to run.
        dataset_csv: Path to CSV dataset file.
        n_samples: Number of samples to use from dataset.
        seed: Random seed for sampling.
        n_adversarial_samples: Number of adversarial samples per detector.
        n_requirements: Number of requirements per detector.
        model_type: Giskard model type.
        persist_output: Whether to save results to disk.
        output_dir: Directory to save results (required if persist_output=True).

    Returns:
        ScanReport object from the Giskard scan.
    """
    configure_giskard_llm_client()

    logger.info("Building Giskard model wrapper")
    gsk_model = build_giskard_model(model_type=model_type)

    sample_docs = load_sample_documents(
        Path(dataset_csv) if dataset_csv else None,
        n_samples=n_samples,
        seed=seed,
    )
    gsk_dataset = build_giskard_dataset(sample_docs)

    logger.info("Initializing Writing Coach graph")
    init_wc_graph()

    logger.info("Running detector: %s", detector)
    detector_params = build_detector_params(detector, n_adversarial_samples, n_requirements)

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


def validate_required_env_vars() -> None:
	missing = [
		name
		for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "WC_APP_SRC")
		if not os.getenv(name)
	]
	if missing:
		raise EnvironmentError(
			"Missing required environment variables: " + ", ".join(missing)
		)

