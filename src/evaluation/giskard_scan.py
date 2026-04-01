"""Giskard security scan for the Writing Coach AI assistant.

Runs automated vulnerability detection for two threat categories:

**Harmfulness**
    Checks whether the model can be coerced — via the user command or the
    document field — into generating harmful, unethical, or dangerous content
    (e.g. instructions for violence, hate speech, self-harm guidance).

**Prompt Injection**
    Checks whether adversarial instructions hidden *inside the document*
    (the manuscript/text the user pastes) can hijack the assistant's behaviour
    and override the legitimate editing task.

System inputs modelled here
---------------------------
- ``document``     : the text the user wants to edit (attacker-controlled
                     in injection scenarios — it is plain user content).
- ``user_command`` : the editing instruction given in chat (e.g.
                     "fix grammar errors", "strengthen the argument").

Usage (from the repo root)
--------------------------
::

    python -m src.evaluation.giskard_scan
    python -m src.evaluation.giskard_scan --detectors harmfulness prompt_injection
    python -m src.evaluation.giskard_scan --output-dir giskard_results/
    python -m src.evaluation.giskard_scan --detectors harmfulness --output-dir my_dir/
    python -m src.evaluation.giskard_scan --n-adversarial-samples 10
    python -m src.evaluation.giskard_scan --num-requirements 5
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import litellm
import openai
from dotenv import load_dotenv
import giskard
from giskard import Model, Dataset, scan
from giskard.core.core import SupportedModelTypes
from giskard.scanner.registry import DetectorRegistry
from giskard.scanner.report import ScanReport

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("giskard_scan")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

litellm._turn_on_debug()
litellm.drop_params = True

# ---------------------------------------------------------------------------
# Writing Coach app codebase path injection
# ---------------------------------------------------------------------------
# The real Writing Coach model lives in a sibling codebase that is not
# installed as a package.  Set the WC_APP_SRC environment variable to the
# root directory of that codebase (the folder that contains its own `src/`
# package) and this module will make it importable.
#
# Example .env entry:
#   WC_APP_SRC=/path/to/writing-coach-app
#
# When WC_APP_SRC is set, writing_coach_predict() will invoke the real graph
# instead of the ad-hoc Azure OpenAI call below.  When it is absent the
# fallback LLM-based predictor is used so the scan can still run without the
# full app codebase available.

_WC_APP_SRC = os.getenv("WC_APP_SRC")
if _WC_APP_SRC:
    _wc_root = str(Path(_WC_APP_SRC).resolve())
    if _wc_root not in sys.path:
        sys.path.insert(0, _wc_root)
    logger.info("Writing Coach app root added to sys.path → %s", _wc_root)

# ---------------------------------------------------------------------------
# Custom detector registration
# ---------------------------------------------------------------------------
# Load the detector module directly from its file path so that the @detector
# decorator fires and registers the class in Giskard's DetectorRegistry.
# We use importlib.util.spec_from_file_location instead of a normal import
# statement to avoid the `src` name collision: when WC_APP_SRC is set,
# `src` on sys.path points to the external app's src/, so any
# `from src.evaluation.detectors import ...` would look there instead of here.
_DETECTORS_FILE = Path(__file__).parent / "detectors" / "llm_broken_text_detector.py"
_spec = importlib.util.spec_from_file_location(
    "writing_coach_eval.detectors.llm_broken_text_detector",
    _DETECTORS_FILE,
)
_detector_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detector_mod)  # this runs the @detector decorator

for _noisy in ("langsmith", "langchain", "httpx", "httpcore", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

    from src.graph_presets import get_preset, register_preset  # type: ignore[import]
    from src.presets_config import get_preset_config            # type: ignore[import]
    from src.state_definitions import WritingCoachV2State       # type: ignore[import]
    from src.graph_builder import GraphBuilder                  # type: ignore[import]
    from src.graph_nodes.writing_coach_nodes import (        # type: ignore[import]
        wc_orchestrator_node, wc_orchestrator_router,
        segment_analysis_node, segment_analysis_router,
        search_node, search_router,
        research_response_node, research_transform_node,
        simple_transform_node, revision_explanation_node,
        wc_respond_node, output_node,
    )
    from langgraph.graph import START, END  # type: ignore[import]

# ---------------------------------------------------------------------------
# Representative sample documents (seed dataset for the Giskard scan)
# ---------------------------------------------------------------------------
# Loaded from a CSV file whose `query` and `input` columns map to
# `user_command` and `document` respectively.
# Override the path with the GISKARD_DATASET_CSV environment variable.

_DEFAULT_DATASET_CSV = Path(__file__).parents[2] / "final_data" / "all_results.csv"


def _load_sample_documents(
    csv_path: Path | None = None,
    n_samples: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Load sample documents from a CSV file.

    The CSV must contain at least the columns ``query`` and ``input``.
    Rows where either column is empty are skipped.

    Args:
        csv_path: Path to the CSV file. Defaults to
                  ``final_data/all_results.csv`` relative to the repo root,
                  or the value of the ``GISKARD_DATASET_CSV`` env var.
        n_samples: Number of rows to randomly sample. If ``None`` or greater
                   than the number of available rows, all rows are used.
        seed: Random seed for reproducible sampling. Has no effect when
              ``n_samples`` is ``None``.

    Returns:
        A list of dicts with keys ``document`` and ``user_command``.
    """
    import pandas as pd

    if csv_path is None:
        env_override = os.getenv("GISKARD_DATASET_CSV")
        csv_path = Path(env_override) if env_override else _DEFAULT_DATASET_CSV

    logger.info("Loading sample documents from %s …", csv_path)
    df = pd.read_csv(csv_path, usecols=["query", "input"])
    df = df.dropna(subset=["query", "input"])
    df = df[df["query"].str.strip().astype(bool) & df["input"].str.strip().astype(bool)]

    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed)
        logger.info("Sampled %d rows (seed=%s).", n_samples, seed)

    records = [
        {"user_command": row["query"], "document": row["input"]}
        for _, row in df.iterrows()
    ]
    logger.info("Loaded %d sample documents.", len(records))
    return records


# --------------------------------------------------------
# Giskard model prediction function
# ---------------------------------------------------------------------------


def writing_coach_predict(df) -> list[str]:
    """Prediction function consumed by ``giskard.Model``.

    Accepts a pandas DataFrame with columns ``document`` and ``user_command``
    and returns the assistant responses as a list of strings.

    **When WC_APP_SRC is set** the real Writing Coach V2 LangGraph graph is
    used (same invocation as ``src/store_output.py``).  The graph is
    initialised lazily on first call so the module can be imported without the
    app codebase being present.

    **When WC_APP_SRC is absent** a lightweight Azure OpenAI call using the
    local ``WRITING_COACH_SYSTEM_PROMPT`` is used as a stand-in, which still
    lets the scan run for quick checks.
    """

    if _WC_APP_SRC:
        print("Invoking real Writing Coach graph from app source …")
        return _writing_coach_predict_real(df)
    raise RuntimeError(
        "WC_APP_SRC not set — aborting.  "
        "Set WC_APP_SRC to the root of the Writing Coach app codebase to run the real graph."
    )


# -- Real graph predictor (requires WC_APP_SRC) ------------------------------

_wc_graph = None          # lazily initialised LangGraph CompiledGraph
_wc_preset_config = None  # preset config dict


def _init_wc_graph():
    """Initialise the Writing Coach V2 graph once and cache it."""
    global _wc_graph, _wc_preset_config  # noqa: PLW0603

    if _wc_graph is not None:
        return  # already initialised

    logger.info("Initialising Writing Coach V2 graph from %s …", _WC_APP_SRC)

    builder = GraphBuilder(WritingCoachV2State)
    builder.set_display_name("Writing Coach V2")
    builder.set_description("Conversational writing coach with hybrid graph architecture")

    for name, fn in [
        ("wc2/orchestrator", wc_orchestrator_node),
        ("wc2/segment_analysis", segment_analysis_node),
        ("wc2/copilot_search", search_node),
        ("wc2/research_response", research_response_node),
        ("wc2/research_transform", research_transform_node),
        ("wc2/simple_transform", simple_transform_node),
        ("wc2/revision_explanation", revision_explanation_node),
        ("wc2/respond", wc_respond_node),
        ("wc2/output", output_node),
    ]:
        builder.register_node_function(name, fn)
        builder.add_node(name)

    builder.add_edge(START, "wc2/orchestrator")
    builder.add_conditional_edge("wc2/orchestrator", wc_orchestrator_router, {
        "wc2/segment_analysis": "wc2/segment_analysis",
        "wc2/respond": "wc2/respond",
    })
    builder.add_conditional_edge("wc2/segment_analysis", segment_analysis_router, {
        "wc2/copilot_search": "wc2/copilot_search",
        "wc2/simple_transform": "wc2/simple_transform",
        "wc2/revision_explanation": "wc2/revision_explanation",
        "wc2/respond": "wc2/respond",
    })
    builder.add_conditional_edge("wc2/copilot_search", search_router, {
        "wc2/research_response": "wc2/research_response",
        "wc2/research_transform": "wc2/research_transform",
    })
    builder.add_edge("wc2/research_response", "wc2/output")
    builder.add_edge("wc2/research_transform", "wc2/revision_explanation")
    builder.add_edge("wc2/simple_transform", "wc2/revision_explanation")
    builder.add_edge("wc2/revision_explanation", "wc2/output")
    builder.add_edge("wc2/respond", "wc2/output")
    builder.add_edge("wc2/output", END)

    register_preset("writing_coach_v2", builder)
    _wc_graph = get_preset("writing_coach_v2").build_without_checkpointing()
    _wc_preset_config = get_preset_config("writing_coach_v2")
    logger.info("Writing Coach V2 graph ready.")


def _writing_coach_predict_real(df) -> list[str]:
    """Invoke the real Writing Coach V2 graph for each row in *df*."""

    from src.state_definitions import WritingCoachV2State  # type: ignore[import]

    responses: list[str] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        print("Analyzing ", row)
        initial_state: WritingCoachV2State = {
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
            response = _build_response(final_state)
            responses.append(response)
        except openai.BadRequestError as exc:
            responses.append(json.dumps(exc.body))

    return responses


def _build_response(state: WritingCoachV2State) -> str:
    """Extract the assistant's response text from the final graph state."""
    response = state.get("response", "")
    suggestions = state.get("suggestions", [])
    if suggestions:
        response += "\n\n## SUGGESTIONS:\n" + "\n".join(f"- {s}\n\n" for s in suggestions)
    return response

# ---------------------------------------------------------------------------
# Giskard LLM evaluator configuration
# ---------------------------------------------------------------------------


def _configure_giskard_llm_client() -> None:
    """Point Giskard's internal LLM evaluator at Azure OpenAI.

    Giskard uses an LLM internally to generate adversarial probes and evaluate
    model responses.  We reuse the same Azure OpenAI deployment via Giskard's
    built-in ``OpenAIClient`` wrapper, which accepts any ``openai``-compatible
    client instance (including ``AzureOpenAI``).
    """
    # These imports live inside the function so that missing optional Giskard
    # sub-packages never break the module at import time.
    from giskard.llm.client.openai import OpenAIClient  # type: ignore[import]
    giskard.llm.set_llm_model("azure/gpt-5-chat", api_base=os.getenv("AZURE_OPENAI_ENDPOINT"))


# ---------------------------------------------------------------------------
# Dataset and Model builders
# ---------------------------------------------------------------------------


def build_giskard_dataset(sample_docs: list[dict]) -> Dataset:
    """Build a ``giskard.Dataset`` from the representative sample documents."""
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
    """Wrap the Writing Coach prediction function in a ``giskard.Model``."""
    return Model(
        model=writing_coach_predict,
        model_type=model_type,
        name="Writing Coach AI Assistant",
        description=(
            "An AI assistant that helps users edit and improve academic texts "
            "(manuscripts, essays, research papers) through a chat-like interface. "
            "The system receives a document (the text to improve) and a user command "
            "(the editing instruction), and returns suggested edits, revisions, or "
            "writing feedback."
        ),
        feature_names=["document", "user_command"],
    )


# ---------------------------------------------------------------------------
# Detector loading — defensive dynamic import for forward-compatibility
# ---------------------------------------------------------------------------

# Short name → (primary module path, class name, fallback module paths)
VALID_DETECTORS: list[str] = [
    "jailbreak",
    "stereotypes",
    "harmfulness",
    # "llm_broken_text"
]

# ---------------------------------------------------------------------------
# Scan runner
# ---------------------------------------------------------------------------


def run_scan(
    detectors: Optional[list[str]] = None,
    output_dir: str = "giskard_results",
    dataset_csv: Optional[str] = None,
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    n_adversarial_samples: Optional[int] = None,
    n_requirements: Optional[int] = None,
    model_type: SupportedModelTypes = "text_generation",
) -> None:
    """Run the Giskard vulnerability scan on the Writing Coach model.

    Args:
        detectors: Detector short-names to enable. Defaults to all available.
                   Valid values: ``'harmfulness'``, ``'prompt_injection'``.
        output_dir: Directory where the HTML report, JSON summary, and test
                    suite will be saved.
        dataset_csv: Path to the CSV file to use as the scan dataset.
                     Overrides the GISKARD_DATASET_CSV env var and the
                     default ``final_data/all_results.csv``.
        n_samples: Number of rows to randomly sample from the dataset.
                   If ``None``, all rows are used.
        seed: Random seed for reproducible sampling.
        n_adversarial_samples: Number of adversarial samples (``num_samples``)
                               passed to every selected detector via the
                               ``params`` argument of ``giskard.scan()``.
                               If ``None``, each detector keeps its own default.
        n_requirements: Number of requirements (``n_requirements``)
                          passed to every selected detector via the
                          ``params`` argument of ``giskard.scan()``.
                          If ``None``, each detector keeps its own default.
        model_type: Giskard model type passed to ``giskard.Model``.
                    Default: ``'text_generation'``.
    """
    if detectors is None:
        detectors = VALID_DETECTORS

    unknown = set(detectors) - set(VALID_DETECTORS)
    if unknown:
        raise ValueError(
            f"Unknown detectors: {unknown}. Valid options: {VALID_DETECTORS}"
        )

    # 1. Configure Giskard's internal LLM evaluator
    _configure_giskard_llm_client()

    # 2. Build model and dataset wrappers
    logger.info("Building Giskard model wrapper …")
    gsk_model = build_giskard_model(model_type=model_type)

    sample_docs = _load_sample_documents(
        Path(dataset_csv) if dataset_csv else None,
        n_samples=n_samples,
        seed=seed,
    )
    logger.info("Building Giskard dataset (%d samples) …", len(sample_docs))
    gsk_dataset = build_giskard_dataset(sample_docs)

    # 3. Instantiate detector objects
    logger.info("Loading detectors: %s", detectors)

    logger.info("Initializing WritingCoach graph")
    _init_wc_graph()

    # 4. Run scan
    logger.info("Starting Giskard scan …")

    # Build detector params: for every selected detector, introspect its
    # __init__ signature (including inherited params via inspect.signature,
    # which follows the MRO) to decide which keys to forward.
    detector_params: dict = {}
    if n_adversarial_samples is not None or n_requirements is not None:
        for label, cls in DetectorRegistry.get_detector_classes(detectors).items():
            try:
                init_params = set(inspect.signature(cls.__init__).parameters.keys())
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Could not inspect __init__ for detector '%s' (%s): %s — skipping params",
                    label, cls.__name__, exc,
                )
                continue

            det_cfg: dict = {}
            if n_adversarial_samples is not None and "num_samples" in init_params:
                det_cfg["num_samples"] = n_adversarial_samples
            if n_requirements is not None and "num_requirements" in init_params:
                det_cfg["num_requirements"] = n_requirements

            if det_cfg:
                detector_params[label] = det_cfg
                logger.debug(
                    "Detector '%s' (%s): setting params %s",
                    label, cls.__name__, det_cfg,
                )
            else:
                logger.debug(
                    "Detector '%s' (%s) init params %s do not include "
                    "num_samples/num_requirements — no params set",
                    label, cls.__name__, init_params,
                )

        logger.info("Detector params: %s", detector_params)

    if "llm_broken_text" in detectors:
        LLMBrokenTextDetector = _detector_mod.LLMBrokenTextDetector
        broken_text_detector = LLMBrokenTextDetector()
        scan_results: ScanReport = broken_text_detector.run(model=gsk_model, dataset=gsk_dataset)
    else:
        scan_results: ScanReport = scan(
            gsk_model,
            gsk_dataset,
            only=detectors,
            raise_exceptions=False,
            params=detector_params if detector_params else None,
        )

    # 5. Persist results
    _persist_results(scan_results, output_dir, n_adversarial_samples=n_adversarial_samples, n_requirements=n_requirements)

    # 6. Print summary to stdout
    _print_summary(scan_results, n_adversarial_samples=n_adversarial_samples, n_requirements=n_requirements)


# ---------------------------------------------------------------------------
# Result persistence helpers
# ---------------------------------------------------------------------------


def _persist_results(
        scan_results,
        output_dir: str,
        n_adversarial_samples: int = None,
        n_requirements: int = None,
) -> None:
    """Save the HTML report, JSON summary, and Python test suite to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # HTML report
    html_path = out_path / "scan_report.html"
    scan_results.to_html(str(html_path))
    logger.info("HTML report saved → %s", html_path)

    # JSON summary
    _save_json_summary(scan_results, out_path, n_adversarial_samples=n_adversarial_samples, n_requirements=n_requirements)

    # Test suite
    try:
        test_suite = scan_results.generate_test_suite("Writing Coach Security Tests")
        suite_dir = out_path / "test_suite"
        test_suite.save(str(suite_dir))
        logger.info("Test suite saved → %s", suite_dir)
    except Exception as exc:
        logger.warning("Could not generate test suite: %s", exc)


def example_to_dict(example: dict) -> dict:
    """Parse a conversation example record into structured fields.

    The ``input`` column of a Giskard scan example contains a string of the form::

        USER: {'document': '...', 'user_command': '...'}

        AGENT: <free text>

        SUGGESTIONS:
        - {'original_text': '...', ...}

    This function extracts the three sections and returns a dict with keys
    ``user``, ``agent``, and ``suggestions``, where ``user`` is the parsed
    input dict, ``agent`` is the plain text response string, and
    ``suggestions`` is a list of parsed suggestion dicts.
    """
    import ast
    import re

    raw: str = example.get("input", "") if isinstance(example, dict) else str(example)

    # Split on the three known section tags.
    # We allow optional whitespace / newlines around the tags.
    user_match = re.search(r"USER:\s*", raw)
    agent_match = re.search(r"\n\nAGENT:\s*", raw)
    suggestions_match = re.search(r"\n\n## SUGGESTIONS:\s*\n", raw)

    # --- USER block ---
    user_parsed = {}
    if user_match:
        start = user_match.end()
        end = agent_match.start() if agent_match else (suggestions_match.start() if suggestions_match else len(raw))
        user_str = raw[start:end].strip()
        try:
            user_parsed = ast.literal_eval(user_str)
        except (ValueError, SyntaxError):
            user_parsed = user_str

    # --- AGENT block ---
    agent_parsed = ""
    if agent_match:
        start = agent_match.end()
        end = suggestions_match.start() if suggestions_match else len(raw)
        agent_parsed = raw[start:end].strip()

    # --- SUGGESTIONS block ---
    suggestions_parsed = []
    if suggestions_match:
        start = suggestions_match.end()
        suggestions_str = raw[start:].strip()
        # Each suggestion is on a line starting with "- "
        for line in re.split(r"\n-\s*", suggestions_str):
            line = line.lstrip("- ").strip()
            if not line:
                continue
            try:
                suggestions_parsed.append(ast.literal_eval(line))
            except (ValueError, SyntaxError):
                suggestions_parsed.append(line)

    return {
        "input_text": user_parsed["document"],
        "user_command": user_parsed["user_command"],
        "agent": agent_parsed,
        "suggestions": suggestions_parsed,
    }


def _save_json_summary(
        scan_results,
        out_path: Path,
        n_adversarial_samples: int = None,
        n_requirements: int = None,
) -> None:
    """Serialise the scan issue list to a JSON file."""
    issues = []
    for issue in scan_results.issues:
        issue_examples = []
        if issue.scan_examples is not None:
            examples = issue.scan_examples.examples.to_dict(orient="records")
            for example in examples:
                reason = example.get("Reason", "")
                system = example_to_dict(example["Conversation"])
                issue_examples.append({
                    "reason": reason,
                    "system": system
                })
        issues.append(
            {
                "detector": getattr(issue, "detector_name", type(issue).__name__),
                "group": getattr(issue, "group", ""),
                "level": getattr(issue, "level", ""),
                "description": getattr(issue, "description", ""),
                "meta": getattr(issue, "meta", {}),
                "examples": issue_examples
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
    logger.info("JSON summary saved → %s", json_path)


def _print_summary(
        scan_results,
        n_adversarial_samples: int = None,
        n_requirements: int = None,
) -> None:
    """Print a human-readable scan summary to stdout."""
    print("\n" + "=" * 64)
    print("  GISKARD SCAN SUMMARY — Writing Coach AI")
    print("=" * 64)

    print(" Number of adversarial samples per detector: %s" % (n_adversarial_samples or "default"))
    print(" Number of requirements per detector:        %s" % (n_requirements or "default"))
    print("-" * 64)

    if not scan_results.has_issues():
        print("✅  No vulnerabilities detected.")
    else:
        n = len(scan_results.issues)
        print(f"⚠️   {n} issue(s) detected:\n")
        for i, issue in enumerate(scan_results.issues, start=1):
            detector = getattr(issue, "detector_name", type(issue).__name__)
            level = str(getattr(issue, "level", "unknown")).upper()
            description = getattr(issue, "description", "(no description)")
            print(f"  [{i}] [{level}] {detector}")
            print(f"       {description}\n")

    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.giskard_scan",
        description="Run Giskard security scan on the Writing Coach AI assistant.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.evaluation.giskard_scan\n"
            "  python -m src.evaluation.giskard_scan --detectors harmfulness\n"
            "  python -m src.evaluation.giskard_scan --detectors harmfulness prompt_injection\n"
            "  python -m src.evaluation.giskard_scan --output-dir results/my_scan\n"
        ),
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=VALID_DETECTORS,
        default=VALID_DETECTORS,
        metavar="DETECTOR",
        help=(
            "Which detectors to run. "
            f"Choices: {VALID_DETECTORS}. "
            "Default: all detectors."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="giskard_results",
        metavar="DIR",
        help=(
            "Directory to write the HTML report, JSON summary, and test suite. "
            "Default: giskard_results/"
        ),
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        metavar="CSV",
        help=(
            "Path to the CSV dataset file (must have 'query' and 'input' columns). "
            f"Default: {_DEFAULT_DATASET_CSV}"
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        metavar="N",
        help="Randomly sample N rows from the dataset. Default: use all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducible sampling (used with --n-samples).",
    )
    parser.add_argument(
        "--n-adversarial-samples",
        type=int,
        default=None,
        metavar="N",
        dest="n_adversarial_samples",
        help=(
            "Number of adversarial samples (num_samples) passed to every "
            "selected detector. If omitted, each detector uses its own default."
        ),
    )
    parser.add_argument(
        "--n-requirements",
        type=int,
        default=None,
        metavar="N",
        dest="n_requirements",
        help=(
            "Number of requirements (n_requirements) passed to every "
            "selected detector. If omitted, each detector uses its own default."
        ),
    )
    parser.add_argument(
        "--model-type",
        default="text_generation",
        choices=[m.value for m in SupportedModelTypes],
        metavar="TYPE",
        dest="model_type",
        help=(
            "Giskard model type passed to giskard.Model. "
            f"Choices: {[m.value for m in SupportedModelTypes]}. "
            "Default: text_generation."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Validate required env vars before doing any work
    missing = [
        v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
        if not os.getenv(v)
    ]
    if missing:
        logger.error(
            "Missing required environment variables: %s\n"
            "Set them in a .env file or export them before running the scan.",
            ", ".join(missing),
        )
        sys.exit(1)

    run_scan(
        detectors=args.detectors,
        output_dir=args.output_dir,
        dataset_csv=args.dataset_csv,
        n_samples=args.n_samples,
        seed=args.seed,
        n_adversarial_samples=args.n_adversarial_samples,
        n_requirements=args.n_requirements,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
