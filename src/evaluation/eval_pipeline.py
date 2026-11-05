"""Dynamic Rubrics Evaluation Pipeline for Writing Coach V2.

Two-stage async LLM pipeline:
  Stage 1 — Rubrics Generator: produces rubric items from (query, input, route)
  Stage 2 — Rubrics Judge: scores each rubric item against the system output

Uses LangChain AzureChatOpenAI with built-in retry, writes results incrementally
as CSV + JSONL for resume support.

Usage:
  python -m src.evaluation.eval_pipeline --input final_data/all_results.csv --limit 5
  python -m src.evaluation.eval_pipeline --routes RESEARCH RESPOND --limit 10
"""

import asyncio
import csv
import json
import logging
import os
import re
import argparse
import time
import statistics
import mlflow
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from collections import Counter  # noqa: E402

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.evaluation.prompt_loader import (
    build_generator_prompts,
    build_judge_prompts,
    format_metrics_definition,
)
from src.constants.metrics_definitions import (
    CORRECTNESS_METRIC_NAME,
    METRICS_DEFINITION,
    SCORING_METRIC_NAMES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

logger = logging.getLogger("eval_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("langsmith").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)

DEFAULT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_CONCURRENCY = 5
MAX_RETRIES = 3
DEFAULT_INPUT = "final_data/all_results.csv"

EVAL_CSV_FIELDNAMES = [
    "row_id",
    "route",
    "status",
    "timestamp",
    "output_relevancy_score",
    "completeness_score",
    "correctness_score",
    "overall_notes",
    "generator_raw_response",
    "rubrics_json",
    "rubrics_reasoning",
    "evaluation_items_json",
    "judge_raw_response",
    "verdicts_json",
]

METRIC_SCORE_FIELDS = {
    "output relevancy": "output_relevancy_score",
    "completeness": "completeness_score",
    "correctness": "correctness_score",
}

SCORE_FIELD_LABELS = {
    "output_relevancy_score": "Output Relevancy",
    "completeness_score": "Completeness",
    "correctness_score": CORRECTNESS_METRIC_NAME,
}

# Short keys used to DRY metric loops in stats/summary code
METRIC_KEYS = ("output_relevancy", "completeness", "correctness")

# SCORING_METRIC_NAMES is imported from src.constants.metrics_definitions
# (single source of truth) and used by the --metrics CLI argument below.

NON_RESPONSE_STATUS = "NO_RESPONSE"

REMOVAL_REQUEST_RE = re.compile(
    r"\b(remove|delete|delet(e|ion)|omit|drop|cut|trim|erase|exclude|eliminate|strip out|take out)\b"
    r"|\b(blank lines?|empty lines?|line breaks?|newlines?)\b",
    re.IGNORECASE,
)

NON_RESPONSE_SPECIFICITY_RE = re.compile(
    r"\b(could you|please)\s+be more specific\b",
    re.IGNORECASE,
)

NON_RESPONSE_FAILURE_RE = re.compile(
    r"\b(could(?:n't| not)|did(?:n't| not)|unable to)\s+"
    r"(?:identify|find)\b[^\n.?!]*\b(improvements?|changes?|edits?)\b",
    re.IGNORECASE,
)


def _is_removal_request(query: str) -> bool:
    """Return True when the user command is primarily asking to remove content."""
    return bool(REMOVAL_REQUEST_RE.search(query or ""))


def _is_non_response_output(response_text: str, suggestions: list[dict[str, Any]] | None = None) -> bool:
    """Return True for fallback outputs that ask the user for more specific instructions.

    These cases are system non-responses rather than meaningful outputs and
    should be excluded from scoring.
    """
    if suggestions:
        return False

    text = (response_text or "").strip()
    if not text:
        return False

    lowered = text.lower()
    if "i analyzed your text" in lowered and NON_RESPONSE_SPECIFICITY_RE.search(text):
        return True

    return bool(
        NON_RESPONSE_SPECIFICITY_RE.search(text)
        and NON_RESPONSE_FAILURE_RE.search(text)
    )


def _suggestion_has_payload(suggestion: dict[str, Any]) -> bool:
    """Return True when a suggestion contains enough data to be judged.

    Empty ``transformed_text`` is still meaningful for deletions, so suggestions
    are retained whenever they include any textual field, explanation, or span.
    """
    if not isinstance(suggestion, dict):
        return False
    return (
        suggestion.get("original_text") is not None
        or suggestion.get("transformed_text") is not None
        or str(suggestion.get("explanation") or "").strip() != ""
        or (isinstance(suggestion.get("char_start"), int) and isinstance(suggestion.get("char_end"), int))
    )


def _suggestion_has_nonempty_transformed_text(suggestion: dict[str, Any]) -> bool:
    """Return True when ``transformed_text`` is present and not the empty string."""
    transformed = suggestion.get("transformed_text")
    return isinstance(transformed, str) and transformed != ""


def _normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs into single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _is_whitespace_only_edit(original: str, transformed: str) -> bool:
    """Return True when the only difference between original and transformed is whitespace."""
    return (
        bool(original) and bool(transformed)
        and _normalize_whitespace(original) == _normalize_whitespace(transformed)
    )


def _is_removal_within_context(original: str, transformed: str) -> bool:
    """Return True when transformed is original with some content removed.

    Checks that every word in the transformed text appears in the original in
    the same order (i.e., transformed is a word-level subsequence of original)
    and that it is meaningfully shorter.
    """
    if not original or not transformed:
        return False
    if len(transformed) >= len(original):
        return False
    orig_words = original.split()
    trans_words = transformed.split()
    if len(trans_words) >= len(orig_words):
        return False
    oi = 0
    for tw in trans_words:
        while oi < len(orig_words) and orig_words[oi] != tw:
            oi += 1
        if oi >= len(orig_words):
            return False
        oi += 1
    return True


def _is_sentence_removed_from_context(original: str, transformed: str) -> bool:
    """Return True when original is a sentence/phrase removed from a longer context.

    Pattern: original (shorter) does not appear in transformed (longer), meaning
    the original text was excised from the surrounding paragraph that is shown
    as the proposed revision.
    """
    if not original or not transformed:
        return False
    if len(original) >= len(transformed):
        return False
    # The removed sentence should not appear in the result
    original_norm = _normalize_whitespace(original)
    transformed_norm = _normalize_whitespace(transformed)
    return original_norm not in transformed_norm


def _compute_removed_text(original: str, transformed: str) -> str:
    """Return the text segments present in original but absent in transformed."""
    import difflib
    orig_words = original.split()
    trans_words = transformed.split()
    sm = difflib.SequenceMatcher(None, orig_words, trans_words)
    removed_parts: list[str] = []
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            removed_parts.append(" ".join(orig_words[i1:i2]))
    return " [...] ".join(removed_parts) if removed_parts else ""


def _classify_suggestion_operation(suggestion: dict[str, Any], is_removal_request: bool = False) -> str:
    """Classify the suggestion as insertion, deletion, removal, whitespace_edit, replacement, or generic edit."""
    original = suggestion.get("original_text")
    transformed = suggestion.get("transformed_text")

    has_original = isinstance(original, str) and original != ""
    has_transformed = isinstance(transformed, str) and transformed != ""

    if has_original and not has_transformed:
        return "deletion"
    if not has_original and has_transformed:
        return "insertion"
    if has_original and has_transformed:
        if _is_whitespace_only_edit(original, transformed):
            return "whitespace_edit"
        if _is_removal_within_context(original, transformed):
            return "removal"
        # Pattern B: original is the sentence removed; transformed is the
        # surrounding paragraph without it (transformed > original).
        if is_removal_request and _is_sentence_removed_from_context(original, transformed):
            return "removal"
        return "replacement"
    if isinstance(suggestion.get("char_start"), int) and isinstance(suggestion.get("char_end"), int):
        if suggestion["char_end"] > suggestion["char_start"] and not has_transformed:
            return "deletion"
    return "edit"


def _format_text_for_judge(text: Any, empty_label: str) -> str:
    """Render text so empty and whitespace-only edits remain visible to the judge."""
    if text is None:
        return empty_label
    if not isinstance(text, str):
        text = str(text)
    if text == "":
        return empty_label
    if text.strip() == "":
        return f"(whitespace-only text; literal={json.dumps(text)})"
    return text


def _should_include_correctness(response_route: str, has_suggestions: bool) -> bool:
    """Return True when Correctness should be evaluated for the actual response route."""
    return response_route.upper().startswith("REVISE") and has_suggestions


def _should_force_zero_correctness(row: dict[str, Any]) -> bool:
    """Flag clearly invalid empty revisions for deterministic Correctness=0.

    Returns True when all retained suggestions have empty transformed_text
    AND the query is not a removal request.
    """
    if bool(row.get("is_removal_request")):
        return False
    return bool(row.get("has_suggestions")) and not bool(row.get("has_nonempty_transformed_text"))

# ---------------------------------------------------------------------------
# LangChain Azure OpenAI model (cached per parameter set)
# ---------------------------------------------------------------------------

_model: AzureChatOpenAI | None = None
_model_key: tuple | None = None


def get_model(
    deployment: str = DEFAULT_DEPLOYMENT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AzureChatOpenAI:
    """Get or create a cached AzureChatOpenAI model (re-creates when params change)."""
    global _model, _model_key
    key = (deployment, temperature, max_tokens)
    if _model is None or _model_key != key:
        _model = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=MAX_RETRIES,
        )
        _model_key = key
        logger.info(
            "Created AzureChatOpenAI model: deployment=%s, temperature=%.2f, max_tokens=%d",
            deployment, temperature, max_tokens,
        )
    return _model


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def call_llm(
    deployment: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Call Azure OpenAI via LangChain and return the text response."""
    model = get_model(deployment, temperature, max_tokens)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = await model.ainvoke(messages)
    return response.content


def parse_json_response(text: str) -> dict | None:
    """Parse a JSON object from an LLM response.

    Strips markdown code fences and falls back to ``parse_markdown_rubrics``.
    Returns None if all parsing strategies fail.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: try to parse the structured Markdown rubrics format
    return parse_markdown_rubrics(text)


def parse_markdown_rubrics(text: str) -> dict | None:
    """Parse the structured Markdown rubrics format (``### Criterion N`` / ``### Metric Rubric N``).

    Returns the normalized ``{"rubrics": [...]}`` dict, or None if not rubrics Markdown.
    """
    header_pattern = r"###\s+(?:Criterion|Metric Rubric)\s+\d+:\s*(.+)"
    rubric_headers = re.findall(header_pattern, text, re.IGNORECASE)
    if not rubric_headers:
        return None

    rubrics = []
    blocks = re.split(
        r"(?=###\s+(?:Criterion|Metric Rubric)\s+\d+:)",
        text,
        flags=re.IGNORECASE,
    )

    for block in blocks:
        header_match = re.match(header_pattern, block, re.IGNORECASE)
        if not header_match:
            continue

        rubric_name = header_match.group(1).strip().strip("*")

        description = ""
        desc_match = re.search(r"\*\*Description\*\*:\s*(.+?)(?=\n\*\*|\n###|$)", block, re.DOTALL)
        if desc_match:
            description = desc_match.group(1).strip()

        linked_to = ""
        linked_match = re.search(r"\*\*Linked to\*\*:\s*(.+)", block)
        if linked_match:
            linked_to = linked_match.group(1).strip()

        weight = ""
        weight_match = re.search(r"\*\*(?:Weight|Metric Importance)\*\*:\s*(.+)", block)
        if weight_match:
            weight = weight_match.group(1).strip()

        evaluation_items: list[dict[str, str]] = []
        items_match = re.search(
            r"\*\*Evaluation Items\*\*:\s*(.+?)(?=\n\|\s*Level\s*\||\n###|$)",
            block,
            re.DOTALL,
        )
        if items_match:
            items_block = items_match.group(1)
            for line in items_block.splitlines():
                stripped = line.strip()
                if not stripped.startswith("-"):
                    continue

                item_text = stripped[1:].strip()
                item_importance = ""
                importance_match = re.search(r"\(Importance:\s*([^)]+)\)", item_text)
                if importance_match:
                    item_importance = importance_match.group(1).strip()
                    item_text = re.sub(r"\s*\(Importance:\s*[^)]+\)[.,;:]?\s*$", "", item_text).strip()

                evaluation_items.append({
                    "item": item_text,
                    "importance": item_importance,
                })

        levels = []
        table_rows = re.findall(
            r"\|\s*\*{0,2}([^|]+?)\*{0,2}\s*\|\s*(\d+)\s*\|([^|]+)\|([^|]+)\|",
            block,
            re.IGNORECASE,
        )
        for level, score, desc, indicators in table_rows:
            if level.strip().lower() == "level":
                continue
            levels.append({
                "level": level.strip(),
                "score": int(score.strip()),
                "description": desc.strip(),
                "indicators": indicators.strip(),
            })

        rubrics.append({
            "metric": rubric_name,
            "description": description,
            "linked_to": linked_to,
            "metric_importance": weight,
            "evaluation_items": evaluation_items,
            "levels": levels,
        })

    if not rubrics:
        return None

    reasoning = ""
    reasoning_match = re.search(
        r"###\s+Overall Assessment Guidelines(.+?)(?=##|$)", text, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    logger.debug("parse_markdown_rubrics: extracted %d rubrics", len(rubrics))
    return {"rubrics": rubrics, "reasoning": reasoning}


def _normalize_metric_name(name: str | None) -> str:
    """Normalize metric names for reliable matching across prompt outputs."""
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip().lower()


def _extract_metric_scores(evaluation: list[dict[str, Any]]) -> dict[str, int | None]:
    """Extract per-metric 0-2 scores from judge evaluation entries."""
    scores = {field_name: None for field_name in METRIC_SCORE_FIELDS.values()}

    for verdict in evaluation:
        metric_name = verdict.get("metric_name") or verdict.get("criterion_name")
        field_name = METRIC_SCORE_FIELDS.get(_normalize_metric_name(metric_name))
        if field_name is None:
            continue

        score = verdict.get("score")
        if isinstance(score, int):
            scores[field_name] = score

    return scores


def _required_score_fields(
    include_correctness: bool = False,
    metrics: list[str] | None = None,
) -> list[str]:
    """Return the metric score fields expected for a given row.

    When *metrics* is provided, only fields for the selected metrics are required.
    """
    if metrics is not None:
        normalized = {m.lower().replace(" ", "_") for m in metrics}
        all_fields = [
            ("output_relevancy_score", "output_relevancy"),
            ("completeness_score", "completeness"),
            ("correctness_score", "correctness"),
        ]
        required = [
            field
            for field, key in all_fields
            if key in normalized and (field != "correctness_score" or include_correctness)
        ]
        return required
    required = ["output_relevancy_score", "completeness_score"]
    if include_correctness:
        required.append("correctness_score")
    return required


def _avg_std(scores: list) -> tuple[float, float]:
    """Return (mean, stdev) of scores, or (nan, 0.0) for an empty list."""
    if not scores:
        return float("nan"), 0.0
    avg = sum(scores) / len(scores)
    return avg, (statistics.stdev(scores) if len(scores) >= 2 else 0.0)


def _score_distribution(scores: list[int]) -> dict[int, dict[str, float | int]]:
    """Return count and percentage for each discrete score value 0, 1, and 2."""
    total = len(scores)
    counts = Counter(score for score in scores if score in (0, 1, 2))
    return {
        score: {
            "count": counts.get(score, 0),
            "pct": (counts.get(score, 0) / total * 100.0) if total else 0.0,
        }
        for score in (0, 1, 2)
    }


def _compute_score_stats(ok_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-route, micro, and macro score statistics from OK-status results."""
    route_buckets: dict[str, dict[str, list]] = {}
    for r in ok_results:
        b = route_buckets.setdefault(
            r["route"],
            {"row_count": 0, **{field: [] for field in SCORE_FIELD_LABELS}},
        )
        b["row_count"] += 1
        for field in SCORE_FIELD_LABELS:
            if r.get(field) is not None:
                b[field].append(r[field])

    per_route = {
        route: {
            "n": b["row_count"],
            "counts": {field: len(b[field]) for field in SCORE_FIELD_LABELS},
            **{mk: _avg_std(b[f"{mk}_score"]) for mk in METRIC_KEYS},
            "distributions": {f"{mk}_score": _score_distribution(b[f"{mk}_score"]) for mk in METRIC_KEYS},
        }
        for route, b in route_buckets.items()
    }
    all_scores = {mk: [s for b in route_buckets.values() for s in b[f"{mk}_score"]] for mk in METRIC_KEYS}
    route_avgs = {mk: [_avg_std(b[f"{mk}_score"])[0] for b in route_buckets.values() if b[f"{mk}_score"]] for mk in METRIC_KEYS}
    return {
        "per_route": per_route,
        "micro": {
            "n": len(ok_results),
            "counts": {f"{mk}_score": len(all_scores[mk]) for mk in METRIC_KEYS},
            **{mk: _avg_std(all_scores[mk]) for mk in METRIC_KEYS},
            "distributions": {f"{mk}_score": _score_distribution(all_scores[mk]) for mk in METRIC_KEYS},
        },
        "macro": {
            "n_routes": len(route_buckets),
            "counts": {f"{mk}_score": len(route_avgs[mk]) for mk in METRIC_KEYS},
            **{mk: _avg_std(route_avgs[mk]) for mk in METRIC_KEYS},
        },
    }


def _format_distribution(distribution: dict[int, dict[str, float | int]]) -> str:
    """Render a score distribution as a compact 0/1/2 percentage summary."""
    return "  ".join(
        f"{score}={distribution[score]['pct']:.1f}% (n={distribution[score]['count']})"
        for score in (0, 1, 2)
    )


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_processed_ids(details_jsonl: Path) -> set:
    """Return set of row_ids already written to the JSONL output file."""
    processed = set()
    if details_jsonl.exists():
        with open(details_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed.add(json.loads(line)["row_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return processed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_input_data(
    input_path: str,
    routes: list[str] | None = None,
    limit: int | None = None,
    route_column: str = "intended",
    data_origin: str = "all",
) -> list[dict[str, Any]]:
    """Load and filter the input dataset.

    Parses the ``output`` JSON column to extract response_text, suggestions,
    and derived flags (has_suggestions, is_removal_request, is_non_response).
    """
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    # Determine which route column to use
    if route_column == "intended":
        preferred_col = "route_intended"
        fallback_col = "route_orch"
    else:  # orchestrator
        preferred_col = "route_orch"
        fallback_col = "route_intended"

    # Use preferred column if present, otherwise fall back
    if preferred_col in df.columns:
        route_col = preferred_col
    elif fallback_col in df.columns:
        logger.info("Preferred column '%s' not found, falling back to '%s'", preferred_col, fallback_col)
        route_col = fallback_col
    else:
        logger.error("Neither '%s' nor '%s' column found in input CSV", preferred_col, fallback_col)
        return []

    if routes:
        routes_upper = [r.upper() for r in routes]
        df = df[df[route_col].str.upper().isin(routes_upper)]
        logger.info("Filtered to routes %s: %d rows", routes_upper, len(df))

    if data_origin != "all":
        if "dataset_source" not in df.columns:
            logger.warning("'dataset_source' column not found — data_origin filter '%s' ignored", data_origin)
        else:
            is_synthetic = (
                df["dataset_source"].str.startswith("Synthetic", na=False)
                | (df["dataset_source"] == "extra_respond_alex")
            )
            if data_origin == "synthetic":
                df = df[is_synthetic]
            else:  # natural
                df = df[~is_synthetic]
            logger.info("Filtered to data_origin='%s': %d rows", data_origin, len(df))

    if limit:
        df = df.head(limit)

    rows = []
    for _, row in df.iterrows():
        output_data: dict = {}
        raw_output = row.get("output", "")
        if pd.notna(raw_output) and raw_output:
            try:
                output_data = json.loads(str(raw_output))
            except (json.JSONDecodeError, ValueError):
                logger.warning("Row %s: Failed to parse output JSON", row.get("row_id"))

        response_text = str(output_data.get("response", "")).strip() if output_data else ""

        raw_suggestions = output_data.get("suggestions", []) if output_data else []
        suggestions = [
            s for s in (raw_suggestions or [])
            if _suggestion_has_payload(s)
        ]
        has_nonempty_transformed_text = any(
            _suggestion_has_nonempty_transformed_text(s) for s in suggestions
        )
        response_route = str(
            output_data.get("route")
            or row.get("route_orch")
            or row.get(route_col, "UNKNOWN")
        )

        rows.append({
            "row_id": int(row["row_id"]),
            "query": str(row.get("query", "")),
            "input": str(row.get("input", "")),
            "route": str(row.get(route_col, "UNKNOWN")),
            "response_route": response_route,
            "response_text": response_text,
            "suggestions": suggestions,
            "has_suggestions": len(suggestions) > 0,
            "has_nonempty_transformed_text": has_nonempty_transformed_text,
            "is_removal_request": _is_removal_request(str(row.get("query", ""))),
            "is_non_response": _is_non_response_output(response_text, suggestions),
        })

    logger.info("Loaded %d rows from %s", len(rows), input_path)
    return rows


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------


def _format_output_for_judge(
    response_text: str,
    suggestions: list[dict],
    is_removal_request: bool = False,
) -> str:
    """Format the writing coach output for the judge LLM.

    For non-suggestion routes returns response_text directly.
    For REVISE routes returns a structured block with numbered suggestions.
    """
    if not suggestions:
        return response_text

    parts: list[str] = []

    if response_text.strip():
        parts.append("### Writing Coach Note\n" + response_text.strip())

    parts.append(
        f"### Revision Suggestions ({len(suggestions)} total)\n"
        "Each suggestion shows the original passage and the proposed revision. "
        "The user can accept or reject each suggestion independently. "
        "An empty proposed revision means delete the original text/span entirely. "
        "Whitespace-only text is shown with an escaped literal so blank-line edits remain visible."
    )

    for i, s in enumerate(suggestions, 1):
        original = s.get("original_text")
        transformed = s.get("transformed_text")
        explanation = (s.get("explanation") or "").strip()
        char_start = s.get("char_start")
        char_end = s.get("char_end")
        operation = _classify_suggestion_operation(s, is_removal_request=is_removal_request)

        block_lines = [f"#### Suggestion {i}"]
        block_lines.append(f"**Operation Type**: {operation}")

        # Add annotations for operations the judge typically misreads
        if operation == "removal":
            orig_str = str(original or "")
            trans_str = str(transformed or "")
            if len(orig_str) > len(trans_str):
                # Pattern A: original is the paragraph, transformed is paragraph minus sentence
                removed = _compute_removed_text(orig_str, trans_str)
                if removed:
                    block_lines.append(
                        f"**Removed Content**: The following text was removed from the "
                        f"original passage while keeping the surrounding context intact:\n{removed}"
                    )
            else:
                # Pattern B: original is the sentence removed; transformed is surrounding context without it
                block_lines.append(
                    f"**Removed Content**: The original text shown below was removed from "
                    f"the paragraph. The proposed revision shows the surrounding paragraph "
                    f"after this sentence/phrase was excised. The remaining text in the "
                    f"proposed revision is NOT new content — it already exists in the document."
                )
        elif operation == "whitespace_edit":
            block_lines.append(
                "**Whitespace-Only Edit**: The text content is identical; only "
                "whitespace has changed (e.g., empty lines removed, line breaks "
                "consolidated). This is a formatting edit, not a content change."
            )

        if explanation:
            block_lines.append(f"**Purpose**: {explanation}")
        if isinstance(char_start, int) and isinstance(char_end, int):
            block_lines.append(f"**Span**: char_start={char_start}, char_end={char_end}")
        if original is None:
            block_lines.append("**Original Text**: (not provided)")
        else:
            block_lines.append(
                "**Original Text**:\n"
                + _format_text_for_judge(
                    original,
                    "(empty string — insertion or empty selected span)",
                )
            )
        block_lines.append(
            "**Proposed Revision**:\n"
            + _format_text_for_judge(
                transformed,
                "(empty string — delete the original text)",
            )
        )

        parts.append("\n".join(block_lines))

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core pipeline: process a single row
# ---------------------------------------------------------------------------


async def process_row(
    row: dict[str, Any],
    deployment: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    rubrics_mode: str = "combined",
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Run Stage 1 (generate rubrics) and Stage 2 (judge) for one row.

    When *metrics* is provided, only the specified metrics are evaluated.
    Accepted row-level values: 'completeness', 'output_relevancy', 'correctness'.
    """
    row_id = row["row_id"]
    result = {
        "row_id": row_id,
        "route": row["route"],
        "status": "OK",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Stage 1 outputs
        "generator_raw_response": None,
        "rubrics": None,
        "rubrics_reasoning": None,
        # Stage 2 outputs
        "judge_raw_response": None,
        "verdicts": None,
        "overall_notes": None,
        # Scores
        "output_relevancy_score": None,
        "completeness_score": None,
        "correctness_score": None,
    }

    async with semaphore:
        try:
            logger.info("Row %d: processing (route=%s)", row_id, row["route"])

            if row.get("is_non_response"):
                logger.info("Row %d: excluded from scoring due to non-response fallback output", row_id)
                result["status"] = NON_RESPONSE_STATUS
                result["overall_notes"] = (
                    "Excluded from evaluation because the system returned a fallback "
                    "request for more specific instructions instead of a meaningful response."
                )
                return result

            # ---- Stage 1: Generate Rubrics ----
            # Build the active subset of prompt-level metrics (excludes Correctness,
            # which uses fixed rubrics and is handled separately in Stage 2).
            _metric_name_map = {
                "output_relevancy": "Output Relevancy",
                "completeness": "Completeness",
            }
            if metrics is not None:
                _normalized_selected = {m.lower().replace(" ", "_") for m in metrics}
                _active_metrics = {
                    display_name: METRICS_DEFINITION[display_name]
                    for key, display_name in _metric_name_map.items()
                    if key in _normalized_selected and display_name in METRICS_DEFINITION
                }
            else:
                _active_metrics = dict(METRICS_DEFINITION)

            if rubrics_mode == "split":
                # Two separate LLM calls, one per metric, then merge rubrics.
                _split_configs = []
                if "Output Relevancy" in _active_metrics:
                    _split_configs.append(
                        (
                            format_metrics_definition({"Output Relevancy": _active_metrics["Output Relevancy"]}),
                            "rubrics_prompt_output_relevancy.txt",
                            "output_relevancy",
                        )
                    )
                if "Completeness" in _active_metrics:
                    _split_configs.append(
                        (
                            format_metrics_definition({"Completeness": _active_metrics["Completeness"]}),
                            "rubrics_prompt_completeness.txt",
                            "completeness",
                        )
                    )

                if not _split_configs:
                    # Only Correctness selected — no generator prompts needed for split mode
                    rubrics = []
                    result["rubrics_reasoning"] = ""
                else:
                    _gen_calls = [
                        build_generator_prompts(
                            user_query=row["query"],
                            input_text=row["input"],
                            route=row["route"],
                            metrics_definition=_metrics_def,
                            prompt_file=_prompt_file,
                        )
                        for _metrics_def, _prompt_file, _ in _split_configs
                    ]

                    _gen_responses = await asyncio.gather(
                        *[call_llm(deployment, sys_p, usr_p, temperature, max_tokens) for sys_p, usr_p in _gen_calls]
                    )

                    _gen_raw = {cfg[2]: resp for cfg, resp in zip(_split_configs, _gen_responses)}
                    result["generator_raw_response"] = json.dumps(_gen_raw, ensure_ascii=False)

                    _gen_parsed_map: dict[str, dict] = {}
                    for _key, _resp in _gen_raw.items():
                        _parsed = parse_json_response(_resp)
                        if _parsed is None:
                            logger.warning(
                                "Row %d: Failed to parse generator response (%s)", row_id, _key
                            )
                            result["status"] = "GENERATOR_PARSE_ERROR"
                            return result
                        _gen_parsed_map[_key] = _parsed

                    rubrics = []
                    for _parsed in _gen_parsed_map.values():
                        rubrics.extend(_parsed.get("rubrics", []))

                    result["rubrics_reasoning"] = json.dumps(
                        {k: v.get("reasoning", "") for k, v in _gen_parsed_map.items()},
                        ensure_ascii=False,
                    )
            else:
                # combined mode: single LLM call with selected metrics
                if not _active_metrics:
                    # Only Correctness selected — skip generator step
                    rubrics = []
                    result["rubrics_reasoning"] = ""
                else:
                    _combined_metrics_def = format_metrics_definition(_active_metrics)
                    gen_system, gen_user = build_generator_prompts(
                        user_query=row["query"],
                        input_text=row["input"],
                        route=row["route"],
                        metrics_definition=_combined_metrics_def,
                    )
                    gen_response = await call_llm(
                        deployment=deployment,
                        system_prompt=gen_system,
                        user_prompt=gen_user,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    result["generator_raw_response"] = gen_response

                    gen_parsed = parse_json_response(gen_response)
                    if gen_parsed is None:
                        logger.warning("Row %d: Failed to parse generator response", row_id)
                        result["status"] = "GENERATOR_PARSE_ERROR"
                        return result

                    rubrics = gen_parsed.get("rubrics", [])
                    result["rubrics_reasoning"] = gen_parsed.get("reasoning", "")

            result["rubrics"] = rubrics

            # When only Correctness is selected there are no generated rubrics;
            # skip the empty-rubrics guard in that case.
            _only_correctness = bool(metrics) and all(
                m.lower().replace(" ", "_") == "correctness" for m in metrics
            )
            if not rubrics and not _only_correctness:
                logger.warning("Row %d: Generator produced zero rubrics", row_id)
                result["status"] = "NO_RUBRICS"
                return result

            # Check if we have anything to judge
            if not row["response_text"] and not row.get("has_suggestions"):
                logger.warning("Row %d: No response text or suggestions to judge", row_id)
                result["status"] = "NO_OUTPUT_TEXT"
                return result

            # ---- Stage 2: Judge Rubrics ----

            rubrics_str = json.dumps(rubrics, indent=2, ensure_ascii=False)

            output_for_judge = _format_output_for_judge(
                response_text=row["response_text"],
                suggestions=row.get("suggestions", []),
                is_removal_request=row.get("is_removal_request", False),
            )

            _correctness_selected = (
                metrics is None
                or any(m.lower().replace(" ", "_") == "correctness" for m in metrics)
            )
            include_correctness = _correctness_selected and _should_include_correctness(
                row.get("response_route", ""),
                bool(row.get("has_suggestions")),
            )

            judge_system, judge_user = build_judge_prompts(
                user_query=row["query"],
                input_text=row["input"],
                output_text=output_for_judge,
                rubrics=rubrics_str,
                include_correctness=include_correctness,
            )

            judge_response = await call_llm(
                deployment=deployment,
                system_prompt=judge_system,
                user_prompt=judge_user,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result["judge_raw_response"] = judge_response

            judge_parsed = parse_json_response(judge_response)
            if judge_parsed is None:
                logger.warning("Row %d: Failed to parse judge response", row_id)
                result["status"] = "JUDGE_PARSE_ERROR"
                return result

            # Extract per-metric scores (0-2)
            evaluation = judge_parsed.get("evaluation", [])
            result["verdicts"] = evaluation
            result["overall_notes"] = judge_parsed.get("summary") or (
                judge_parsed.get("overall_assessment", {}).get("summary", "")
            )

            metric_scores = _extract_metric_scores(evaluation)
            result.update(metric_scores)

            if not include_correctness:
                result["correctness_score"] = None

            # Null-out scores for metrics that were not selected
            if metrics is not None:
                _normalized_selected = {m.lower().replace(" ", "_") for m in metrics}
                _field_to_key = {
                    "output_relevancy_score": "output_relevancy",
                    "completeness_score": "completeness",
                    "correctness_score": "correctness",
                }
                for _field, _key in _field_to_key.items():
                    if _key not in _normalized_selected:
                        result[_field] = None

            if _should_force_zero_correctness(row):
                result["correctness_score"] = 0

            required_score_fields = _required_score_fields(
                include_correctness=include_correctness, metrics=metrics
            )
            if any(result.get(field) is None for field in required_score_fields):
                logger.warning("Row %d: Judge response missing one or more metric scores", row_id)
                result["status"] = "JUDGE_INCOMPLETE"
                return result

            logger.info(
                "Row %d: done — Output Relevancy=%s, Completeness=%s, Correctness=%s",
                row_id,
                result["output_relevancy_score"],
                result["completeness_score"],
                result["correctness_score"],
            )

        except Exception as e:
            logger.error("Row %d: Error — %s", row_id, e)
            result["status"] = "ERROR"
            result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Output writing (thread-safe with asyncio.Lock)
# ---------------------------------------------------------------------------


class OutputWriter:
    """Thread-safe incremental writer for CSV + JSONL evaluation outputs."""

    def __init__(self, output_dir: Path, run_name: str):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_csv = output_dir / f"{run_name}_results.csv"
        self.details_jsonl = output_dir / f"{run_name}_details.jsonl"
        self._lock = asyncio.Lock()

        csv_is_new = not self.results_csv.exists() or self.results_csv.stat().st_size == 0
        self._csv_file = open(self.results_csv, "a", newline="", encoding="utf-8")
        self._jsonl_file = open(self.details_jsonl, "a", encoding="utf-8")

        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=EVAL_CSV_FIELDNAMES)
        if csv_is_new:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    async def write_result(self, result: dict[str, Any]) -> None:
        """Write one evaluation result to both CSV and JSONL."""
        async with self._lock:
            self._csv_writer.writerow({
                "row_id": result["row_id"],
                "route": result["route"],
                "status": result["status"],
                "timestamp": result["timestamp"],
                "output_relevancy_score": result["output_relevancy_score"],
                "completeness_score": result["completeness_score"],
                "correctness_score": result["correctness_score"],
                "overall_notes": result["overall_notes"],
                "generator_raw_response": result["generator_raw_response"],
                "rubrics_json": json.dumps(result["rubrics"], ensure_ascii=False)
                    if result["rubrics"] is not None else "",
                "rubrics_reasoning": result["rubrics_reasoning"] if isinstance(result["rubrics_reasoning"], str)
                    else json.dumps(result["rubrics_reasoning"], ensure_ascii=False)
                    if result["rubrics_reasoning"] is not None else "",
                "evaluation_items_json": json.dumps(
                    (
                        [
                            {
                                "metric": v.get("metric_name") or v.get("criterion_name"),
                                "evaluation_items": v.get("evaluation_items", []),
                            }
                            for v in (result["verdicts"] or [])
                        ]
                        if result["verdicts"] is not None
                        else [
                            {"metric": r.get("metric"), "evaluation_items": r.get("evaluation_items", [])}
                            for r in (result["rubrics"] or [])
                        ]
                    ),
                    ensure_ascii=False,
                ) if result["rubrics"] is not None else "",
                "judge_raw_response": result["judge_raw_response"],
                "verdicts_json": json.dumps(result["verdicts"], ensure_ascii=False)
                    if result["verdicts"] is not None else "",
            })
            self._csv_file.flush()

            self._jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            self._jsonl_file.flush()

    def close(self) -> None:
        """Close output files."""
        self._csv_file.close()
        self._jsonl_file.close()

    @property
    def details_path(self) -> Path:
        return self.details_jsonl

    @property
    def results_path(self) -> Path:
        return self.results_csv


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(
    input_path: str,
    output_dir: str = "data_outputs/eval",
    deployment: str = DEFAULT_DEPLOYMENT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    concurrency: int = DEFAULT_CONCURRENCY,
    routes: list[str] | None = None,
    limit: int | None = None,
    resume: bool = True,
    run_name: str | None = None,
    rubrics_mode: str = "combined",
    route_column: str = "intended",
    save_local: bool = False,
    data_origin: str = "all",
    metrics: list[str] | None = None,
) -> None:
    """Run the full dynamic rubrics evaluation pipeline.

    When *metrics* is provided, only the selected row-level subset is executed.
    Supported values: 'completeness', 'output_relevancy', 'correctness'.
    """
    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"eval_{ts}"

    logger.info("=" * 80)
    logger.info("DYNAMIC RUBRICS EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info("Deployment:  %s", deployment)
    logger.info("Concurrency: %d", concurrency)
    logger.info("Temperature: %.2f", temperature)
    logger.info("Input:       %s", input_path)
    logger.info("Output dir:  %s", output_dir)
    logger.info("Run name:    %s", run_name)
    logger.info("Rubrics mode: %s", rubrics_mode)
    logger.info("Route column: %s", route_column)
    logger.info("Data origin:  %s", data_origin)
    logger.info("Metrics:      %s", metrics if metrics is not None else "all")
    logger.info("=" * 80)

    # Load data
    rows = load_input_data(input_path, routes=routes, limit=limit, route_column=route_column, data_origin=data_origin)
    if not rows:
        logger.warning("No rows to process. Exiting.")
        return

    # Set up output writer
    writer = OutputWriter(Path(output_dir), run_name)

    # Resume support
    if resume:
        processed_ids = load_processed_ids(writer.details_path)
        if processed_ids:
            before = len(rows)
            rows = [r for r in rows if r["row_id"] not in processed_ids]
            logger.info(
                "Resume: %d rows already processed, %d remaining",
                before - len(rows),
                len(rows),
            )

    if not rows:
        logger.info("All rows already processed. Nothing to do.")
        writer.close()
        return

    # Set up concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    logger.info("Processing %d rows with concurrency=%d...", len(rows), concurrency)
    start_time = time.time()

    # Progress bar over completed queries (advances as each row finishes,
    # regardless of completion order; counts failures too via finally).
    pbar = tqdm(total=len(rows), desc="Evaluating", unit="query")

    # Create tasks for all rows
    async def process_and_write(row: dict) -> dict:
        try:
            result = await process_row(
                row, deployment, temperature, max_tokens, semaphore,
                rubrics_mode=rubrics_mode,
                metrics=metrics,
            )
            await writer.write_result(result)
            return result
        finally:
            pbar.update(1)

    tasks = [process_and_write(row) for row in rows]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        pbar.close()

    elapsed = time.time() - start_time

    # Handle any exceptions from gather
    actual_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error("Row task %d raised exception: %s", i, r)
        else:
            actual_results.append(r)

    writer.close()

    # Build enriched all_results copy with eval columns appended
    enriched_paths = build_enriched_output(
        input_path=input_path,
        details_jsonl=writer.details_path,
        output_dir=Path(output_dir),
        run_name=run_name,
    )

    # Print summary
    _print_summary(actual_results, elapsed, writer, enriched_paths, save_local=save_local)

    # Log to MLflow
    _log_to_mlflow(
        results=actual_results,
        elapsed=elapsed,
        enriched_paths=enriched_paths,
        results_csv=writer.results_path,
        details_jsonl=writer.details_path,
        save_local=save_local,
        params={
            "deployment": deployment,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "rubrics_mode": rubrics_mode,
            "route_column": route_column,
            "input_path": input_path,
            "routes": routes,
            "limit": limit,
            "run_name": run_name,
            "data_origin": data_origin,
            "metrics": metrics,
        },
    )


def build_enriched_output(
    input_path: str,
    details_jsonl: Path,
    output_dir: Path,
    run_name: str,
) -> tuple[Path, Path] | None:
    """Copy evaluated rows from input and append eval result columns.

    Writes ``{run_name}_all_results_enriched.csv`` and ``.jsonl`` in output_dir.
    Returns (enriched_csv_path, enriched_jsonl_path), or None on failure.
    """
    # Load eval results indexed by row_id
    eval_by_id: dict[int, dict] = {}
    if details_jsonl.exists():
        with open(details_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        eval_by_id[int(rec["row_id"])] = rec
                    except (json.JSONDecodeError, KeyError):
                        pass

    if not eval_by_id:
        logger.warning("No eval results found in %s — skipping enriched output", details_jsonl)
        return None

    # Load input CSV and keep only the rows that were evaluated
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df[df["row_id"].isin(eval_by_id.keys())].copy()
    df.reset_index(drop=True, inplace=True)

    # Append eval columns
    _scalar_map = {
        "eval_status": "status", "eval_timestamp": "timestamp",
        "eval_output_relevancy_score": "output_relevancy_score",
        "eval_completeness_score": "completeness_score",
        "eval_correctness_score": "correctness_score",
        "eval_overall_notes": "overall_notes",
        "eval_generator_raw": "generator_raw_response",
        "eval_judge_raw": "judge_raw_response",
    }
    _json_map = {
        "eval_rubrics_json": "rubrics",
        "eval_verdicts_json": "verdicts",
    }
    for col in [*_scalar_map, *_json_map]:
        df[col] = None

    def _to_json_str(val: Any) -> str | None:
        if val is None:
            return None
        if isinstance(val, str):
            return val
        return json.dumps(val, ensure_ascii=False)

    for idx, row in df.iterrows():
        ev = eval_by_id[int(row["row_id"])]
        for col, key in _scalar_map.items():
            df.at[idx, col] = ev.get(key)
        for col, key in _json_map.items():
            df.at[idx, col] = _to_json_str(ev.get(key))

    logger.info("Enriched output: %d rows merged", len(df))

    enriched_csv = output_dir / f"{run_name}_all_results_enriched.csv"
    df.to_csv(enriched_csv, index=False, encoding="utf-8")
    logger.info("Enriched CSV written: %s", enriched_csv)

    enriched_jsonl = output_dir / f"{run_name}_all_results_enriched.jsonl"
    with open(enriched_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            # Parse the JSON string columns back into objects for the JSONL
            for col in ("eval_rubrics_json", "eval_verdicts_json"):
                v = rec.get(col)
                if v and isinstance(v, str):
                    try:
                        rec[col] = json.loads(v)
                    except json.JSONDecodeError:
                        pass
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    logger.info("Enriched JSONL written: %s", enriched_jsonl)

    return enriched_csv, enriched_jsonl


def _format_score_line(label: str, s: dict[str, Any], suffix: str = "") -> str:
    """Format a single stats row as: label  OR=x.xxx (±y.yyy)  C=...  [Corr=...]  suffix."""
    rel_avg, rel_std = s["output_relevancy"]
    comp_avg, comp_std = s["completeness"]
    corr_avg, corr_std = s["correctness"]
    corr_text = (
        f"  Correctness={corr_avg:.3f} (±{corr_std:.3f}) [n={s['counts']['correctness_score']}]"
        if s["counts"]["correctness_score"] > 0
        else "  Correctness=N/A"
    )
    return (
        f"  {label:20} Output Relevancy={rel_avg:.3f} (±{rel_std:.3f})  "
        f"Completeness={comp_avg:.3f} (±{comp_std:.3f})"
        f"{corr_text}{suffix}"
    )


def _print_summary(
    results: list[dict[str, Any]],
    elapsed: float,
    writer: OutputWriter,
    enriched_paths: tuple[Path, Path] | None = None,
    save_local: bool = False,
) -> None:
    """Print evaluation summary to console."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    total_rows = len(results)
    excluded_non_response = sum(1 for r in results if r["status"] == NON_RESPONSE_STATUS)
    scored_rows = total_rows - excluded_non_response
    excluded_pct = (excluded_non_response / total_rows * 100.0) if total_rows else 0.0

    print(f"Total rows processed: {total_rows}")
    print(f"Rows included in scoring: {scored_rows}")
    if excluded_non_response:
        print(
            f"No-response rows excluded: {excluded_non_response} "
            f"({excluded_pct:.1f}%)"
        )
    print(f"Elapsed time: {elapsed:.1f}s")
    print()

    # Status breakdown
    status_counts = Counter(r["status"] for r in results)
    print("Status breakdown:")
    for status, count in status_counts.most_common():
        print(f"  {status:30} {count:4}")

    # Score summary by route
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        stats = _compute_score_stats(ok_results)
        print()
        print("Scores by route:")
        for route in sorted(stats["per_route"]):
            s = stats["per_route"][route]
            print(_format_score_line(route, s, f"  n={s['n']}"))
            for mk in METRIC_KEYS:
                sf = f"{mk}_score"
                if s["counts"][sf] > 0:
                    label = SCORE_FIELD_LABELS[sf]
                    print(f"    {label + ' dist:':<32} {_format_distribution(s['distributions'][sf])}")

        print()
        micro, macro = stats["micro"], stats["macro"]
        print(_format_score_line("OVERALL (micro)", micro, f"  n={micro['n']}"))
        for mk in METRIC_KEYS:
            sf = f"{mk}_score"
            if micro["counts"][sf] > 0:
                label = SCORE_FIELD_LABELS[sf]
                print(f"    {label + ' distribution:':<30} {_format_distribution(micro['distributions'][sf])}")
        print(_format_score_line("OVERALL (macro)", macro, f"  n_routes={macro['n_routes']}"))

    print()
    if save_local:
        print("Outputs saved locally:")
        print(f"  Summary  : {writer.results_path}")
        print(f"  Details  : {writer.details_path}")
        if enriched_paths:
            print(f"  Enriched CSV  : {enriched_paths[0]}")
            print(f"  Enriched JSONL: {enriched_paths[1]}")
    else:
        print("Outputs uploaded to MLflow (local files will be removed).")
    print("=" * 80)


def _log_to_mlflow(
    results: list[dict[str, Any]],
    elapsed: float,
    params: dict[str, Any],
    enriched_paths: tuple[Path, Path] | None = None,
    results_csv: Path | None = None,
    details_jsonl: Path | None = None,
    save_local: bool = False,
) -> None:
    """Log evaluation run parameters, metrics, and artifacts to MLflow."""
    try:
        _log_to_mlflow_inner(
            results=results,
            elapsed=elapsed,
            params=params,
            enriched_paths=enriched_paths,
            results_csv=results_csv,
            details_jsonl=details_jsonl,
            save_local=save_local,
        )
    except Exception as exc:
        logger.warning("MLflow logging skipped (server may not be running): %s", exc)


def _log_to_mlflow_inner(
    results: list[dict[str, Any]],
    elapsed: float,
    params: dict[str, Any],
    enriched_paths: tuple[Path, Path] | None = None,
    results_csv: Path | None = None,
    details_jsonl: Path | None = None,
    save_local: bool = False,
) -> None:
    """Inner MLflow logging — raises on connection errors so the caller can handle them."""
    rubrics_mode = params.get("rubrics_mode", "combined")
    route_source = params.get("route_column", "intended")  # "intended" or "orchestrator"
    n_total = len(results)
    ts = datetime.now().strftime("%m%d_%H%M")
    mlflow_run_name = f"{ts}_{route_source}_{n_total}rows"

    experiment_name = f"wc-eval-rubrics-{rubrics_mode}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=mlflow_run_name):
        # ---- Parameters ----
        mlflow.log_params({
            "deployment": params.get("deployment"),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            "concurrency": params.get("concurrency"),
            "rubrics_mode": rubrics_mode,
            "route_source": route_source,
            "data_origin": params.get("data_origin", "all"),
            "input_path": params.get("input_path"),
            "routes_filter": json.dumps(params["routes"]) if params.get("routes") else "all",
            "limit": params.get("limit"),
        })

        # ---- Input dataset ----
        try:
            input_path = params.get("input_path", "")
            if input_path:
                evaluated_ids = {r["row_id"] for r in results}
                input_df = pd.read_csv(input_path, encoding="utf-8-sig")
                input_df_eval = input_df[input_df["row_id"].isin(evaluated_ids)]
                dataset = mlflow.data.from_pandas(
                    input_df_eval,
                    source=input_path,
                    name="eval_input",
                )
                mlflow.log_input(dataset, context="evaluation")
        except Exception as exc:
            logger.warning("Failed to log dataset to MLflow: %s", exc)

        # ---- Run-level counters ----
        status_counts = Counter(r["status"] for r in results)
        mlflow.log_metric("total_rows", len(results))
        mlflow.log_metric("ok_rows", status_counts.get("OK", 0))
        mlflow.log_metric("no_response_rows", status_counts.get(NON_RESPONSE_STATUS, 0))
        if results:
            mlflow.log_metric(
                "no_response_pct",
                status_counts.get(NON_RESPONSE_STATUS, 0) / len(results) * 100.0,
            )
        mlflow.log_metric("elapsed_seconds", round(elapsed, 1))
        for status, count in status_counts.items():
            mlflow.log_metric(f"status_{status.lower()}_count", count)

        # ---- Per-route and aggregate scores ----
        ok_results = [r for r in results if r["status"] == "OK"]
        if ok_results:
            stats = _compute_score_stats(ok_results)
            for route in sorted(stats["per_route"]):
                s = stats["per_route"][route]
                prefix = route.lower()
                mlflow.log_metric(f"{prefix}_n", s["n"])
                for metric in ("output_relevancy", "completeness", "correctness"):
                    score_field = f"{metric}_score"
                    if s["counts"].get(score_field, 0) > 0:
                        avg, std = s[metric]
                        mlflow.log_metric(f"{prefix}_{metric}_avg", avg)
                        mlflow.log_metric(f"{prefix}_{metric}_std", std)

            for level, n_key in (("micro", "n"), ("macro", "n_routes")):
                ls = stats[level]
                if ls[n_key] > 0:
                    for metric in ("output_relevancy", "completeness", "correctness"):
                        score_field = f"{metric}_score"
                        if ls["counts"].get(score_field, 0) == 0:
                            continue
                        avg, std = ls[metric]
                        mlflow.log_metric(f"{level}_{metric}_avg", avg)
                        mlflow.log_metric(f"{level}_{metric}_std", std)
                        if level == "micro":
                            distribution = ls["distributions"][score_field]
                            for score in (0, 1, 2):
                                mlflow.log_metric(
                                    f"{level}_{metric}_score_{score}_pct",
                                    distribution[score]["pct"],
                                )
                                mlflow.log_metric(
                                    f"{level}_{metric}_score_{score}_count",
                                    distribution[score]["count"],
                                )

        # ---- Artifacts: actual result data ----
        artifact_paths = [results_csv, details_jsonl, *(enriched_paths or [])]
        for path in artifact_paths:
            if path and Path(path).exists():
                mlflow.log_artifact(str(path), artifact_path="results")

        # ---- Cleanup local files (only after successful upload) ----
        # Note: keep `details_jsonl` locally to support pipeline resume and debugging.
        cleanup_paths = [results_csv, *(enriched_paths or [])]
        if not save_local:
            for path in cleanup_paths:
                if path and Path(path).exists():
                    try:
                        Path(path).unlink()
                    except Exception as exc:
                        logger.warning("Failed to delete local file %s: %s", path, exc)

    logger.info("MLflow run logged to experiment '%s'", experiment_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Rubrics Evaluation Pipeline for Writing Coach V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.evaluation.eval_pipeline --input final_data/all_results.csv --limit 5\n"
            "  python -m src.evaluation.eval_pipeline --deployment gpt-4o --concurrency 3\n"
            "  python -m src.evaluation.eval_pipeline --routes RESEARCH RESPOND --limit 10\n"
            "  python -m src.evaluation.eval_pipeline --run-name my_experiment\n"
            "\nEnvironment variables (Azure OpenAI):\n"
            "  AZURE_OPENAI_ENDPOINT    — Azure OpenAI endpoint URL\n"
            "  AZURE_OPENAI_API_KEY     — Azure OpenAI API key\n"
            "  AZURE_OPENAI_API_VERSION — API version (default: 2025-01-01-preview)\n"
            "  AZURE_OPENAI_DEPLOYMENT  — Default deployment name (default: gpt-4o)\n"
        ),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to input CSV with system outputs (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        default="data_outputs/eval",
        help="Output directory for evaluation results (default: data_outputs/eval)",
    )
    parser.add_argument(
        "--deployment",
        default=DEFAULT_DEPLOYMENT,
        help=f"Azure OpenAI deployment name (default: {DEFAULT_DEPLOYMENT})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"LLM temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per LLM call (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max parallel LLM calls (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        default=None,
        help="Only evaluate rows with these routes (e.g., RESEARCH RESPOND)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to process (for testing)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume — reprocess all rows even if already in output",
    )
    parser.add_argument(
        "--rubrics-mode",
        choices=["combined", "split"],
        default="combined",
        help=(
            "How to run the rubrics generator. "
            "'combined' (default): single LLM call with both metrics in one prompt (rubrics_prompt.txt). "
            "'split': two parallel LLM calls, one per metric, using the metric-specific prompt files."
        ),
    )
    parser.add_argument(
        "--route-column",
        choices=["intended", "orchestrator"],
        default="intended",
        help=(
            "Which route column to use from the input CSV. "
            "'intended' (default): use route_intended (the ground truth route). "
            "'orchestrator': use route_orch (the orchestrator's predicted route). "
            "Falls back to the other column if the preferred one is missing."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for output files (default: eval_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        default=False,
        help=(
            "Keep output files on disk after the run (default: False). "
            "By default, files are uploaded to MLflow and then deleted locally."
        ),
    )
    parser.add_argument(
        "--data-origin",
        choices=["all", "natural", "synthetic"],
        default="all",
        help=(
            "Filter rows by data origin (default: all). "
            "'synthetic': rows where dataset_source starts with 'Synthetic' or equals 'extra_respond_alex'. "
            "'natural': all other rows."
        ),
    )
    parser.add_argument(
        "--metrics",
        "--metric",
        nargs="+",
        default=None,
        choices=SCORING_METRIC_NAMES,
        metavar="METRIC",
        help=(
            f"Subset of metrics to evaluate (default: all). "
            f"Choices: {', '.join(SCORING_METRIC_NAMES)}. "
            "Example: --metrics completeness output_relevancy"
        ),
    )

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            deployment=args.deployment,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            routes=args.routes,
            limit=args.limit,
            resume=not args.no_resume,
            run_name=args.run_name,
            rubrics_mode=args.rubrics_mode,
            route_column=args.route_column,
            save_local=args.save_local,
            data_origin=args.data_origin,
            metrics=args.metrics,
        )
    )


if __name__ == "__main__":
    main()
