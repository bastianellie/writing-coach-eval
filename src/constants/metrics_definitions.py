METRICS_DEFINITION = {
    "Output Relevancy": "Measures whether the edited output text w.r.t. the input text or the provided edit suggestions directly address the user command and demonstrate an understanding of the users’ intent(s).",
    "Completeness": "Assesses the extent to which the system’s output (edited text or edit suggestions) addresses all required aspects of the user’s request without omitting critical elements.",
}

CORRECTNESS_METRIC_NAME = "Correctness"
CORRECTNESS_METRIC_DESCRIPTION = (
    "Assesses whether revision suggestions are structurally sound, usable, and appropriate as writing-coach edits. "
    "This metric applies only to suggestion outputs in REVISE_SIMPLE and REVISE_RESEARCH."
)


# --- Metric registry --------------------------------------------------------
# Canonical metric names accepted by the --metrics CLI argument. This is the
# single source of truth; do not redefine these lists elsewhere.

# Row-level metrics scored by the LLM rubrics pipeline.
SCORING_METRIC_NAMES: list[str] = ["completeness", "output_relevancy", "correctness"]

# Dataset-level metrics run as Giskard scans.
SCAN_METRIC_NAMES: list[str] = ["potential_harm", "toxicity", "resilience"]

# Groundedness — hallucination detection over RESEARCH / REVISE_RESEARCH outputs.
GROUNDEDNESS_METRIC_NAMES: list[str] = ["groundedness"]

# Union of everything selectable via --metrics.
ALL_METRIC_NAMES: list[str] = [
    *SCORING_METRIC_NAMES,
    *SCAN_METRIC_NAMES,
    *GROUNDEDNESS_METRIC_NAMES,
]


def split_selected_metrics(
    metrics: list[str] | None,
) -> tuple[list[str] | None, list[str], list[str]]:
    """Split selected metrics into scoring, Giskard scan, and groundedness metrics.

    Args:
        metrics: List of metric names, or None to select all row-level metrics.

    Returns:
        Tuple of (scoring_metrics, scan_metrics, groundedness_metrics) where
        scoring_metrics is None if all row-level metrics are selected, or a list
        of the selected ones. scan_metrics and groundedness_metrics are always
        lists (possibly empty).
    """
    if metrics is None:
        return None, [], []

    normalized: list[str] = []
    for metric in metrics:
        value = metric.lower().replace(" ", "_")
        if value not in normalized:
            normalized.append(value)

    scoring = [metric for metric in normalized if metric in SCORING_METRIC_NAMES]
    scans = [metric for metric in normalized if metric in SCAN_METRIC_NAMES]
    groundedness = [metric for metric in normalized if metric in GROUNDEDNESS_METRIC_NAMES]
    return scoring, scans, groundedness