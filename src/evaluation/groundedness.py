"""Groundedness metric — run hallucination detection over WC outputs.

For every input row whose orchestrator route (``route_orch``) is ``RESEARCH`` or
``REVISE_RESEARCH``, this decomposes the row into one or more "summaries"
(the ``response``, or each ``suggestion``'s ``transformed_text``) paired with the
structured references cited in it, scores each summary with a
:class:`~src.evaluation.hallucination.base.HallucinationDetector`, and aggregates
the per-summary label distributions into a grounded ratio exactly as
``grounded_ratio.py`` does.

Outputs (written into the run folder):
  * ``groundedness.json``        — full per-summary results + aggregate
  * ``groundedness_summary.csv`` — one row per summary plus an OVERALL row
and a printable table is returned for the terminal / a ``.txt`` file.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonlines
import pandas as pd
from tqdm import tqdm

from src.metrics.accuracy import _BRACKET_GROUP, _ids_in_bracket
from src.evaluation.hallucination.base import HallucinationDetector, SummaryScore

# Thresholds copied from grounded_ratio.py.
GROUNDED_THRESHOLD = 85.0
UNVERIFIABLE_WEIGHT = 0.87  # weight of the error contributed by unverifiable claims

GROUNDEDNESS_ROUTES = {"RESEARCH", "REVISE_RESEARCH"}


def is_grounded(distribution: dict[str, float]) -> bool:
    """Apply the grounded_ratio.py rule to a per-summary label distribution."""
    entailment = distribution.get("entailment(%)", 0.0)
    unverifiable = distribution.get("unverifiable(%)", 0.0)
    return (entailment + unverifiable * UNVERIFIABLE_WEIGHT) >= GROUNDED_THRESHOLD


def decompose_row(row: dict) -> list[tuple[str, list[dict]]]:
    """Split a row into ``(text, references)`` summary units.

    * ``RESEARCH``        -> a single ``(response, references)`` unit.
    * ``REVISE_RESEARCH`` -> one unit per suggestion, pairing the suggestion's
      ``transformed_text`` with only the references it cites.

    Units without any references are skipped. Reproduces the routing/pairing
    done by ``src.metrics.accuracy.load_and_preprocess_revise_rows`` (without the
    integer-remapping step, which each detector performs internally).
    """
    route = str(row.get("route_orch", ""))
    if route not in GROUNDEDNESS_ROUTES:
        return []

    output = row.get("output")
    if isinstance(output, str):
        output = json.loads(output)
    if not isinstance(output, dict):
        return []

    all_references: list[dict] = output.get("references") or []

    if route == "RESEARCH":
        response = output.get("response") or ""
        if not all_references or not response:
            return []
        return [(response, all_references)]

    # REVISE_RESEARCH
    ref_lookup = {r["referenceId"]: r for r in all_references if r.get("referenceId")}
    units: list[tuple[str, list[dict]]] = []
    for suggestion in output.get("suggestions") or []:
        text = suggestion.get("transformed_text", "") or ""
        if not text:
            continue
        cited_ids: list[str] = []
        seen: set[str] = set()
        for match in _BRACKET_GROUP.finditer(text):
            for raw_id in _ids_in_bracket(match):
                if raw_id not in seen:
                    cited_ids.append(raw_id)
                    seen.add(raw_id)
        suggestion_refs = [ref_lookup[rid] for rid in cited_ids if rid in ref_lookup]
        if not suggestion_refs:
            continue
        units.append((text, suggestion_refs))
    return units


def _iter_rows(input_path: Path):
    """Yield ``(input_line, row_dict)`` from a .jsonl or .csv file."""
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        with jsonlines.open(input_path) as reader:
            for line_no, row in enumerate(reader, start=1):
                yield line_no, row
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
        for df_idx, row in df.iterrows():
            yield df_idx + 2, row.to_dict()  # +2: 0-based index + header row
    else:
        raise ValueError(f"Unsupported input format '{suffix}'. Expected .jsonl or .csv.")


def evaluate_groundedness(
    input_path: str | Path,
    detector: HallucinationDetector,
    limit: int | None = None,
    workers: int = 1,
) -> list[dict]:
    """Score every qualifying summary. Returns one result dict per summary.

    Summary units are scored concurrently with a pool of *workers* threads
    (``score_summary`` is a blocking, network-bound call). Results preserve
    input order regardless of completion order.
    """
    input_path = Path(input_path)

    # Flatten rows into ordered candidate summary units (one entry per
    # (text, references) pair) before scoring, so they can be dispatched to
    # the thread pool.
    candidates: list[dict] = []
    for input_line, row in _iter_rows(input_path):
        route = str(row.get("route_orch", ""))
        for text, references in decompose_row(row):
            candidates.append(
                {
                    "input_line": input_line,
                    "route": route,
                    "query": row.get("query", ""),
                    "text": text,
                    "references": references,
                }
            )

    # `limit` is a testing aid: cap the number of summary units scored. Each
    # candidate normally yields one result (decompose_row already drops units
    # without references), so this closely matches the previous "first N
    # summaries" behaviour while avoiding scoring the entire dataset.
    if limit is not None:
        candidates = candidates[:limit]

    def _score(candidate: dict) -> SummaryScore:
        return detector.score_summary(candidate["text"], candidate["references"])

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        # executor.map preserves submission order, so results stay deterministic.
        scores = executor.map(_score, candidates)
        for candidate, score in tqdm(
            zip(candidates, scores),
            total=len(candidates),
            desc="Groundedness",
            unit="summary",
        ):
            if score.n_cited_claims == 0:
                continue  # no cited claims -> nothing to ground
            distribution = score.claim_distribution_per_summary
            results.append(
                {
                    "input_line": candidate["input_line"],
                    "route": candidate["route"],
                    "query": candidate["query"],
                    "summary": candidate["text"],
                    "# cited claims": score.n_cited_claims,
                    "claim_distribution_per_summary": distribution,
                    "grounded": is_grounded(distribution),
                    "claims": score.claims,
                }
            )

    return results


def _aggregate_subset(results: list[dict]) -> dict:
    """Aggregate a list of per-summary results into grounded counts + label totals."""
    grounded = sum(1 for r in results if r["grounded"])
    total = len(results)
    label_counts = {"Entailment": 0, "Unverifiable": 0, "Contradiction": 0}
    for r in results:
        for claim in r["claims"]:
            label = claim.get("claim_decision")
            if label in label_counts:
                label_counts[label] += 1
    return {
        "total_summaries": total,
        "grounded": grounded,
        "non_grounded": total - grounded,
        "grounded_ratio(%)": round(grounded / total * 100, 1) if total else 0.0,
        "total_cited_claims": sum(r["# cited claims"] for r in results),
        "label_counts": label_counts,
    }


def aggregate(results: list[dict]) -> dict:
    """Aggregate per-summary results into the grounded ratio (à la grounded_ratio.py).

    Includes an overall aggregate plus a per-route breakdown under ``by_route``.
    """
    summary = {
        "grounded_threshold": GROUNDED_THRESHOLD,
        "unverifiable_weight": UNVERIFIABLE_WEIGHT,
        **_aggregate_subset(results),
    }
    by_route: dict[str, dict] = {}
    for route in sorted(GROUNDEDNESS_ROUTES):
        route_results = [r for r in results if r["route"] == route]
        if route_results:
            by_route[route] = _aggregate_subset(route_results)
    summary["by_route"] = by_route
    return summary


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

GROUNDEDNESS_CSV_FIELDNAMES = [
    "scope",
    "input_line",
    "route",
    "n_cited_claims",
    "entailment(%)",
    "unverifiable(%)",
    "contradiction(%)",
    "grounded",
]


def build_groundedness_csv_rows(results: list[dict], summary: dict) -> list[dict]:
    rows: list[dict] = []
    for r in results:
        d = r["claim_distribution_per_summary"]
        rows.append(
            {
                "scope": "summary",
                "input_line": r["input_line"],
                "route": r["route"],
                "n_cited_claims": r["# cited claims"],
                "entailment(%)": d.get("entailment(%)"),
                "unverifiable(%)": d.get("unverifiable(%)"),
                "contradiction(%)": d.get("contradiction(%)"),
                "grounded": r["grounded"],
            }
        )
    rows.append(
        {
            "scope": "OVERALL",
            "input_line": "",
            "route": "",
            "n_cited_claims": summary["total_cited_claims"],
            "entailment(%)": "",
            "unverifiable(%)": "",
            "contradiction(%)": "",
            "grounded": f"{summary['grounded']}/{summary['total_summaries']} "
            f"({summary['grounded_ratio(%)']}%)",
        }
    )
    return rows


def write_groundedness_csv(output_path: Path, rows: list[dict]) -> None:
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GROUNDEDNESS_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def format_groundedness_table(summary: dict) -> str:
    lc = summary["label_counts"]
    lines = [
        f"Grounded threshold : {summary['grounded_threshold']}% "
        f"(entailment + {summary['unverifiable_weight']}*unverifiable)",
        f"Total summaries    : {summary['total_summaries']}",
        f"Grounded           : {summary['grounded']}",
        f"Non-grounded       : {summary['non_grounded']}",
        f"Grounded ratio     : {summary['grounded_ratio(%)']}%",
        f"Total cited claims : {summary['total_cited_claims']}",
        f"  Entailment       : {lc['Entailment']}",
        f"  Unverifiable     : {lc['Unverifiable']}",
        f"  Contradiction    : {lc['Contradiction']}",
    ]
    return "\n".join(lines)


def run_groundedness(
    input_path: str | Path,
    detector: HallucinationDetector,
    output_dir: str | Path,
    limit: int | None = None,
    workers: int = 1,
) -> tuple[dict, str]:
    """Full groundedness step: evaluate, aggregate, write JSON + CSV, return (summary, table)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate_groundedness(input_path, detector, limit=limit, workers=workers)
    summary = aggregate(results)

    # Full results JSON
    json_path = output_dir / "groundedness.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": detector.name,
                "model": getattr(detector, "model", None),
                "prompt_version": getattr(detector, "prompt_version", None),
                "summary": summary,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Per-summary CSV
    csv_path = output_dir / "groundedness_summary.csv"
    write_groundedness_csv(csv_path, build_groundedness_csv_rows(results, summary))

    return summary, format_groundedness_table(summary)
