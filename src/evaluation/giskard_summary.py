"""Build a combined summary of Giskard scan results (CSV + printable table).

For each scanner's ``scan_summary.json`` this computes:

* A ``total`` row per scanner: failure rate = ``exp_sum / (n_adversarial_samples
  * n_requirements)`` where ``exp_sum`` is the summed length of every issue's
  ``examples`` list.
* One ``issue`` row per issue: failure rate = ``len(examples_i) /
  n_adversarial_samples``.

Both scopes also report ``n_adversarial_samples`` and the example count used in
the numerator (``n_examples``).
"""

import csv
import json
from pathlib import Path


GISKARD_SUMMARY_FIELDNAMES = [
    "scanner",
    "scope",
    "issue_index",
    "detector",
    "requirement",
    "n_adversarial_samples",
    "n_requirements",
    "n_examples",
    "failure_pct",
]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Return numerator/denominator, or None when the denominator is 0/missing."""
    if not denominator:
        return None
    return numerator / denominator


def _as_pct(ratio: float | None) -> float | None:
    """Convert a 0-1 ratio into a percentage rounded to 2 decimals."""
    if ratio is None:
        return None
    return round(ratio * 100, 2)


def _issue_requirement(issue: dict) -> str:
    """Best-effort human-readable requirement label for an issue."""
    meta = issue.get("meta") or {}
    return meta.get("requirement") or issue.get("description", "") or ""


def build_giskard_summary_rows(base_output_dir: Path, scan_metrics: list[str]) -> list[dict]:
    """Read each scanner's scan_summary.json and build combined summary rows.

    Returns a flat list of dicts (one ``total`` row plus one ``issue`` row per
    issue, per scanner). Scanners whose ``scan_summary.json`` is missing are
    skipped silently.
    """
    base_output_dir = Path(base_output_dir)
    rows: list[dict] = []

    for metric in scan_metrics:
        summary_path = base_output_dir / metric / "scan_summary.json"
        if not summary_path.exists():
            continue

        with summary_path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        issues = data.get("issues", []) or []
        n_adv = data.get("n_adversarial_samples")
        n_req = data.get("n_requirements")
        total_issues = data.get("total_issues", len(issues))
        example_counts = [len(issue.get("examples", []) or []) for issue in issues]
        exp_sum = sum(example_counts)

        denom = (n_adv or 0) * (n_req or 0)
        rows.append(
            {
                "scanner": metric,
                "scope": "total",
                "issue_index": "",
                "detector": "",
                "requirement": f"{total_issues} issue(s)",
                "n_adversarial_samples": n_adv,
                "n_requirements": n_req,
                "n_examples": exp_sum,
                "failure_pct": _as_pct(_safe_ratio(exp_sum, denom)),
            }
        )

        for idx, issue in enumerate(issues):
            n_ex = example_counts[idx]
            rows.append(
                {
                    "scanner": metric,
                    "scope": "issue",
                    "issue_index": idx,
                    "detector": issue.get("detector", ""),
                    "requirement": _issue_requirement(issue),
                    "n_adversarial_samples": n_adv,
                    "n_requirements": "",
                    "n_examples": n_ex,
                    "failure_pct": _as_pct(_safe_ratio(n_ex, n_adv)),
                }
            )

    return rows


def write_giskard_summary_csv(output_path: Path, rows: list[dict]) -> None:
    """Write the combined Giskard summary rows to a CSV file."""
    with Path(output_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GISKARD_SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def format_giskard_summary_table(rows: list[dict], max_requirement: int = 55) -> str:
    """Render the summary rows as an aligned, fixed-width text table."""
    if not rows:
        return "No Giskard scan summaries found."

    columns = [
        ("scanner", "scanner"),
        ("scope", "scope"),
        ("issue_index", "issue"),
        ("detector", "detector"),
        ("n_adversarial_samples", "n_adv"),
        ("n_requirements", "n_req"),
        ("n_examples", "n_ex"),
        ("failure_pct", "failure_%"),
        ("requirement", "requirement"),
    ]

    def cell(row: dict, key: str) -> str:
        value = row.get(key)
        if value is None or value == "":
            return "-"
        if key == "failure_pct":
            return f"{value}%"
        if key == "requirement":
            text = str(value)
            return text if len(text) <= max_requirement else text[: max_requirement - 1] + "…"
        return str(value)

    widths = {
        key: max(len(header), *(len(cell(row, key)) for row in rows))
        for key, header in columns
    }

    def render(values: dict[str, str]) -> str:
        return "  ".join(values[key].ljust(widths[key]) for key, _ in columns)

    header = render({key: header for key, header in columns})
    separator = "  ".join("-" * widths[key] for key, _ in columns)
    body = "\n".join(render({key: cell(row, key) for key, _ in columns}) for row in rows)
    return "\n".join([header, separator, body])
