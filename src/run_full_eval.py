"""Run the full evaluation pipeline end-to-end.

Orchestrates these scripts in order:
  Optional: Giskard scans      — dataset-level safety / robustness scans
  Optional: groundedness       — hallucination detection over RESEARCH / REVISE_RESEARCH
  1. eval_pipeline.py          — LLM rubrics evaluation
  2. split_natural_synthetic.py — split enriched results by data origin
  3. heuristic_scoring.py       — heuristic scoring for all / natural / synthetic
  4. extract_reasoning_by_score.py — extract per-metric/score reasoning
  5. summarize_reasoning.py     — LLM summarisation of reasoning patterns

Usage:
  python -m src.run_full_eval
  python -m src.run_full_eval --route-column orchestrator
  python -m src.run_full_eval --routes RESEARCH RESPOND --run-name my_experiment
  python -m src.run_full_eval --metrics completeness output_relevancy
  python -m src.run_full_eval --metrics completeness potential_harm resilience
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `src.*` imports work whether the script
# is invoked as `python src/run_full_eval.py` or `python -m src.run_full_eval`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load environment (AZURE_OPENAI_*, WC_APP_SRC, ...) from the repo-root .env up
# front, so it is available regardless of which metric step runs. Some step
# modules also call load_dotenv() on import, but those imports are lazy — e.g. a
# groundedness-only run never imports them, so the orchestrator must load it.
from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

from src.constants.metrics_definitions import ALL_METRIC_NAMES, split_selected_metrics
from src.evaluation.giskard_summary import (
    build_giskard_summary_rows,
    format_giskard_summary_table,
    write_giskard_summary_csv,
)
from src.evaluation.hallucination import DEFAULT_HD_METHOD, available_detectors
from src.evaluation.summary_scores import SUMMARY_CSV_FIELDNAMES, build_summary_csv_rows, write_summary_csv
from src.orchestration.eval_steps import (
    step_eval_pipeline,
    step_extract_reasoning,
    step_giskard_scans,
    step_groundedness,
    step_heuristic_scoring,
    step_split_natural_synthetic,
    step_summarize_reasoning,
)
from src.orchestration.utils import find_file_ending_with, timestamp



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Writing-Coach evaluation pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="final_data/all_results.csv",
        help="Path to input CSV with system outputs (default: final_data/all_results.csv)",
    )
    parser.add_argument(
        "--route-column",
        choices=["intended", "orchestrator"],
        default="intended",
        help=(
            "Which route column to use. "
            "'intended' (default): uses route_intended and stores results under route_intended/. "
            "'orchestrator': uses route_orch and stores results under route_orch/."
        ),
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        default=None,
        help="Only evaluate rows with these routes (e.g. RESEARCH RESPOND)",
    )
    parser.add_argument(
        "--rubrics-mode",
        choices=["combined", "split"],
        default="combined",
        help=(
            "How to run the rubrics generator. "
            "'combined' (default): single LLM call with all metrics in one prompt "
            "(output folder uses 'one_prompt'). "
            "'split': one LLM call per metric (output folder uses 'multi_prompt')."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for output files (default: eval_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--metrics",
        "--metric",
        nargs="+",
        default=list(ALL_METRIC_NAMES),
        choices=ALL_METRIC_NAMES,
        metavar="METRIC",
        help=(
            "Subset of metrics to evaluate (default: all metrics, including "
            "Giskard scans). "
            "Row-level metrics: completeness, output_relevancy, correctness. "
            "Giskard scan metrics: potential_harm, toxicity, resilience. "
            "Example: --metrics completeness output_relevancy potential_harm"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of rows to process (for testing)",
    )
    parser.add_argument(
        "--output-root",
        default="eval_data",
        help=(
            "Base directory under which the run's root folder is created "
            "(default: eval_data). The full path is "
            "<output-root>/wc<version>_<one|multi>_prompt/<route_folder>/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of input rows to sample as seed data for Giskard scans (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Giskard scan sampling and adversarial generation (default: 42)",
    )
    parser.add_argument(
        "--n-adversarial-samples",
        type=int,
        default=5,
        help="Number of adversarial samples per Giskard detector (default: 5)",
    )
    parser.add_argument(
        "--n-requirements",
        type=int,
        default=4,
        help="Number of generated requirements per Giskard detector (default: 4)",
    )
    parser.add_argument(
        "--wc-version",
        choices=["v2", "v3"],
        default="v3",
        help="Writing Coach interface version used by Giskard scans (default: v3)",
    )
    parser.add_argument(
        "--wc-app-src",
        default=(os.getenv("WC_APP_SRC") or "").strip('"').strip("'") or None,
        help=(
            "Path to Writing Coach app root or src directory for Giskard scans "
            "(optional if WC_APP_SRC is set)"
        ),
    )
    parser.add_argument(
        "--hd-method",
        default=DEFAULT_HD_METHOD,
        choices=available_detectors(),
        help=(
            "Hallucination-detection method for the 'groundedness' metric "
            f"(default: {DEFAULT_HD_METHOD})"
        ),
    )
    parser.add_argument(
        "--hd-model",
        default="gpt-5-mini",
        help="Azure deployment used by the HD detector (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--hd-prompt-version",
        default="V5_anchor_V7",
        help="HD detection prompt version (default: V5_anchor_V7)",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    scoring_metrics, scan_metrics, groundedness_metrics = split_selected_metrics(args.metrics)
    should_run_scoring_pipeline = args.metrics is None or bool(scoring_metrics)
    run_groundedness = bool(groundedness_metrics)
    total_steps = (
        (1 if scan_metrics else 0)
        + (1 if run_groundedness else 0)
        + (5 if should_run_scoring_pipeline else 0)
    )

    route_folder = "route_orch" if args.route_column == "orchestrator" else "route_intended"
    prompt_segment = "one_prompt" if args.rubrics_mode == "combined" else "multi_prompt"
    ts = timestamp()
    base_output_dir = Path(args.output_root) / f"wc{args.wc_version}_{prompt_segment}" / route_folder / ts
    base_output_dir.mkdir(parents=True, exist_ok=True)
    step_index = 0

    print(f"Selected row-level metrics: {scoring_metrics if scoring_metrics is not None else 'all'}")
    print(f"Selected Giskard scan metrics: {scan_metrics if scan_metrics else 'none'}")
    print(f"Groundedness (HD): {'yes' if run_groundedness else 'no'}")

    if scan_metrics:
        step_index += 1
        print("\n" + "=" * 80)
        print(f"STEP {step_index}/{total_steps}  giskard_scans  ->  {base_output_dir}")
        print("=" * 80)
        step_giskard_scans(
            input_path=args.input,
            output_dir=base_output_dir,
            scan_metrics=scan_metrics,
            n_samples=args.n_samples,
            seed=args.seed,
            n_adversarial_samples=args.n_adversarial_samples,
            n_requirements=args.n_requirements,
            wc_version=args.wc_version,
            wc_app_src=args.wc_app_src,
        )

        # ---- Giskard scan summary (CSV + on-screen table) ----
        giskard_rows = build_giskard_summary_rows(base_output_dir, scan_metrics)
        giskard_summary_csv = base_output_dir / "giskard_scan_summary.csv"
        write_giskard_summary_csv(giskard_summary_csv, giskard_rows)
        print("\nGiskard scan summary:")
        print(format_giskard_summary_table(giskard_rows))
        print(f"\nSaved Giskard summary -> {giskard_summary_csv}")

    groundedness_summary: dict | None = None
    if run_groundedness:
        step_index += 1
        print("\n" + "=" * 80)
        print(f"STEP {step_index}/{total_steps}  groundedness  ->  {base_output_dir}")
        print("=" * 80)
        groundedness_summary, gnd_table = step_groundedness(
            input_path=args.input,
            output_dir=base_output_dir,
            hd_method=args.hd_method,
            model=args.hd_model,
            prompt_version=args.hd_prompt_version,
            limit=args.limit,
        )
        print("\nGroundedness summary:")
        print(gnd_table)
        print(f"\nSaved groundedness -> {base_output_dir / 'groundedness_summary.csv'}")

    if not should_run_scoring_pipeline:
        # No rubrics scoring — still emit a consolidated summary CSV if groundedness ran.
        if groundedness_summary is not None:
            summary_csv_path = base_output_dir / "summary_scores.csv"
            write_summary_csv(
                summary_csv_path,
                build_summary_csv_rows(groundedness_summary=groundedness_summary),
            )
            print(f"Saved summary CSV -> {summary_csv_path}")
        print("\n" + "=" * 80)
        print("FULL PIPELINE COMPLETE")
        print("Completed scan / groundedness metrics only (no row-level scoring metrics selected).")
        print(f"All outputs in: {base_output_dir}")
        print("=" * 80)
        return

    # ---- Step 1: eval pipeline ----
    step_index += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_index}/{total_steps}  eval_pipeline  ->  {base_output_dir}")
    print("=" * 80)
    step_eval_pipeline(
        input_path=args.input,
        output_dir=str(base_output_dir),
        route_column=args.route_column,
        routes=args.routes,
        run_name=args.run_name,
        metrics=scoring_metrics,
        limit=args.limit,
        rubrics_mode=args.rubrics_mode,
    )

    # Discover enriched CSV produced by step 1
    enriched_csv = find_file_ending_with(base_output_dir, "_enriched.csv")
    results_csv = find_file_ending_with(base_output_dir, "_results.csv")
    print(f"\nEnriched CSV: {enriched_csv}")

    # ---- Step 2: split natural / synthetic ----
    step_index += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_index}/{total_steps}  split_natural_synthetic  ->  {base_output_dir}")
    print("=" * 80)
    split_output_dir = base_output_dir / "natural_synthetic_split"
    split_text, summaries = step_split_natural_synthetic(
        enriched_csv,
        split_output_dir,
        metrics=scoring_metrics,
    )
    print(split_text)

    scores_txt = base_output_dir / "scores_all_natural_synthetic.txt"
    scores_txt.write_text(split_text, encoding="utf-8")
    print(f"Saved scores -> {scores_txt}")

    # ---- Step 3: heuristic scoring (all, natural, synthetic) ----
    step_index += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_index}/{total_steps}  heuristic_scoring  (all / natural / synthetic)")
    print("=" * 80)
    heuristic_parts: list[str] = []
    heuristic_summaries: dict[str, dict] = {}

    # 3a — all data (uses *_results.csv from the base output dir, enriched also works)
    print("\n--- all data ---")
    part, h_summary = step_heuristic_scoring(results_csv, metrics=scoring_metrics)
    print(part)
    heuristic_parts.append(f"=== ALL DATA ===\n{part}")
    heuristic_summaries["all"] = h_summary

    # 3b — natural
    natural_csv = find_file_ending_with(split_output_dir, "_natural.csv")
    print("--- natural data ---")
    part, h_summary = step_heuristic_scoring(natural_csv, metrics=scoring_metrics)
    print(part)
    heuristic_parts.append(f"=== NATURAL DATA ===\n{part}")
    heuristic_summaries["natural"] = h_summary

    # 3c — synthetic
    synthetic_csv = find_file_ending_with(split_output_dir, "_synthetic.csv")
    print("--- synthetic data ---")
    part, h_summary = step_heuristic_scoring(synthetic_csv, metrics=scoring_metrics)
    print(part)
    heuristic_parts.append(f"=== SYNTHETIC DATA ===\n{part}")
    heuristic_summaries["synthetic"] = h_summary

    heuristic_txt = base_output_dir / "heuristic_scoring_all_natural_synthetic.txt"
    heuristic_txt.write_text("\n\n".join(heuristic_parts), encoding="utf-8")
    print(f"Saved heuristic scores -> {heuristic_txt}")

    # ---- Build consolidated summary CSV ----
    summary_csv_path = base_output_dir / "summary_scores.csv"
    csv_rows = build_summary_csv_rows(summaries, heuristic_summaries, groundedness_summary)
    write_summary_csv(summary_csv_path, csv_rows)
    print(f"Saved summary CSV -> {summary_csv_path}")

    # ---- Step 4: extract reasoning by score ----
    step_index += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_index}/{total_steps}  extract_reasoning_by_score")
    print("=" * 80)
    reasoning_dir = base_output_dir / "metrics_score_combinations"
    step_extract_reasoning(enriched_csv, reasoning_dir, metrics=scoring_metrics)

    # ---- Step 5: summarize reasoning ----
    step_index += 1
    print("\n" + "=" * 80)
    print(f"STEP {step_index}/{total_steps}  summarize_reasoning")
    print("=" * 80)
    item_level_dir = reasoning_dir / "item_level"
    step_summarize_reasoning(item_level_dir, metrics=scoring_metrics)

    print("\n" + "=" * 80)
    print("FULL PIPELINE COMPLETE")
    print(f"All outputs in: {base_output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
