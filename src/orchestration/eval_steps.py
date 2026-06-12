"""Step runner functions for the evaluation pipeline."""

from pathlib import Path


def step_eval_pipeline(
    input_path: str,
    output_dir: str,
    route_column: str,
    routes: list[str] | None,
    run_name: str | None,
    metrics: list[str] | None = None,
    limit: int | None = None,
    rubrics_mode: str = "combined",
) -> None:
    """Run the LLM rubrics evaluation pipeline."""
    import asyncio
    from src.evaluation.eval_pipeline import run_pipeline

    asyncio.run(
        run_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            route_column=route_column,
            routes=routes,
            run_name=run_name,
            save_local=True,
            metrics=metrics,
            limit=limit,
            rubrics_mode=rubrics_mode,
        )
    )


def step_groundedness(
    input_path: str,
    output_dir: Path,
    hd_method: str,
    model: str,
    prompt_version: str,
    limit: int | None = None,
) -> tuple[dict, str]:
    """Run hallucination detection (groundedness) over RESEARCH / REVISE_RESEARCH rows.

    Returns (summary_dict, printable_table).
    """
    from src.evaluation.hallucination import get_detector
    from src.evaluation.groundedness import run_groundedness

    detector = get_detector(hd_method, model=model, prompt_version=prompt_version)
    return run_groundedness(
        input_path=input_path,
        detector=detector,
        output_dir=output_dir,
        limit=limit,
    )


def step_giskard_scans(
    input_path: str,
    output_dir: Path,
    scan_metrics: list[str],
    n_samples: int = 20,
    seed: int = 42,
    n_adversarial_samples: int = 5,
    n_requirements: int = 4,
    wc_version: str = "v3",
    wc_app_src: str | None = None,
) -> None:
    """Run the selected dataset-level Giskard scans."""
    from src.evaluation.giskard_orchestrator import SCAN_METRIC_RUNNERS

    for metric_name in scan_metrics:
        runner = SCAN_METRIC_RUNNERS[metric_name]
        metric_output_dir = output_dir / metric_name
        print(f"Running Giskard scan '{metric_name}' -> {metric_output_dir}")
        runner(
            dataset_csv=input_path,
            n_samples=n_samples,
            seed=seed,
            n_adversarial_samples=n_adversarial_samples,
            n_requirements=n_requirements,
            wc_version=wc_version,
            wc_app_src=wc_app_src,
            persist_output=True,
            output_dir=str(metric_output_dir),
        )
        print(f"Completed Giskard scan '{metric_name}'")


def step_split_natural_synthetic(
    enriched_csv: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
) -> tuple[str, dict]:
    """Split enriched results into natural / synthetic and print scores.

    Returns (captured_stdout_text, summaries_dict).
    """
    import io
    import sys
    from src.scripts.split_natural_synthetic import (
        build_synthetic_mask,
        load_enriched_results,
        maybe_write_outputs,
        normalize_eval_frame,
        print_subset_summary,
        resolve_route_column,
        summarize_subset,
    )

    df = load_enriched_results(enriched_csv)
    route_column = resolve_route_column(df, "auto", enriched_csv)
    df = normalize_eval_frame(df, route_column)

    synthetic_mask = build_synthetic_mask(df)
    subsets = {
        "all": df.copy(),
        "natural": df.loc[~synthetic_mask].copy(),
        "synthetic": df.loc[synthetic_mask].copy(),
    }
    summaries = {
        name: summarize_subset(name, subset_df, route_column, metrics=metrics)
        for name, subset_df in subsets.items()
    }

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print(f"Input file: {enriched_csv}")
        print(f"Route column: {route_column}")
        print_subset_summary(summaries["all"])
        print_subset_summary(summaries["natural"])
        print_subset_summary(summaries["synthetic"])

        maybe_write_outputs(
            output_dir=output_dir,
            input_path=enriched_csv,
            subsets=subsets,
            summaries=summaries,
            write_splits=True,
            save_summary_json=False,
        )
    finally:
        sys.stdout = old_stdout

    return buf.getvalue(), summaries


def step_heuristic_scoring(
    input_path: Path,
    metrics: list[str] | None = None,
) -> tuple[str, dict]:
    """Run heuristic_scoring on one file and return (captured_stdout, summary_dict)."""
    import io
    import sys
    from src.scripts.heuristic_scoring import (
        analyze_rows,
        build_augmented_output_path,
        load_rows,
        print_summary,
        write_rows,
    )

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        rows = load_rows(input_path)
        augmented_rows, summary = analyze_rows(rows, metrics=metrics)
        print_summary(summary, input_path)
        output_path = build_augmented_output_path(input_path)
        write_rows(output_path, augmented_rows)
        print()
        print(f"Augmented output written to: {output_path}")
    finally:
        sys.stdout = old_stdout

    return buf.getvalue(), summary


def step_extract_reasoning(
    enriched_csv: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    """Extract per-metric/score reasoning."""
    from src.evaluation.extract_reasoning_by_score import main as extract_main

    extract_main(input_csv=enriched_csv, output_dir=output_dir, metrics=metrics)


def step_summarize_reasoning(
    data_dir: Path,
    metrics: list[str] | None = None,
) -> None:
    """LLM-summarise reasoning patterns."""
    from src.evaluation.summarize_reasoning import main as summarize_main

    summarize_main(data_dir=data_dir, metrics=metrics)

