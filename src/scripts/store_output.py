"""Universal batch runner for Writing Coach outputs across multiple versions."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


def _raise_csv_field_limit() -> None:
    """Raise the CSV per-field size limit so large `input` documents can be read.

    Input rows can carry document text well beyond the default 128 KB field
    limit. Use the largest value the platform's C long accepts, backing off on
    OverflowError (sys.maxsize overflows the C long on some platforms).
    """
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int = int(max_int // 10)


_raise_csv_field_limit()

from src.scripts.writing_coach_interfaces import create_writing_coach
from src.scripts.writing_coach_interfaces.utils import install_wc_app_path


CSV_FIELDNAMES = [
    'row_id', 'query', 'input_preview', 'input', 'route_orch', 'intent', 'reasoning',
    'segments_count', 'tools_used', 'response', 'suggestions', 'references',
    'research_papers', 'output',
    'folder_source', 'dataset_source', 'route_intended', 'metadata',
]

ROW_DELAY_SECONDS = 5  # Delay between rows to avoid HTTP 429 rate limit errors


def _load_environment() -> None:
    """Load env vars from repo .env so direct script runs get required config."""
    repo_env = Path(__file__).resolve().parents[2] / '.env'
    if repo_env.exists():
        load_dotenv(repo_env)
    else:
        load_dotenv()


def _datapoint_key(query: Any, input_text: Any) -> tuple[str, str]:
    """Content key identifying a datapoint: the (query, input) pair, stripped.

    Used to diff input rows against already-processed output so resume is based
    on content rather than positional row_id. (query, input) — not query alone —
    because the same command/query is commonly reused across different inputs.
    """
    return (str(query or '').strip(), str(input_text or '').strip())


def _already_processed_keys(details_jsonl: Path) -> set:
    """Return the set of (query, input) keys already written to the JSONL output."""
    processed = set()
    if details_jsonl.exists():
        with open(details_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    processed.add(_datapoint_key(record.get('query'), record.get('input')))
                except json.JSONDecodeError:
                    pass
    return processed


def _build_output_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build nested output dict with agent state fields only (no input dataset fields)."""
    return {
        'route': result.get('route', ''),
        'intent': result.get('intent', ''),
        'reasoning': result.get('reasoning', ''),
        'response': result.get('response', ''),
        'segments_count': result.get('segments_count', 0),
        'tools_used': result.get('tools_used', []),
        'suggestions': result.get('suggestions', []),
        'references': result.get('references', []),
        'research_papers': result.get('research_papers', []),
    }


def _serialize_csv_json(value: Any) -> str:
    """Serialize structured values for CSV columns using JSON strings."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _result_to_csv_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Project a full result payload into the flat CSV schema."""
    response = result.get('response', '') or ''
    route_orch = result.get('route_orch', result.get('route', ''))
    tools_used = result.get('tools_used') or []
    if isinstance(tools_used, str):
        tools_used_value = tools_used
    else:
        tools_used_value = ','.join(str(tool) for tool in tools_used)

    return {
        'row_id': result['row_id'],
        'query': result['query'],
        'input_preview': result['input_preview'],
        'input': result.get('input', ''),
        'route_orch': route_orch,
        'intent': result['intent'],
        'reasoning': result['reasoning'],
        'segments_count': result.get('segments_count', 0),
        'tools_used': tools_used_value,
        'response': response,
        'suggestions': _serialize_csv_json(result.get('suggestions', [])),
        'references': _serialize_csv_json(result.get('references', [])),
        'research_papers': _serialize_csv_json(result.get('research_papers', [])),
        'output': json.dumps(_build_output_dict(result), ensure_ascii=False),
        'folder_source': result.get('folder_source', ''),
        'dataset_source': result.get('dataset_source', ''),
        'route_intended': result.get('route_intended', ''),
        'metadata': result.get('metadata', ''),
    }


def _write_csv_row(csv_writer, result: Dict[str, Any]) -> None:
    """Write a single result row to the CSV."""
    csv_writer.writerow(_result_to_csv_row(result))


def _load_jsonl_results(details_jsonl: Path) -> list[Dict[str, Any]]:
    """Read result payloads from JSONL, ignoring malformed lines."""
    results = []
    if not details_jsonl.exists():
        return results

    with open(details_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def _ensure_results_csv_schema(results_csv: Path, details_jsonl: Path) -> None:
    """Upgrade legacy results CSVs so resumed runs include the response column."""
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return

    with open(results_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, [])

    if header == CSV_FIELDNAMES:
        return

    source_results = _load_jsonl_results(details_jsonl)
    if not source_results:
        with open(results_csv, 'r', encoding='utf-8-sig', newline='') as f:
            source_results = list(csv.DictReader(f))

    tmp_path = results_csv.with_suffix(results_csv.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in source_results:
            _write_csv_row(writer, result)

    tmp_path.replace(results_csv)


def _compute_row_result(
    coach,
    row_id: int,
    row: Dict[str, Any],
    n_input_rows: int,
    folder_source: str,
    dataset_source: str,
) -> Optional[Dict[str, Any]]:
    """Run Writing Coach on a single row (with retries) and annotate provenance.

    Returns the result dict, or None if the row is skipped (missing query/input).
    Performs no I/O on the output files — safe to call from worker threads.
    """
    query = row.get('query', '').strip()
    document_text = row.get('input', '').strip()
    row_route_intended = (row.get('route_intended') or '').strip().upper()
    row_metadata = row.get('metadata') or ''

    if not query or not document_text:
        print(f"Row {row_id}: SKIPPED (missing query or input)")
        return None

    print(f"\nRow {row_id}/{n_input_rows}: {query[:60]}...")

    try:
        result = coach.run_query(row_id, query, document_text)
    except Exception as e:
        if row_id == 1 and "orchestrator failed" in str(e):
            # Retry first row once (cold start recovery)
            time.sleep(2)
            result = coach.run_query(row_id, query, document_text)
        else:
            print(f"  Row {row_id} ERROR (attempt 1): {e} — retrying in 10s...")
            time.sleep(10)
            try:
                result = coach.run_query(row_id, query, document_text)
                print(f"  Row {row_id} retry succeeded.")
            except Exception as e2:
                print(f"  Row {row_id} ERROR (attempt 2): {e2} — marking as ERROR.")
                result = {
                    'row_id': row_id,
                    'query': query,
                    'input_preview': document_text[:200],
                    'input': document_text,
                    'route': 'ERROR',
                    'intent': 'error',
                    'reasoning': str(e2),
                    'response': '',
                    'suggestions': [],
                    'references': [],
                    'research_papers': [],
                    'segments_count': 0,
                    'tools_used': [],
                }

    # Annotate with input-dataset provenance fields
    result['folder_source'] = folder_source
    result['dataset_source'] = dataset_source
    result['route_intended'] = row_route_intended
    result['metadata'] = row_metadata
    return result


def _persist_result(result: Dict[str, Any], csv_writer, csv_file, jsonl_file) -> None:
    """Write a result to the CSV + JSONL outputs and flush (no progress is lost)."""
    _write_csv_row(csv_writer, result)
    csv_file.flush()
    jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
    jsonl_file.flush()


def process_csv(
    input_csv: str,
    version: str = 'v3',
    output_dir: str = 'batch_outputs',
    filter_route: Optional[str] = None,
    limit: Optional[int] = None,
    results_csv_override: Optional[str] = None,
    details_jsonl_override: Optional[str] = None,
    wc_app_src: Optional[str] = None,
    workers: int = 1,
) -> None:
    """Process rows in CSV, writing results incrementally after each row.

    Supports resume: rows already present in the JSONL output are skipped.

    Args:
        input_csv: Path to input CSV with 'query', 'input', and optionally 'route' columns.
        version: Writing Coach version key (v2 or v3).
        output_dir: Directory for output files.
        filter_route: If set, only process rows whose 'route' column matches this value
                      (case-insensitive). Pass None to process all rows.
        limit: If set, only consider the first N selected rows after route filtering.
        workers: Number of concurrent Writing Coach queries. 1 (default) keeps the
                 sequential behaviour with inter-row rate-limit delays; >1 runs a
                 thread pool with no delays (writes are serialised with a lock).
    """
    if limit is not None and limit < 0:
        raise ValueError('limit must be >= 0')
    if workers < 1:
        raise ValueError('workers must be >= 1')

    _load_environment()
    install_wc_app_path(wc_app_src or os.getenv('WC_APP_SRC'))
    coach = create_writing_coach(version)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Provenance fields still derive from the input file.
    stem = Path(input_csv).stem
    folder_source = Path(input_csv).resolve().parent.name
    dataset_source = stem
    route_suffix = f'_{filter_route.upper()}' if filter_route else ''

    # Default output filenames embed the WC version and the month-day, e.g.
    # final_results_v3_0612_results.csv. The date keeps runs from different days
    # separate; note this means resume only kicks in for runs on the same day.
    # Allow caller to override output paths (e.g. to append into an existing file).
    base_name = f'final_results_{version}_{datetime.now().strftime("%m%d")}'
    if results_csv_override:
        results_csv = Path(results_csv_override)
        results_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_csv = output_path / f'{base_name}{route_suffix}_results.csv'

    if details_jsonl_override:
        details_jsonl = Path(details_jsonl_override)
        details_jsonl.parent.mkdir(parents=True, exist_ok=True)
    else:
        details_jsonl = output_path / f'{base_name}{route_suffix}_details.jsonl'

    _ensure_results_csv_schema(results_csv, details_jsonl)

    # Resume via content diff: read the (query, input) keys already in the output.
    processed_keys = _already_processed_keys(details_jsonl)

    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    # Apply route filter if requested
    if filter_route:
        rows_to_run = [
            (i + 1, row) for i, row in enumerate(rows)
            if row.get('route', '').strip().upper() == filter_route.upper()
        ]
        print(
            f"\n[{coach.version}] Route filter: '{filter_route.upper()}' — "
            f"{len(rows_to_run)} matching rows out of {len(rows)} total"
        )
    else:
        rows_to_run = [(i + 1, row) for i, row in enumerate(rows)]
        print(f"\n[{coach.version}] Processing all {len(rows)} rows from {input_csv}")

    # Resume: run only the diff — input rows whose (query, input) is not already
    # in the output. Robust to row reordering, unlike positional row_id matching.
    if processed_keys:
        before = len(rows_to_run)
        rows_to_run = [
            (row_id, row) for (row_id, row) in rows_to_run
            if _datapoint_key(row.get('query'), row.get('input')) not in processed_keys
        ]
        print(
            f"[{coach.version}] Resume diff: {len(processed_keys)} datapoint(s) already "
            f"in output; skipping {before - len(rows_to_run)} of {before}, "
            f"{len(rows_to_run)} new to run."
        )

    if limit is not None:
        original_count = len(rows_to_run)
        rows_to_run = rows_to_run[:limit]
        print(
            f"[{coach.version}] Row limit: processing first {len(rows_to_run)} "
            f"of {original_count} selected rows"
        )

    print("=" * 80)

    # Open output files in append mode so partial progress is never lost
    csv_is_new = not results_csv.exists() or results_csv.stat().st_size == 0
    csv_file   = open(results_csv,   'a', newline='', encoding='utf-8')
    jsonl_file = open(details_jsonl, 'a', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if csv_is_new:
        csv_writer.writeheader()

    completed = 0
    total = len(rows_to_run)

    if total == 0:
        csv_file.close()
        jsonl_file.close()
        print("\nNo rows selected for processing.")
    else:
        # Warm up with dummy query to consume cold start (first row CONVERSATION fallback)
        print("\n" + "=" * 80)
        print(f"WARMUP [{coach.version.upper()}]: Running dummy query to initialize graph...")
        try:
            dummy_result = coach.run_query(
                row_id=0,
                query="What is machine learning?",
                document_text="Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on tasks through experience, without being explicitly programmed."
            )
            print(f"  Dummy query completed - route: {dummy_result['route']}")
        except Exception as e:
            print(f"  Dummy query failed (continuing anyway): {e}")
        print("=" * 80 + "\n")

        def _progress(result: Dict[str, Any], done: int) -> None:
            print(f"  Route: {result['route']} | Intent: {result['intent']} | "
                  f"Papers: {len(result['research_papers'])} | "
                  f"Response: {len(result['response'])} chars "
                  f"[{done}/{total} done]")

        try:
            if workers > 1:
                # Parallel mode: thread pool, no inter-row delays (worker count is
                # the rate cap). Queries run concurrently but results are written in
                # the original input order: completions are buffered and the
                # contiguous prefix is flushed as soon as it is ready, so output
                # order matches input and partial progress is still saved on crash.
                from concurrent.futures import ThreadPoolExecutor, as_completed

                print(f"[{coach.version}] Parallel mode: {workers} workers "
                      f"(inter-row delay disabled, output kept in input order)")
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            _compute_row_result, coach, row_id, row,
                            len(rows), folder_source, dataset_source,
                        ): idx
                        for idx, (row_id, row) in enumerate(rows_to_run)
                    }
                    pending: Dict[int, Optional[Dict[str, Any]]] = {}
                    next_idx = 0
                    for future in as_completed(future_to_idx):
                        pending[future_to_idx[future]] = future.result()
                        # Flush every contiguous-from-next_idx result, in order.
                        while next_idx in pending:
                            result = pending.pop(next_idx)
                            next_idx += 1
                            if result is None:
                                continue
                            _persist_result(result, csv_writer, csv_file, jsonl_file)
                            completed += 1
                            _progress(result, completed)
            else:
                # Sequential mode: keep the rate-limit delays between rows.
                for idx, (row_id, row) in enumerate(rows_to_run):
                    result = _compute_row_result(
                        coach, row_id, row, len(rows), folder_source, dataset_source,
                    )
                    if result is None:
                        continue

                    _persist_result(result, csv_writer, csv_file, jsonl_file)
                    completed += 1
                    _progress(result, completed)

                    if idx == 0:  # First row just completed
                        print("  Warming up connections for 5s...")
                        time.sleep(5)

                    # Delay between rows to avoid HTTP 429 rate limit errors
                    if idx < total - 1:
                        print(f"  Waiting {ROW_DELAY_SECONDS}s before next row...")
                        time.sleep(ROW_DELAY_SECONDS)

        finally:
            csv_file.close()
            jsonl_file.close()

    # Build summary from the full JSONL file (includes all previous runs)
    all_results = []
    with open(details_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(all_results)}")

    print("\nRouting breakdown:")
    for route, count in Counter(r['route'] for r in all_results).most_common():
        print(f"  {route:20} {count:3} queries")

    print("\nIntent breakdown:")
    for intent, count in Counter(r['intent'] for r in all_results).most_common():
        print(f"  {intent:20} {count:3} queries")

    print(f"\nOutputs saved:")
    print(f"  Summary : {results_csv}")
    print(f"  Details : {details_jsonl}")

    for _log_file in ["llm_outputs.txt", "llm_prompts.txt"]:
        _log_path = Path(_log_file)
        if _log_path.exists():
            _log_path.unlink()
            print(f"  Deleted : {_log_file}")


if __name__ == "__main__":
    import argparse

    _load_environment()

    parser = argparse.ArgumentParser(
        description="Batch Writing Coach Query Processor (universal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/scripts/store_output.py queries.csv --version v3\n"
            "  python src/scripts/store_output.py queries.csv --version v2 --output my_output\n"
            "  python src/scripts/store_output.py queries.csv --version v3 --route RESEARCH\n"
            "  python src/scripts/store_output.py queries.csv --version v3 --limit 5\n"
            "  python src/scripts/store_output.py queries.csv --version v3 --route RESPOND --output respond_only\n"
            "\nValid route values (must match 'route' column in CSV):\n"
            "  RESEARCH, REVISE_RESEARCH, REVISE_SIMPLE, RESPOND"
        ),
    )
    parser.add_argument('input_csv', help='Path to input CSV with query and input columns')
    parser.add_argument('--version', choices=['v2', 'v3'], default='v3',
                        help='Writing Coach version to run (default: v3)')
    parser.add_argument('--output', default='batch_outputs', help='Output directory (default: batch_outputs)')
    parser.add_argument('--route', default=None, help='Only process rows matching this route value (e.g. RESEARCH, RESPOND)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process the first N selected rows after route filtering (useful for debugging)')
    parser.add_argument('--results-csv', default=None, dest='results_csv',
                        help='Override output CSV path (useful for appending into an existing file)')
    parser.add_argument('--details-jsonl', default=None, dest='details_jsonl',
                        help='Override output JSONL path (useful for appending into an existing file)')
    parser.add_argument('--wc-app-src', default=(os.getenv('WC_APP_SRC') or '').strip('"').strip("'") or None,
                        dest='wc_app_src', help='Path to Writing Coach app root (optional if WC_APP_SRC is set)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of concurrent queries (default: 1 = sequential with '
                             'inter-row delays). >1 enables a thread pool with no delays.')

    args = parser.parse_args()

    process_csv(
        input_csv=args.input_csv,
        version=args.version,
        output_dir=args.output,
        filter_route=args.route,
        limit=args.limit,
        results_csv_override=args.results_csv,
        details_jsonl_override=args.details_jsonl,
        wc_app_src=args.wc_app_src,
        workers=args.workers,
    )