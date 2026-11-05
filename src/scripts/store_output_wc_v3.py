"""Compatibility wrapper that runs the universal batch runner with Writing Coach V3."""

from __future__ import annotations

import argparse
import os

from src.scripts.store_output import process_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch Writing Coach V3 Query Processor (compat wrapper)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/scripts/store_output_wc_v3.py queries.csv\n"
            "  python src/scripts/store_output_wc_v3.py queries.csv --route RESEARCH\n"
            "  python src/scripts/store_output_wc_v3.py queries.csv --output respond_only\n"
            "\nFor version switching, use src/scripts/store_output.py --version <v2|v3>."
        ),
    )
    parser.add_argument('input_csv', help='Path to input CSV with query and input columns')
    parser.add_argument('--output', default='batch_outputs', help='Output directory (default: batch_outputs)')
    parser.add_argument('--route', default=None, help='Only process rows matching this route value (e.g. RESEARCH, RESPOND)')
    parser.add_argument('--results-csv', default=None, dest='results_csv',
                        help='Override output CSV path (useful for appending into an existing file)')
    parser.add_argument('--details-jsonl', default=None, dest='details_jsonl',
                        help='Override output JSONL path (useful for appending into an existing file)')
    parser.add_argument('--wc-app-src', default=(os.getenv('WC_APP_SRC') or '').strip('"').strip("'") or None,
                        dest='wc_app_src', help='Path to Writing Coach app root (optional if WC_APP_SRC is set)')
    parser.add_argument('--max-retries-no-suggestions', type=int, default=0,
                        dest='max_retries_no_suggestions',
                        help='Re-run a query up to N times when WC returns no suggestions '
                             '(likely a misclassification). 0 disables retries (default: 0).')
    args = parser.parse_args()

    process_csv(
        input_csv=args.input_csv,
        version='v3',
        output_dir=args.output,
        filter_route=args.route,
        results_csv_override=args.results_csv,
        details_jsonl_override=args.details_jsonl,
        wc_app_src=args.wc_app_src,
        max_retries_no_suggestions=args.max_retries_no_suggestions,
    )


if __name__ == '__main__':
    main()

