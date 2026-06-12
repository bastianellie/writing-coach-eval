# writing-coach-eval

Evaluation workspace for Writing Coach V2. The repository prepares evaluation inputs, runs the system in batch, normalizes outputs into a reviewable dataset, and scores those outputs with a dynamic rubrics pipeline.

The two main automation scripts are:

- **`src/dataset_handling/data_pipeline.py`** — collect Writing Coach outputs across all datasets and consolidate them into a single `final_data/all_results.csv`.
- **`src/run_full_eval.py`** — run the complete evaluation pipeline end-to-end against that consolidated file.

## What This Repository Covers

The evaluation flow has four stages:

1. Build or extend route-labeled evaluation data.
2. Run Writing Coach V2 on each row and capture routing plus model outputs.
3. Consolidate those outputs into a single evaluation dataset with intended-route labels.
4. Score the results with a two-stage LLM evaluation pipeline.

Writing Coach routes each example into one of four task types: 

| Route | Purpose |
|---|---|
| `REVISE_SIMPLE` | Edit text without external research |
| `REVISE_RESEARCH` | Revise text and add evidence or citations |
| `RESPOND` | Answer, summarize, compare, or discuss based on given context |
| `RESEARCH` | Search literature and return evidence-grounded findings |

## Setup

Requirements: Python 3.12 and Azure OpenAI credentials.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
```

`WC_APP_SRC` must also be set (or passed via `--wc-app-src`) to point at the Writing Coach app root when running the data pipeline.

---

## Stage 1 — Collect Outputs: `data_pipeline.py`

`src/dataset_handling/data_pipeline.py` automates collecting Writing Coach outputs across all configured datasets and producing the consolidated `final_data/all_results_MMDD.csv`.

```bash
python -m src.dataset_handling.data_pipeline
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--round-dir` | `data_outputs/round_DDMM` | Versioned output base for this run |
| `--wc-app-src` | `$WC_APP_SRC` | Path to the Writing Coach app root |
| `--jobs` | (see below) | JSON array of job objects to override the default list |
| `--skip-store` | — | Skip store_output step and run only reassign |
| `--skip-reassign` | — | Skip reassign step and run only store_output |

### Default jobs

The pipeline runs `store_output` once per job. `data_routes_expanded.csv` is run four times, once per route:

| Input CSV | Output folder | Route filter |
|---|---|---|
| `data/data_routes_expanded.csv` | `research_only` | RESEARCH |
| `data/data_routes_expanded.csv` | `respond_only` | RESPOND |
| `data/data_routes_expanded.csv` | `revise_research_only` | REVISE_RESEARCH |
| `data/data_routes_expanded.csv` | `revise_simple_only` | REVISE_SIMPLE |
| `data/alex-9-extra-responds.csv` | `respond_only` | — |
| `data/manu-10-extra-queries.csv` | `extra10` | — |
| `data/manual_edge_cases_cor.csv` | `cor_edge_cases` | — |
| `data/wc2_eval_21.csv` | `new21` | — |
| `data/data_routes_kiwi_selected167.csv` | `extra167kiwi` | — |

Pass `--jobs` as a JSON array to override with custom datasets:

```bash
python -m src.dataset_handling.data_pipeline \
  --jobs '[{"csv":"data/my.csv","folder":"my_folder","route":"RESEARCH"}]'
```

### Scripts called internally

#### `src/scripts/store_output.py`

Runs Writing Coach V2 on each row of an input CSV and writes results incrementally. The input CSV must have `query`, `input`, and optionally `route` columns. Supports resume via incremental writes.

Can also be run standalone for a single dataset:

```bash
python -m src.scripts.store_output data/my_data.csv \
  --results-csv data_outputs/round_custom/myfolder/results.csv \
  --details-jsonl data_outputs/round_custom/myfolder/details.jsonl \
  --route RESPOND
```

The `--route` argument limits processing to rows matching that route value.

#### `src/dataset_handling/reassign_data_v2.py`

Consolidates per-folder JSONL outputs into a single `final_data/all_results_MMDD.csv`. This is the current recommended consolidation script.

It loads **intended routes from the existing `final_data/all_results.csv` as a gold standard** rather than re-deriving them from orchestrator decisions or hardcoded overrides. This means:

- `route_intended` from a previous reviewed run is preserved automatically.
- No manual overrides need to be re-applied when re-running for a new WC version.
- A first-time run (or a new dataset) still requires a manual review pass to confirm which orchestrator route should be treated as intended.

The older `reassign_data.py` uses hardcoded per-row exceptions and is kept for reference only.

**Outputs:** `final_data/all_results_MMDD.csv` and `final_data/all_results_MMDD.jsonl`, each row carrying a global `row_id`, `route_orch`, `route_intended`, nested `output` payload, and `dataset_source`.

---

## Stage 2 — Run Evaluation: `run_full_eval.py`

`src/run_full_eval.py` orchestrates the full evaluation pipeline end-to-end. By default it runs **all** metrics: the five row-level evaluation/reporting steps below **and** the Giskard dataset-level scans, all in the same output folder. Use `--metrics` to restrict to a subset.

```bash
python -m src.run_full_eval
```

> **Note:** Because Giskard scans run by default, `WC_APP_SRC` must be set (or `--wc-app-src` passed) for a default run, and default runs are slower (they generate adversarial samples and run scan suites). For a fast, scoring-only run, pass the row-level metrics explicitly:
>
> ```bash
> python -m src.run_full_eval --metrics completeness output_relevancy correctness
> ```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--input` | `final_data/all_results.csv` | Input CSV with system outputs |
| `--route-column` | `intended` | `intended` or `orchestrator` |
| `--routes` | (all) | Restrict to specific routes, e.g. `RESEARCH RESPOND` |
| `--metrics` | **all metrics** | Restrict to specific metrics. Row-level: `completeness output_relevancy correctness`. Giskard scans: `potential_harm toxicity resilience`. Groundedness: `groundedness` |
| `--rubrics-mode` | `combined` | `combined` (one prompt for all metrics) or `split` (one prompt per metric); controls the `one_prompt` / `multi_prompt` output segment |
| `--run-name` | `eval_YYYYMMDD_HHMMSS` | Namespace for output files |
| `--limit` | — | Max rows to process (useful for testing) |
| `--output-root` | `eval_data` | Base directory under which the run folder is created |
| `--wc-version` | `v3` | Writing Coach interface version (`v2` or `v3`); sets the `wc<version>` output segment |
| `--wc-app-src` | `$WC_APP_SRC` | Path to the Writing Coach app root (required when Giskard scans run) |
| `--n-samples` | `20` | Seed sample size for Giskard scans |
| `--seed` | `42` | Random seed for Giskard sampling / adversarial generation |
| `--n-adversarial-samples` | `5` | Adversarial samples per Giskard detector |
| `--n-requirements` | `4` | Requirements generated per Giskard detector |
| `--hd-method` | `trustcard` | Hallucination-detection method for the `groundedness` metric |
| `--hd-model` | `gpt-5-mini` | Azure deployment used by the HD detector |
| `--hd-prompt-version` | `V5_anchor_V7` | HD detection prompt version |

Examples:

```bash
# Default: all metrics, including Giskard scans (requires WC_APP_SRC)
python -m src.run_full_eval

# Row-level scoring only (fast, no Giskard / groundedness)
python -m src.run_full_eval --metrics completeness output_relevancy correctness

# Giskard scans only
python -m src.run_full_eval --metrics potential_harm toxicity resilience

# Groundedness (hallucination detection) only
python -m src.run_full_eval --metrics groundedness

# Mixed subset
python -m src.run_full_eval --metrics completeness correctness groundedness
```

### Output location

All outputs land under:

```
{output-root}/wc{version}_{one|multi}_prompt/{route_intended|route_orch}/{timestamp}/
```

The path segments are derived from the run options: `{output-root}` from `--output-root` (default `eval_data`), `wc{version}` from `--wc-version` (default `v3`), `{one|multi}_prompt` from `--rubrics-mode` (`combined` → `one_prompt`, `split` → `multi_prompt`), and the route folder from `--route-column`. A default run therefore lands under `eval_data/wcv3_one_prompt/route_intended/{timestamp}/`.

Key files produced:

| File | Contents |
|---|---|
| `*_results.csv` | Per-row scores and packed JSON fields |
| `*_details.jsonl` | Full generator and judge records |
| `*_all_results_enriched.csv/jsonl` | Eval columns merged back onto original rows |
| `scores_all_natural_synthetic.txt` | Printed score summary for all / natural / synthetic splits |
| `heuristic_scoring_all_natural_synthetic.txt` | Heuristic scoring summary |
| `summary_scores.csv` | Flat summary keyed by `data_origin` × `scope` × `route` (includes `scope="groundedness"` rows when groundedness runs) |
| `metrics_score_combinations/item_level/reasoning_summaries.json` | LLM-generated pattern summaries per metric and score |
| `{potential_harm|toxicity|resilience}/...` | Giskard HTML/JSON/test-suite artifacts for each scan metric that ran (all three by default) |
| `groundedness.json` | Full per-summary HD results (claims, decisions, reasoning) + aggregate |
| `groundedness_summary.csv` | One row per scored summary plus an `OVERALL` aggregate row |

### Steps and scripts called internally

#### Pre-step — Giskard dataset scans

By default `--metrics` includes `potential_harm`, `toxicity`, and `resilience`, so `run_full_eval.py` first runs the matching detector-specific Giskard scan wrappers and writes the generated report artifacts into metric-specific subdirectories under the timestamped run folder. Restrict `--metrics` to row-level metrics only to skip this stage; if only scan metrics are selected, the command exits after this stage.

#### Pre-step — Groundedness (hallucination detection)

When `--metrics` includes `groundedness` (in the default set), `run_full_eval.py` runs hallucination detection over every row whose **orchestrator** route (`route_orch`) is `RESEARCH` or `REVISE_RESEARCH`:

1. **Decompose** the row into "summaries" — the `response` (RESEARCH) or each suggestion's `transformed_text` (REVISE_RESEARCH) — each paired with the structured references it cites.
2. **Extract claims** with a rule (a *claim* is the smallest text span ending at a citation tag) and pair each claim with the abstract/snippet of its cited references.
3. **Classify** each claim/reference pair as `Entailment` / `Contradiction` / `Unverifiable` via the `--hd-model` Azure deployment, producing a per-summary distribution:
   ```json
   "claim_distribution_per_summary": { "entailment(%)": 90.0, "unverifiable(%)": 10.0, "contradiction(%)": 0.0 }
   ```
4. **Aggregate** into a grounded ratio: a summary is *grounded* when `entailment(%) + 0.87 * unverifiable(%) >= 85`; the grounded ratio is the fraction of grounded summaries.

The HD method is swappable behind the `HallucinationDetector` interface in `src/evaluation/hallucination/`; select one with `--hd-method` (currently: `trustcard`). The detector uses this repo's Azure config (`AZURE_OPENAI_*`). Outputs: `groundedness.json`, `groundedness_summary.csv`, and `scope="groundedness"` rows in `summary_scores.csv`. If only `groundedness` (and/or scan metrics) are selected, the command exits after this stage.

#### Step 1 — `src/evaluation/eval_pipeline.py`

Two-stage async LLM evaluation:

1. **Rubrics generation** — given `query`, `input`, and `route`, generates task-specific rubric criteria.
2. **Rubrics judging** — scores the system output against those rubrics.

Runs with a semaphore-limited concurrency, writes incrementally to CSV + JSONL for resume support. Produces the enriched CSV consumed by all downstream steps.

Can be run standalone for row-level rubric scoring only:

```bash
python -m src.evaluation.eval_pipeline \
  --input final_data/all_results.csv \
  --routes RESEARCH RESPOND \
  --concurrency 3 \
  --limit 10
```

#### Step 2 — `src/scripts/split_natural_synthetic.py`

Splits the enriched results into natural and synthetic subsets and recomputes micro/macro score aggregations per route. Can be run standalone against any enriched CSV.

#### Step 3 — `src/scripts/heuristic_scoring.py`

Reconstructs metric-level scores from per-item verdicts using fixed importance weights (low=1, medium=2, high=3). Runs against all, natural, and synthetic subsets. Produces `*_heuristic.csv` files with augmented score columns.

#### Step 4 — `src/evaluation/extract_reasoning_by_score.py`

Extracts reasoning text from the enriched CSV, grouped by metric and score value (0, 1, 2), into item-level JSON files under `metrics_score_combinations/item_level/`.

#### Step 5 — `src/evaluation/summarize_reasoning.py`

Sends the per-metric, per-score reasoning JSONs to an LLM, which summarizes recurring patterns as strengths and weaknesses. Writes `reasoning_summaries.json`.

All five scripts can also be run independently if a partial re-run is needed.

---

## Prompts

Evaluation prompts live in `src/prompts/` and are loaded by `src/evaluation/prompt_loader.py`. Each file uses `{% block system %}` / `{% block prompt %}` blocks — the system block holds stable evaluator behavior, and the prompt block receives row-specific inputs at runtime.

- `rubrics_prompt.txt` — generates dynamic rubrics from the query, input, and route context
- `rubrics_judge_prompt.txt` — scores the system output against those rubrics

Route-specific guidance is supplied from `src/constants/route_prompts.py` and shared metric definitions from `src/constants/metrics_definitions.py`.

---

## Repository Pointers

| Path | Purpose |
|---|---|
| `src/dataset_handling/data_pipeline.py` | End-to-end data collection pipeline |
| `src/run_full_eval.py` | End-to-end evaluation pipeline |
| `src/scripts/store_output.py` | Batch WC V2 inference runner |
| `src/dataset_handling/reassign_data_v2.py` | Consolidates JSONL outputs using gold-standard routes |
| `src/dataset_handling/reassign_data.py` | Legacy consolidation script with hardcoded overrides |
| `src/evaluation/eval_pipeline.py` | Async rubrics evaluation pipeline |
| `src/evaluation/prompt_loader.py` | Prompt parsing and slot injection |
| `src/scripts/split_natural_synthetic.py` | Splits enriched results by data origin |
| `src/scripts/heuristic_scoring.py` | Heuristic scoring from verdict weights |
| `src/evaluation/extract_reasoning_by_score.py` | Extracts reasoning by metric and score |
| `src/evaluation/summarize_reasoning.py` | LLM summarization of reasoning patterns |
| `src/prompts/` | Evaluation prompt templates |
| `src/constants/` | Route prompts, constraints, and metric definitions |
| `src/dataset_handling/` | Dataset construction and consolidation scripts |

## Dependencies

Key packages:

- `langgraph` for Writing Coach graph execution
- `langchain` and `langchain-openai` for async evaluation calls
- `openai` for Azure OpenAI access
- `pandas` and `pyarrow` for dataset handling
- `datasets` and `huggingface_hub` for public data ingestion
- `python-dotenv` for environment loading

