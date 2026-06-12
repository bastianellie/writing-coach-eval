"""Domain classification script using LiteLLM.

Reads a CSV file with `query` and `input` columns, classifies each row into
one of the predefined domain labels using an LLM, and saves the updated CSV
with a `domain` column to the specified output path.

Usage
-----
    python classify_domain.py <input_csv> <output_csv>

Example
-------
    python classify_domain.py data/input.csv data/output.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv
import litellm

load_dotenv()

# ---------------------------------------------------------------------------
# Validate required env vars
# ---------------------------------------------------------------------------

REQUIRED = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
missing = [v for v in REQUIRED if not os.getenv(v)]
if missing:
    print(f"[ERROR] Missing environment variables: {', '.join(missing)}")
    print("Make sure they are set in your .env file.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# LiteLLM configuration
# ---------------------------------------------------------------------------

litellm.drop_params = True

MODEL = "azure/gpt-5-chat"

DOMAIN_LABELS = [
    "Computer Science",
    "Physics",
    "Biochemistry",
    "Humanities & Social Sciences",
    "General",
]

SYSTEM_PROMPT = """\
You are a domain classification assistant. Given a text, your task is to classify \
the domain to which it belongs.

You must output exactly one label from the following list (and nothing else):
- Computer Science
- Physics
- Biochemistry
- Humanities & Social Sciences
- General

Rules:
1. Only assign a specific domain (Computer Science, Physics, Biochemistry, or \
Humanities & Social Sciences) when the text contains CLEAR and EXPLICIT evidence \
that it belongs to that domain — such as specific terminology, concepts, or subject \
matter that is unambiguously tied to that field.
2. Do NOT assign "Humanities & Social Sciences" just because the text uses academic \
or scholarly language (e.g. questions about evidence, methodology, literature gaps, \
argumentation quality, or critical analysis). Such texts should be classified as \
"General" unless the subject matter itself is explicitly about a humanities or social \
science topic.
3. Use "General" for all other cases: when the content is general/non-specific, \
when it uses academic framing without a clear domain subject, or when it does not \
belong to any of the named categories.
4. Output ONLY the label — no explanation, no punctuation, no extra text.
"""

USER_PROMPT_TEMPLATE = """\
Text: {text}

Domain label:"""

# System prompt for input text: tries harder to find a domain, including outside the predefined list
INPUT_SYSTEM_PROMPT = """\
You are a domain classification assistant. Given a text, your task is to identify \
the academic or scientific domain it belongs to.

Follow this decision process in order:

Step 1 — Check against the predefined domains.
Try to assign the text to one of these domains if there is CLEAR and EXPLICIT evidence \
(specific terminology, named concepts, or subject matter unambiguously tied to the field):
- Computer Science
- Physics
- Biochemistry
- Humanities & Social Sciences

Important: Do NOT assign "Humanities & Social Sciences" just because the text uses \
academic or scholarly language (e.g. questions about evidence, methodology, literature \
gaps, argumentation quality, or critical analysis). Only assign it if the subject matter \
itself is explicitly about a humanities or social science topic.

Step 2 — If none of the predefined domains apply, consider whether the text clearly \
belongs to another well-known, broad academic domain (e.g. Mathematics, Medicine, \
Economics, Law, Chemistry, Engineering, Biology, Environmental Science). Only do this \
if you have strong and clear evidence. Do not invent narrow or overly specific domains — \
always prefer the most common, widely recognised label. Use consistent naming \
(e.g. always "Mathematics", not "Math" or "Maths").

Step 3 — If there is no clear evidence of any specific domain, output: General

Output ONLY the domain label — no explanation, no punctuation, no extra text.
"""


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------


def _call_llm(text: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Call the LLM to classify the domain of a single piece of text."""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(text=text),
        },
    ]

    response = litellm.completion(
        model=MODEL,
        messages=messages,
        max_tokens=16,
        temperature=0.0,
        api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )

    raw_label = response.choices[0].message.content.strip()

    # Normalise: pick the label that best matches the raw output (predefined list only)
    for label in DOMAIN_LABELS:
        if label.lower() in raw_label.lower():
            return label

    # For input text, the LLM may return a valid domain outside the predefined list —
    # return it as-is if it looks like a plausible single label (no newlines, not too long)
    if system_prompt is INPUT_SYSTEM_PROMPT:
        if raw_label and "\n" not in raw_label and len(raw_label) <= 50:
            return raw_label

    # Fallback if the model returned something unexpected
    print(
        f"[WARNING] Unexpected label from LLM: '{raw_label}'. Defaulting to 'General'.",
        file=sys.stderr,
    )
    return "General"


def classify_domain(query: str, input_text: str) -> tuple[str, str]:
    """Classify the domain of the query and the input text independently.

    The query uses the conservative prompt (predefined domains + General).
    The input text uses the extended prompt that also allows domains outside
    the predefined list when there is clear evidence.
    """
    return _call_llm(query, SYSTEM_PROMPT), _call_llm(input_text, INPUT_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify rows of a CSV file into domain labels using an LLM."
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_csv", help="Path to save the output CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load CSV
    print(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    for col in ("query", "input"):
        if col not in df.columns:
            print(f"[ERROR] Required column '{col}' not found in the CSV.")
            sys.exit(1)

    total = len(df)
    print(f"Classifying {total} rows...")

    query_domains: list[str] = []
    input_domains: list[str] = []
    for idx, row in df.iterrows():
        query = str(row["query"]) if pd.notna(row["query"]) else ""
        input_text = str(row["input"]) if pd.notna(row["input"]) else ""

        # Handle Synthetic-* dataset sources without calling the LLM for query
        dataset_source = str(row.get("dataset_source", "")) if "dataset_source" in df.columns else ""
        synthetic_suffix = None
        if dataset_source.startswith("Synthetic-"):
            synthetic_suffix = dataset_source[len("Synthetic-"):]

        if synthetic_suffix is not None:
            if synthetic_suffix == "Generic":
                # Query is Generic; still classify input via LLM (extended prompt)
                query_label = "General"
                try:
                    input_label = _call_llm(input_text, INPUT_SYSTEM_PROMPT)
                except Exception as exc:
                    print(
                        f"[WARNING] Failed to classify input for row {idx}: {exc}. Defaulting to 'General'.",
                        file=sys.stderr,
                    )
                    input_label = "General"
            else:
                # Use the domain embedded in the source name for both, no LLM call
                query_label = synthetic_suffix
                input_label = synthetic_suffix
        else:
            try:
                query_label, input_label = classify_domain(query, input_text)
            except Exception as exc:
                print(
                    f"[WARNING] Failed to classify row {idx}: {exc}. Defaulting to 'General'.",
                    file=sys.stderr,
                )
                query_label, input_label = "General", "General"

        query_domains.append(query_label)
        input_domains.append(input_label)
        print(f"  [{idx + 1}/{total}] query -> {query_label}  |  input -> {input_label}")

    df["query_domain"] = query_domains
    df["input_domain"] = input_domains

    # Save output
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(args.output_csv, index=False)
    print(f"\nDone! Output saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
