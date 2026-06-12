"""Rule-based claim extraction.

The claim-splitting rule is copied from co-researcher-one
``trustcard_src/claim_extractor.py`` (``RuleBasedClaimExtractor``) and its
citation pattern from ``src/config.py`` (``CITATION_PATTERN``), so this repo
carries no cross-repo import. A *claim* is the smallest text span that ends at a
citation tag; each claim is then paired with the references it cites.

This module operates on text whose citations have already been remapped to
sequential integers by :func:`src.metrics.accuracy.preprocess_references`
(e.g. ``... improves outcomes [1, 2].``).
"""

from __future__ import annotations

import re

from markdown import markdown
from bs4 import BeautifulSoup


# Copied from co-researcher-one src/config.py:
#   CITATION_PATTERN = r'\[([CD]?[0-9<hyphens>]+)\]'
# The leading [CD]? is optional, so this matches both the original
# "C13-1-1-3" reference IDs and the integer tags produced by preprocessing.
_HYPHEN_CHARS = r"\-‐‑‒–—―"
CITATION_PATTERN = rf"\[([CD]?[0-9{_HYPHEN_CHARS}]+)\]"

# Core id pattern (contents of the bracket capture group), used to build a
# comma-separated citation pattern like "[1, 2, 3]".
_CORE_ID = rf"[CD]?[0-9{_HYPHEN_CHARS}]+"
_CITATION_GROUP = rf"\[((?:{_CORE_ID})(?:,\s*{_CORE_ID})*)\]"

# A claim = smallest span ending at a citation, terminated by sentence
# punctuation or a table-cell / line boundary. Copied from
# RuleBasedClaimExtractor.__init__.
CLAIM_PATTERN = re.compile(
    rf".+?{_CITATION_GROUP}(?:[.!?]|(?=\s*(?:\||$)))",
    re.DOTALL,
)
CITATION_GROUP_PATTERN = re.compile(_CITATION_GROUP)


def _markdown_to_text(markdown_text: str) -> str:
    """Flatten markdown to plain text lines (paragraphs, list items, cells)."""
    html = markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    text_lines = []
    for tag in soup.find_all(["p", "li", "td"]):
        line = tag.get_text().strip()
        if line:
            text_lines.append(line)
    return "\n".join(text_lines)


def _remove_tail_references(text: str) -> str:
    """Strip a trailing block of ``[d-d-d] ...`` reference lines, if present."""
    remove_reference_pattern = re.compile(r"^\[\d+-\d+-\d+\]\s+.*$", re.MULTILINE)
    lines = text.strip().splitlines()
    i = len(lines)
    while i > 0 and remove_reference_pattern.match(lines[i - 1]):
        i -= 1
    return "\n".join(lines[:i]).strip()


def extract_claims(summary: str) -> list[str]:
    """Extract all cited-claim spans from a (markdown) summary, in order."""
    text = _markdown_to_text(summary)
    text = _remove_tail_references(text)
    claims: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for match in CLAIM_PATTERN.finditer(line):
            claims.append(match.group(0).strip())
    return claims


def extract_citation_ids(claim: str) -> list[str]:
    """Return the citation ids (as strings) cited in a single claim span.

    Handles comma-separated groups, e.g. ``"[1, 2, 3]"`` -> ``["1", "2", "3"]``.
    """
    ids: list[str] = []
    for group in CITATION_GROUP_PATTERN.findall(claim):
        for cid in group.split(","):
            cid = cid.strip()
            if cid:
                ids.append(cid)
    return ids


# Tool ids whose full-text results carry chunks in ``inner_hits`` rather than a
# top-level abstract. Mirrors co-researcher-one builders._extract_abstract.
_FULLTEXT_TOOLS = {"SD_FULLTEXT_VECTOR", "SD_FULLTEXT_LEXICAL"}


def reference_text(ref: dict) -> str:
    """Extract the abstract (or snippet) string from a structured reference dict.

    Copied from co-researcher-one ``builders._extract_abstract``: prefer an
    explicit ``abstract``/``abs`` field; for full-text results without one, fall
    back to the first (most relevant) ``inner_hits`` chunk.
    """
    abstract = ref.get("abstract") or ref.get("abs")
    if not abstract:
        inner_hits = ref.get("inner_hits") or []
        if isinstance(inner_hits, list) and inner_hits and isinstance(inner_hits[0], dict):
            hit = inner_hits[0]
            abstract = hit.get("text") or hit.get("chunk") or ""
    return abstract or ""
