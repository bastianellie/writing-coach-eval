"""TrustCard hallucination detector (claim-pattern + Azure LLM).

Concrete :class:`HallucinationDetector` that owns the whole pipeline for one
summary: it remaps citations to integers (reusing
:func:`src.metrics.accuracy.preprocess_references`), splits the text into cited
claims with the copied claim pattern, pairs each claim with the abstract/snippet
of the references it cites, classifies each pair with this repo's Azure OpenAI
deployment using the copied ``V5_anchor_V7`` prompts, and returns the per-summary
label distribution.

The classification logic (prompt selection, JSON parsing, label normalisation)
is a self-contained reimplementation of co-researcher-one's
``trustcard_src/claim_detector.py`` — no cross-repo import.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.metrics.accuracy import preprocess_references
from src.evaluation.hallucination.base import (
    CONTRADICTION,
    ENTAILMENT,
    UNVERIFIABLE,
    HallucinationDetector,
    SummaryScore,
    register_detector,
)
from src.evaluation.hallucination.claim_extraction import (
    extract_citation_ids,
    extract_claims,
    reference_text,
)
from src.evaluation.hallucination.prompts import HD_DETECTION_PROMPTS

logger = logging.getLogger(__name__)

DEFAULT_HD_MODEL = "gpt-5-mini"
DEFAULT_HD_PROMPT_VERSION = "V5_anchor_V7"
DEFAULT_HD_TEMPERATURE = 1.0  # safe for reasoning models (e.g. gpt-5-mini)
DEFAULT_HD_MAX_TOKENS = 4000
DEFAULT_HD_WORKERS = int(os.getenv("HALU_DETECTOR_WORKERS", "8"))


def normalize_label(label: str) -> str:
    """Normalise a raw classification string to a canonical 3-class label."""
    if not label:
        return UNVERIFIABLE
    value = str(label).strip().lower()
    if "entail" in value:
        return ENTAILMENT
    if "contra" in value:
        return CONTRADICTION
    return UNVERIFIABLE


def parse_detection_response(response: str) -> dict:
    """Parse the detector's JSON response, tolerating fences / surrounding text.

    Reimplemented from the HD rubric runner's ``_parse_response``.
    """
    if not response:
        return {}

    candidates = [response.strip()]
    fenced = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response.strip(), flags=re.I)
    if fenced and fenced != candidates[0]:
        candidates.append(fenced.strip())
    first_brace = response.find("{")
    last_brace = response.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(response[first_brace : last_brace + 1].strip())

    for candidate in candidates:
        # The prompt's JSON example uses curly quotes; normalise them so the
        # model's (occasional) verbatim echo still parses.
        candidate = candidate.replace("“", '"').replace("”", '"')
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return {}


def _flatten_reasoning(parsed: dict) -> str:
    """Build a single reasoning string from the parsed response."""
    reasoning = parsed.get("reasoning", "")
    if isinstance(reasoning, list):
        parts = []
        for item in reasoning:
            if isinstance(item, dict):
                text = item.get("reasoning") or ""
                ref = item.get("ref")
                parts.append(f"[{ref}] {text}" if ref is not None else str(text))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(reasoning)


@register_detector("trustcard")
class TrustCardDetector(HallucinationDetector):
    """Claim-pattern extraction + Azure-LLM claim/reference classification."""

    name = "trustcard"

    def __init__(
        self,
        model: str = DEFAULT_HD_MODEL,
        prompt_version: str = DEFAULT_HD_PROMPT_VERSION,
        temperature: float = DEFAULT_HD_TEMPERATURE,
        max_tokens: int = DEFAULT_HD_MAX_TOKENS,
        workers: int = DEFAULT_HD_WORKERS,
    ) -> None:
        if prompt_version not in HD_DETECTION_PROMPTS:
            raise ValueError(
                f"Unknown HD prompt version '{prompt_version}'. "
                f"Available: {', '.join(HD_DETECTION_PROMPTS)}"
            )
        self.model = model
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.workers = max(1, workers)
        self._system_prompt, self._user_template = HD_DETECTION_PROMPTS[prompt_version]
        self._llm: AzureChatOpenAI | None = None

    # -- LLM ---------------------------------------------------------------
    def _get_llm(self) -> AzureChatOpenAI:
        if self._llm is None:
            self._llm = AzureChatOpenAI(
                azure_deployment=self.model,
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_retries=3,
            )
        return self._llm

    def _classify(self, claim_text: str, ref_block: str) -> tuple[str, str]:
        """Classify a single (claim, reference) pair -> (label, reasoning)."""
        user_prompt = self._user_template.format(
            reference=ref_block,
            claim=claim_text,
            context_metadata={},
        )
        try:
            response = self._get_llm().invoke(
                [
                    SystemMessage(content=self._system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            ).content
        except Exception as e:  # noqa: BLE001 — degrade to Unverifiable on failure
            logger.error("HD classification failed: %s", e)
            return UNVERIFIABLE, f"Claim verification failed: {e}"

        parsed = parse_detection_response(response)
        label = normalize_label(parsed.get("classification", UNVERIFIABLE))
        return label, _flatten_reasoning(parsed)

    # -- Interface ---------------------------------------------------------
    def score_summary(self, text: str, references: list[dict]) -> SummaryScore:
        # 1. Remap citations to integers and expand references per occurrence.
        processed = preprocess_references({"response": text, "references": references or []})
        summary = processed["response"]
        # index (int) -> reference abstract/snippet text
        ref_text_by_index: dict[int, str] = {}
        for ref in processed["references"]:
            idx = ref.get("index")
            if idx is not None:
                ref_text_by_index[int(idx)] = reference_text(ref)

        # 2. Split into cited claims and pair each with its reference text.
        pairs: list[tuple[str, str, str]] = []  # (claim_text, ref_block, ref_block)
        for claim in extract_claims(summary):
            cited_indexes: list[int] = []
            for raw_id in extract_citation_ids(claim):
                if raw_id.isdigit():
                    cited_indexes.append(int(raw_id))
            ref_lines = [
                f"[{i}]: {ref_text_by_index[i]}"
                for i in cited_indexes
                if i in ref_text_by_index and ref_text_by_index[i]
            ]
            if not ref_lines:
                continue  # citation present but no resolvable reference text
            ref_block = "\n".join(ref_lines)
            pairs.append((claim, ref_block, ref_block))

        # 3. Classify each pair (in parallel).
        claims: list[dict] = []
        if pairs:
            max_workers = max(1, min(self.workers, len(pairs)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                decisions = list(
                    executor.map(lambda p: self._classify(p[0], p[1]), pairs)
                )
            for (claim_text, ref_block, _), (label, reasoning) in zip(pairs, decisions):
                claims.append(
                    {
                        "claim": claim_text,
                        "claim_decision": label,
                        "reasoning": reasoning,
                        "reference": ref_block,
                    }
                )

        # 4. Build the per-summary label distribution.
        distribution = _distribution(claims)
        return SummaryScore(
            claim_distribution_per_summary=distribution,
            claims=claims,
            n_cited_claims=len(claims),
        )


def _distribution(claims: list[dict]) -> dict[str, float]:
    """Percentage of claims per label (keys match grounded_ratio.py)."""
    n = len(claims)
    if n == 0:
        return {"entailment(%)": 0.0, "unverifiable(%)": 0.0, "contradiction(%)": 0.0}
    counts = {ENTAILMENT: 0, UNVERIFIABLE: 0, CONTRADICTION: 0}
    for c in claims:
        counts[c["claim_decision"]] = counts.get(c["claim_decision"], 0) + 1
    return {
        "entailment(%)": round(counts[ENTAILMENT] / n * 100, 1),
        "unverifiable(%)": round(counts[UNVERIFIABLE] / n * 100, 1),
        "contradiction(%)": round(counts[CONTRADICTION] / n * 100, 1),
    }
