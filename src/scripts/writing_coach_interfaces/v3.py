from __future__ import annotations

import logging
import re
from typing import Any, Dict

from src.scripts.writing_coach_interfaces.base import WritingCoachInterface
from src.scripts.writing_coach_interfaces.utils import install_invoke_llm_shim

logger = logging.getLogger(__name__)


# Mirror the production blueprint's HTML sniff so we pick the same builder
# (ManuscriptManager.from_html vs. the markdown constructor).
_HTML_TAG_RE = re.compile(
    r"</?(?:p|div|span|h[1-6]|ul|ol|li|table|thead|tbody|tr|td|th|"
    r"blockquote|pre|code|br|hr|strong|em|b|i|a|section|article|figure)"
    r"(?:\s|/?>)",
    re.IGNORECASE,
)


def _looks_like_html(text: str) -> bool:
    return bool(_HTML_TAG_RE.search(text or ""))


class WritingCoachV3Interface(WritingCoachInterface):
    """Writing Coach V3 adapter with standalone preset graph initialization."""

    def __init__(self) -> None:
        self._preset_config: Dict[str, Any] = {}
        self._graph = None
        # Lazily-loaded (ManuscriptManager, get_wc_summarizer); None until first
        # use, then either the real symbols or (None, None) if the running WC
        # app doesn't ship the writing_coach positioner package.
        self._manuscript_deps: tuple | None = None
        self._initialize()

    @property
    def version(self) -> str:
        return "v3"

    def _initialize_writing_coach_v3_only(self) -> None:
        from src.graph_builder import GraphBuilder
        from src.graph_nodes.writing_coach_nodes import (
            output_node,
            research_response_node,
            research_transform_node,
            revision_explanation_node,
            search_node,
            search_router,
            segment_analysis_node,
            segment_analysis_router,
            simple_transform_node,
            wc_orchestrator_node,
            wc_orchestrator_router,
            wc_respond_node,
        )
        from src.graph_presets import register_preset
        from src.state_definitions import WritingCoachV2State

        builder = GraphBuilder(WritingCoachV2State)
        builder.set_display_name("Writing Coach V3")
        builder.set_description("Conversational writing coach with hybrid graph architecture (GPT-5.4)")

        for name, fn in [
            ("wc3/orchestrator", wc_orchestrator_node),
            ("wc3/segment_analysis", segment_analysis_node),
            ("wc3/copilot_search", search_node),
            ("wc3/research_response", research_response_node),
            ("wc3/research_transform", research_transform_node),
            ("wc3/simple_transform", simple_transform_node),
            ("wc3/revision_explanation", revision_explanation_node),
            ("wc3/respond", wc_respond_node),
            ("wc3/output", output_node),
        ]:
            builder.register_node_function(name, fn)
            builder.add_node(name)

        # Routers still return wc2/* labels in current app code; map to wc3 nodes here.
        builder.add_edge("START", "wc3/orchestrator")
        builder.add_conditional_edge(
            "wc3/orchestrator",
            wc_orchestrator_router,
            {
                "wc2/segment_analysis": "wc3/segment_analysis",
                "wc2/respond": "wc3/respond",
            },
        )

        builder.add_conditional_edge(
            "wc3/segment_analysis",
            segment_analysis_router,
            {
                "wc2/copilot_search": "wc3/copilot_search",
                "wc2/simple_transform": "wc3/simple_transform",
                "wc2/revision_explanation": "wc3/revision_explanation",
                "wc2/respond": "wc3/respond",
            },
        )
        builder.add_conditional_edge(
            "wc3/copilot_search",
            search_router,
            {
                "wc2/research_response": "wc3/research_response",
                "wc2/research_transform": "wc3/research_transform",
            },
        )

        builder.add_edge("wc3/research_response", "wc3/output")
        builder.add_edge("wc3/research_transform", "wc3/revision_explanation")
        builder.add_edge("wc3/simple_transform", "wc3/revision_explanation")
        builder.add_edge("wc3/revision_explanation", "wc3/output")
        builder.add_edge("wc3/respond", "wc3/output")
        builder.add_edge("wc3/output", "END")

        register_preset("writing_coach_v3", builder)
        print("Writing Coach V3 preset initialized (standalone)")

    def _initialize(self) -> None:
        from src.graph_presets import get_preset
        from src.presets_config import get_preset_config

        print("Initializing Writing Coach V3...")
        install_invoke_llm_shim()
        self._preset_config = get_preset_config("writing_coach_v3")
        self._initialize_writing_coach_v3_only()
        builder = get_preset("writing_coach_v3")
        self._graph = builder.build_without_checkpointing()
        # Resolve the optional positioner deps once on the main thread so the
        # import (and any "not available" warning) happens deterministically
        # before parallel per-row processing starts.
        self._load_manuscript_deps()

    def _load_manuscript_deps(self) -> tuple:
        """Import the optional ``writing_coach`` positioner deps once.

        Returns ``(ManuscriptManager, get_wc_summarizer)``, or ``(None, None)``
        when the running WC app doesn't ship the ``writing_coach`` package
        (e.g. the baseline app without the hierarchical positioner). Cached so
        we don't retry the import — or re-log the warning — on every row.
        """
        if self._manuscript_deps is None:
            try:
                from src.writing_coach.llm_summarizer import get_wc_summarizer
                from src.writing_coach.manuscript import ManuscriptManager

                self._manuscript_deps = (ManuscriptManager, get_wc_summarizer)
            except ImportError as exc:
                logger.warning(
                    "writing_coach positioner module not available (%s); using "
                    "the legacy segment-analysis path (manuscript=None). This is "
                    "expected when running against the baseline app.",
                    exc,
                )
                self._manuscript_deps = (None, None)
        return self._manuscript_deps

    def _build_manuscript(self, document_text: str):
        """Build a ManuscriptManager so segment_analysis takes the positioner
        path (not the legacy anchor path). Without this, the trained
        hierarchical positioner never runs — segment_analysis_node falls back to
        `_legacy_segment_analysis` whenever `state['manuscript']` is None, so
        placement would ignore the positioner entirely. Mirrors the production
        blueprint: summarizer uses the preset's `segment_analysis` model.

        Returns ``None`` when the running WC app lacks the ``writing_coach``
        package (baseline app), which keeps segment_analysis on its legacy path.
        """
        manuscript_cls, get_wc_summarizer = self._load_manuscript_deps()
        if manuscript_cls is None:
            return None

        summarizer_model = self._preset_config.get("models", {}).get("segment_analysis")
        summarizer = (
            get_wc_summarizer("writing_coach_v3", summarizer_model)
            if summarizer_model
            else None
        )
        if _looks_like_html(document_text):
            return manuscript_cls.from_html(document_text, summarizer=summarizer)
        return manuscript_cls(document_text, summarizer=summarizer)

    def run_query(self, row_id: int, query: str, document_text: str) -> Dict[str, Any]:
        from src.state_definitions import WritingCoachV2State

        initial_state: WritingCoachV2State = {
            "message": query,
            "document_text": document_text,
            "selected_text": None,
            "conversation_history": [],
            "prior_references": [],
            "conversation_id": f"batch_{row_id}",
            "conversation_turn": 1,
            "streaming": False,
            "writer": None,
            "preset": "writing_coach_v3",
            "model_config": self._preset_config.get("models", {}),
            "prompt_versions": self._preset_config.get("prompts", {}),
            "parameters": self._preset_config.get("parameters", {}),
            "manuscript": self._build_manuscript(document_text),
            "suggestions": [],
            "references": [],
            "research_papers": [],
        }

        final_state = self._graph.invoke(initial_state)
        return self.build_result_payload(
            row_id=row_id,
            query=query,
            document_text=document_text,
            final_state=final_state,
        )

