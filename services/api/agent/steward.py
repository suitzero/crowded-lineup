"""Steward Agent orchestrator combining typed tools, ACL checks, citations, and audit trace."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.api.agent.llm import (
    DeterministicStewardLLM,
    GeminiHttpStewardClient,
    StewardLLMClient,
)
from services.api.agent.tools import (
    Citation,
    InspectHeatArgs,
    QueryLeaderboardArgs,
    SearchCorpusArgs,
    StewardToolbox,
)
from services.api.state import ServiceContainer
from services.security.acl import PrincipalContext


class IntelligenceResponse(BaseModel):
    answer: str
    principal_id: str
    tenant_id: str
    trace_id: str
    tools_invoked: list[str]
    citations: list[Citation] = Field(default_factory=list)
    audit_events_recorded: int


class StewardAgentOrchestrator:
    def __init__(self, container: ServiceContainer) -> None:
        self.container = container
        api_key = container.settings.gemini_api_key.strip()
        self.llm: StewardLLMClient = (
            GeminiHttpStewardClient(api_key=api_key, model=container.settings.gemini_model)
            if api_key
            else DeterministicStewardLLM()
        )

    def handle_query(
        self,
        query: str,
        principal: PrincipalContext,
        heat_id: str | None = None,
        doc_id_filter: str | None = None,
    ) -> IntelligenceResponse:
        toolbox = StewardToolbox(self.container, principal)
        before_audit = len(self.container.audit.list_entries(principal.trace_id))

        hits = toolbox.search_private_corpus(
            SearchCorpusArgs(query=query, top_k=5, doc_id_filter=doc_id_filter)
        )
        lb = toolbox.query_leaderboard(QueryLeaderboardArgs(limit=5))
        heat_info = (
            toolbox.inspect_heat_replay(InspectHeatArgs(heat_id=heat_id)) if heat_id else None
        )

        citations = [
            Citation(
                source_type="document",
                source_id=h.doc_id,
                page_number=h.page_number,
                excerpt=h.snippet[:160],
            )
            for h in hits
        ]
        if heat_info and heat_info.get("found"):
            citations.append(
                Citation(
                    source_type="heat_replay",
                    source_id=str(heat_info["heat_id"]),
                    excerpt=(
                        f"{heat_info['num_surfers']} surfers, "
                        f"{heat_info['total_drop_ins']} drop-ins"
                    ),
                )
            )

        hits_summary = [f"[{h.doc_id} p.{h.page_number}] {h.snippet}" for h in hits]
        lb_summary = [f"{e.agent_id}: Elo {e.elo_rating:.0f}" for e in lb]
        answer = self.llm.synthesize(
            query=query,
            principal=principal,
            hits_summary=hits_summary,
            heat_summary=heat_info,
            leaderboard_summary=lb_summary,
            citations=citations,
        )

        after_audit = len(self.container.audit.list_entries(principal.trace_id))
        return IntelligenceResponse(
            answer=answer,
            principal_id=principal.principal_id,
            tenant_id=principal.tenant_id,
            trace_id=principal.trace_id,
            tools_invoked=toolbox.tool_trace,
            citations=citations,
            audit_events_recorded=after_audit - before_audit,
        )
