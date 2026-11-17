"""Typed Pydantic agent tools executing under strict PrincipalContext ACL + audit logging."""

from __future__ import annotations

from pydantic import BaseModel, Field

from services.api.state import ServiceContainer
from services.security.acl import PrincipalContext
from services.storage.hybrid_index import SearchHit
from services.storage.postgres import LeaderboardEntry


class SearchCorpusArgs(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    doc_id_filter: str | None = None


class ReadDocumentPageArgs(BaseModel):
    doc_id: str
    page_number: int = Field(default=1, ge=1)


class InspectHeatArgs(BaseModel):
    heat_id: str


class QueryLeaderboardArgs(BaseModel):
    limit: int = Field(default=10, ge=1, le=50)


class Citation(BaseModel):
    source_type: str
    source_id: str
    page_number: int | None = None
    excerpt: str


class StewardToolbox:
    """Binds a `PrincipalContext` and `ServiceContainer` so every tool call is ACL-gated & logged."""

    def __init__(self, container: ServiceContainer, principal: PrincipalContext) -> None:
        self.container = container
        self.principal = principal
        self.tool_trace: list[str] = []

    def query_leaderboard(self, args: QueryLeaderboardArgs) -> list[LeaderboardEntry]:
        self.tool_trace.append("query_leaderboard")
        self.container.audit.record(
            self.principal,
            "agent_tool_call",
            "leaderboard",
            "global",
            "ALLOW",
            "typed_tool_execution",
            metadata={"limit": args.limit},
        )
        return self.container.repo.get_leaderboard(limit=args.limit)

    def inspect_heat_replay(self, args: InspectHeatArgs) -> dict[str, object]:
        self.tool_trace.append("inspect_heat_replay")
        self.container.audit.record(
            self.principal,
            "agent_tool_call",
            "heat",
            args.heat_id,
            "ALLOW",
            "typed_tool_execution",
        )
        rec = self.container.repo.get_heat(args.heat_id, self.principal)
        if not rec:
            return {"found": False, "heat_id": args.heat_id}
        return {
            "found": True,
            "heat_id": rec.heat_id,
            "tenant_id": rec.tenant_id,
            "num_surfers": rec.num_surfers,
            "total_waves": rec.total_waves,
            "total_drop_ins": rec.total_drop_ins,
            "total_collisions": rec.total_collisions,
            "replay_blob_uri": rec.replay_blob_uri,
        }

    def search_private_corpus(self, args: SearchCorpusArgs) -> list[SearchHit]:
        self.tool_trace.append("search_private_corpus")
        self.container.audit.record(
            self.principal,
            "agent_tool_call",
            "corpus",
            args.doc_id_filter or "*",
            "ALLOW",
            "typed_tool_execution",
            query_text=args.query,
        )
        return self.container.index.search(
            query=args.query,
            principal=self.principal,
            top_k=args.top_k,
            doc_id_filter=args.doc_id_filter,
        )

    def read_document_page(self, args: ReadDocumentPageArgs) -> dict[str, object]:
        self.tool_trace.append("read_document_page")
        self.container.audit.record(
            self.principal,
            "agent_tool_call",
            "document_page",
            f"{args.doc_id}#p{args.page_number}",
            "ALLOW",
            "typed_tool_execution",
        )
        doc = self.container.repo.get_document(args.doc_id, self.principal)
        if not doc:
            return {"found": False, "doc_id": args.doc_id}
        page_text = doc.pages.get(args.page_number)
        return {
            "found": page_text is not None,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "page_number": args.page_number,
            "text": page_text or "",
        }
