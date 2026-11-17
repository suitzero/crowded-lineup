"""Steward & Strategy Intelligence query endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from services.api.agent.steward import IntelligenceResponse, StewardAgentOrchestrator
from services.api.deps import ContainerDep, PrincipalDep

router = APIRouter(prefix="/v1", tags=["intelligence"])


class IntelligenceQueryRequest(BaseModel):
    query: str
    heat_id: str | None = None
    doc_id_filter: str | None = None


@router.post("/intelligence/query", response_model=IntelligenceResponse)
def query_steward_intelligence(
    req: IntelligenceQueryRequest,
    principal: PrincipalDep,
    container: ContainerDep,
) -> IntelligenceResponse:
    orchestrator = StewardAgentOrchestrator(container)
    return orchestrator.handle_query(
        query=req.query,
        principal=principal,
        heat_id=req.heat_id,
        doc_id_filter=req.doc_id_filter,
    )
