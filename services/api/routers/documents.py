"""Document access, ACL-filtered hybrid search, and steward-gated audit log endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.api.deps import ContainerDep, PrincipalDep

router = APIRouter(prefix="/v1", tags=["documents"])


class CorpusSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    doc_id_filter: str | None = None


@router.get("/documents")
def list_authorized_documents(
    principal: PrincipalDep,
    container: ContainerDep,
) -> dict[str, object]:
    docs = container.repo.list_documents(principal)
    return {
        "principal_id": principal.principal_id,
        "tenant_id": principal.tenant_id,
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "tenant_id": d.tenant_id,
                "classification": int(d.classification),
                "custodian": d.custodian,
                "pages": len(d.pages),
            }
            for d in docs
        ],
    }


@router.get("/documents/{doc_id}")
def read_document(
    doc_id: str,
    principal: PrincipalDep,
    container: ContainerDep,
) -> dict[str, object]:
    if doc_id not in container.repo.documents:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = container.repo.get_document(doc_id, principal)
    if doc is None:
        raise HTTPException(status_code=403, detail="Access denied by document ACL")

    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "tenant_id": doc.tenant_id,
        "classification": int(doc.classification),
        "custodian": doc.custodian,
        "pages": doc.pages,
    }


@router.post("/documents/search")
def search_documents(
    req: CorpusSearchRequest,
    principal: PrincipalDep,
    container: ContainerDep,
) -> dict[str, object]:
    hits = container.index.search(
        query=req.query,
        principal=principal,
        top_k=req.top_k,
        doc_id_filter=req.doc_id_filter,
    )
    return {
        "principal_id": principal.principal_id,
        "trace_id": principal.trace_id,
        "hits": [h.model_dump() for h in hits],
    }


@router.get("/audit/logs")
def get_audit_logs(
    principal: PrincipalDep,
    container: ContainerDep,
) -> dict[str, object]:
    if "steward" not in principal.roles:
        container.audit.record(
            principal,
            "sql_query",
            "audit_logs",
            "*",
            "DENY",
            "steward_role_required",
        )
        raise HTTPException(status_code=403, detail="Steward role required to inspect audit logs")

    entries = container.audit.list_entries()
    return {"count": len(entries), "entries": [e.model_dump() for e in entries[-100:]]}
