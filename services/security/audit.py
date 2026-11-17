"""Append-only structured audit logger for every SQL, search, blob, and agent access."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from services.security.acl import PrincipalContext

AuditDecision = Literal["ALLOW", "DENY"]
AuditAction = Literal[
    "sql_query",
    "vector_search",
    "blob_read",
    "doc_ingest",
    "agent_tool_call",
    "telemetry_ingest",
    "document_read",
]


def hash_query_text(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


class AuditEntry(BaseModel):
    audit_id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    trace_id: str
    principal_id: str
    tenant_id: str
    agent_id: str | None = None
    action: AuditAction
    resource_type: str
    resource_id: str
    decision: AuditDecision
    reason: str
    query_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Thread-safe, append-only audit recorder with optional JSONL sink."""

    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._jsonl_path = jsonl_path
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        principal: PrincipalContext,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        decision: AuditDecision,
        reason: str,
        query_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            trace_id=principal.trace_id,
            principal_id=principal.principal_id,
            tenant_id=principal.tenant_id,
            agent_id=principal.agent_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            decision=decision,
            reason=reason,
            query_hash=hash_query_text(query_text) if query_text else None,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.model_dump()) + "\n")
        return entry

    def list_entries(self, trace_id: str | None = None) -> list[AuditEntry]:
        if trace_id is None:
            return list(self._entries)
        return [e for e in self._entries if e.trace_id == trace_id]
