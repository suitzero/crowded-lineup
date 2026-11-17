"""Postgres RLS repository and fast in-memory mirror for heats, leaderboard, and documents."""

from __future__ import annotations

import json
from typing import Any, Protocol

import psycopg
from pydantic import BaseModel, Field

from packages.python_sim.contracts import HeatTelemetryBatch
from services.security.acl import (
    ClearanceLevel,
    DocumentAclMeta,
    PrincipalContext,
    can_principal_access,
)
from services.security.audit import AuditEntry, AuditLogger


class LeaderboardEntry(BaseModel):
    agent_id: str
    policy_type: str
    elo_rating: float = 1500.0
    heats_played: int = 0
    catches: int = 0
    drop_ins: int = 0
    collisions: int = 0


class StoredHeatRecord(BaseModel):
    heat_id: str
    tenant_id: str
    seed: int
    preset: str
    num_surfers: int
    total_waves: int
    total_drop_ins: int
    total_collisions: int
    replay_blob_uri: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    doc_id: str
    tenant_id: str
    title: str
    source_uri: str
    classification: ClearanceLevel
    is_public: bool = False
    allowed_principals: list[str] = Field(default_factory=list)
    custodian: str | None = None
    doc_type: str = "pdf"
    pages: dict[int, str] = Field(default_factory=dict)

    def to_acl(self) -> DocumentAclMeta:
        return DocumentAclMeta(
            doc_id=self.doc_id,
            tenant_id=self.tenant_id,
            classification=self.classification,
            allowed_principals=self.allowed_principals,
            is_public=self.is_public,
        )


class ArenaRepository(Protocol):
    def upsert_heat_and_ratings(
        self, batch: HeatTelemetryBatch, replay_uri: str, principal: PrincipalContext
    ) -> None: ...
    def get_heat(self, heat_id: str, principal: PrincipalContext) -> StoredHeatRecord | None: ...
    def get_leaderboard(self, limit: int = 20) -> list[LeaderboardEntry]: ...
    def upsert_document(self, doc: DocumentRecord) -> None: ...
    def get_document(self, doc_id: str, principal: PrincipalContext) -> DocumentRecord | None: ...
    def list_documents(self, principal: PrincipalContext) -> list[DocumentRecord]: ...


def compute_elo_updates(
    current_ratings: dict[str, float],
    roster_scores: list[tuple[str, float]],
    k_factor: float = 24.0,
) -> dict[str, float]:
    """Multi-player pairwise normalized Elo update across surfers in a heat."""
    n = len(roster_scores)
    if n < 2:
        return {}
    deltas: dict[str, float] = {aid: 0.0 for aid, _ in roster_scores}
    scale = k_factor / (n - 1)
    for i in range(n):
        ai, si = roster_scores[i]
        ri = current_ratings.get(ai, 1500.0)
        for j in range(i + 1, n):
            aj, sj = roster_scores[j]
            rj = current_ratings.get(aj, 1500.0)
            exp_i = 1.0 / (1.0 + 10.0 ** ((rj - ri) / 400.0))
            actual_i = 0.5 if abs(si - sj) < 1e-6 else (1.0 if si > sj else 0.0)
            delta = scale * (actual_i - exp_i)
            deltas[ai] += delta
            deltas[aj] -= delta
    return {aid: round(current_ratings.get(aid, 1500.0) + d, 2) for aid, d in deltas.items()}


class InMemoryArenaRepository:
    """In-memory repository enforcing identical ACL & Elo semantics as Postgres RLS."""

    def __init__(self, audit_logger: AuditLogger) -> None:
        self.audit = audit_logger
        self.heats: dict[str, StoredHeatRecord] = {}
        self.ratings: dict[str, LeaderboardEntry] = {}
        self.documents: dict[str, DocumentRecord] = {}

    def upsert_heat_and_ratings(
        self,
        batch: HeatTelemetryBatch,
        replay_uri: str,
        principal: PrincipalContext,
    ) -> None:
        metrics = batch.metrics
        total_waves = metrics.totalWavesSpawned if metrics else 0
        total_drop_ins = (
            sum(metrics.surferDropIns)
            if metrics
            else sum(1 for e in batch.events if e.eventType == "drop_in_violation")
        )
        total_collisions = (
            sum(metrics.surferCollisions)
            if metrics
            else sum(1 for e in batch.events if e.eventType == "surfer_collision")
        )

        rec = StoredHeatRecord(
            heat_id=batch.heatId,
            tenant_id=batch.tenantId,
            seed=batch.config.seed
            if hasattr(batch.config, "seed")
            else (batch.events[0].seed if batch.events else 0),
            preset=batch.preset,
            num_surfers=batch.config.numSurfers,
            total_waves=total_waves,
            total_drop_ins=total_drop_ins,
            total_collisions=total_collisions,
            replay_blob_uri=replay_uri,
            metrics=metrics.model_dump() if metrics else {},
        )
        self.heats[batch.heatId] = rec

        if metrics:
            scores = [
                (
                    r.agentId,
                    metrics.surferScores[r.surferId]
                    if r.surferId < len(metrics.surferScores)
                    else 0.0,
                )
                for r in batch.agentRoster
            ]
            curr = {
                aid: self.ratings.get(
                    aid, LeaderboardEntry(agent_id=aid, policy_type="unknown")
                ).elo_rating
                for aid, _ in scores
            }
            new_elos = compute_elo_updates(curr, scores)
            for r in batch.agentRoster:
                prev = self.ratings.get(
                    r.agentId,
                    LeaderboardEntry(agent_id=r.agentId, policy_type=r.policyType),
                )
                prev.elo_rating = new_elos.get(r.agentId, prev.elo_rating)
                prev.heats_played += 1
                if r.surferId < len(metrics.surferCatches):
                    prev.catches += metrics.surferCatches[r.surferId]
                    prev.drop_ins += metrics.surferDropIns[r.surferId]
                    prev.collisions += metrics.surferCollisions[r.surferId]
                self.ratings[r.agentId] = prev

        self.audit.record(
            principal,
            "telemetry_ingest",
            "heat",
            batch.heatId,
            "ALLOW",
            "ingested_batch",
            metadata={"events": len(batch.events)},
        )

    def get_heat(self, heat_id: str, principal: PrincipalContext) -> StoredHeatRecord | None:
        rec = self.heats.get(heat_id)
        if not rec:
            return None
        allowed = rec.tenant_id == "public_league" or rec.tenant_id == principal.tenant_id
        self.audit.record(
            principal,
            "sql_query",
            "heat",
            heat_id,
            "ALLOW" if allowed else "DENY",
            "heat_tenant_check",
        )
        return rec if allowed else None

    def get_leaderboard(self, limit: int = 20) -> list[LeaderboardEntry]:
        return sorted(self.ratings.values(), key=lambda x: x.elo_rating, reverse=True)[:limit]

    def upsert_document(self, doc: DocumentRecord) -> None:
        self.documents[doc.doc_id] = doc

    def get_document(self, doc_id: str, principal: PrincipalContext) -> DocumentRecord | None:
        doc = self.documents.get(doc_id)
        if not doc:
            return None
        allowed, reason = can_principal_access(principal, doc.to_acl())
        self.audit.record(
            principal,
            "document_read",
            "document",
            doc_id,
            "ALLOW" if allowed else "DENY",
            reason,
            metadata={"doc_tenant": doc.tenant_id, "classification": int(doc.classification)},
        )
        return doc if allowed else None

    def list_documents(self, principal: PrincipalContext) -> list[DocumentRecord]:
        visible: list[DocumentRecord] = []
        for doc in self.documents.values():
            ok, _ = can_principal_access(principal, doc.to_acl())
            if ok:
                visible.append(doc)
        self.audit.record(
            principal,
            "sql_query",
            "documents",
            "*",
            "ALLOW",
            "rls_list_documents",
            metadata={"visible_count": len(visible)},
        )
        return visible


class PostgresRlsRepository:
    """Real PostgreSQL 16 + pgvector repository enforcing Row-Level Security per transaction."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _apply_rls_session(self, cur: psycopg.Cursor[Any], principal: PrincipalContext) -> None:
        cur.execute("SELECT set_config('app.principal_id', %s, true)", (principal.principal_id,))
        cur.execute("SELECT set_config('app.tenant_id', %s, true)", (principal.tenant_id,))
        cur.execute(
            "SELECT set_config('app.clearance_level', %s, true)", (str(int(principal.clearance)),)
        )

    def insert_audit_entry(self, entry: AuditEntry) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_logs (
                    audit_id, timestamp_iso, trace_id, principal_id, tenant_id,
                    agent_id, action, resource_type, resource_id, decision,
                    reason, query_hash, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    entry.audit_id,
                    entry.timestamp_iso,
                    entry.trace_id,
                    entry.principal_id,
                    entry.tenant_id,
                    entry.agent_id,
                    entry.action,
                    entry.resource_type,
                    entry.resource_id,
                    entry.decision,
                    entry.reason,
                    entry.query_hash,
                    json.dumps(entry.metadata),
                ),
            )
            conn.commit()

    def query_documents_rls(self, principal: PrincipalContext) -> list[dict[str, Any]]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            self._apply_rls_session(cur, principal)
            cur.execute(
                "SELECT doc_id, tenant_id, title, classification, is_public FROM documents ORDER BY doc_id"
            )
            rows = cur.fetchall()
            return [
                {
                    "doc_id": r[0],
                    "tenant_id": r[1],
                    "title": r[2],
                    "classification": r[3],
                    "is_public": r[4],
                }
                for r in rows
            ]
