"""Telemetry ingestion, replay retrieval, and live streaming window metrics endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from packages.python_sim.contracts import HeatTelemetryBatch
from services.api.deps import ContainerDep, PrincipalDep
from services.security.acl import ClearanceLevel, DocumentAclMeta
from services.storage.blob import BlobAccessDeniedError

router = APIRouter(prefix="/v1", tags=["telemetry"])


@router.post("/telemetry/batches")
def ingest_telemetry_batch(
    batch: HeatTelemetryBatch,
    principal: PrincipalDep,
    container: ContainerDep,
) -> dict[str, object]:
    replay_acl = DocumentAclMeta(
        doc_id=f"replay-{batch.heatId}",
        tenant_id=batch.tenantId,
        classification=ClearanceLevel.PUBLIC
        if batch.tenantId == "public_league"
        else ClearanceLevel.TEAM_INTERNAL,
        is_public=(batch.tenantId == "public_league"),
    )
    blob_key = f"replays/{batch.tenantId}/{batch.heatId}.json"
    replay_uri = container.blob.put_bytes(
        blob_key, batch.model_dump_json(indent=2).encode("utf-8"), replay_acl
    )

    for ev in batch.events:
        container.stream_engine.process_event(ev)

    container.repo.upsert_heat_and_ratings(batch, replay_uri, principal)
    return {
        "heat_id": batch.heatId,
        "accepted_events": len(batch.events),
        "replay_uri": replay_uri,
        "stream_snapshot": container.stream_engine.snapshot_metrics(),
    }


@router.get("/heats/{heat_id}/replay")
def get_heat_replay(
    heat_id: str,
    principal: PrincipalDep,
    container: ContainerDep,
) -> dict[str, object]:
    raw_heat = container.repo.heats.get(heat_id)
    if raw_heat is None:
        raise HTTPException(status_code=404, detail="Heat not found")

    rec = container.repo.get_heat(heat_id, principal)
    if rec is None:
        raise HTTPException(status_code=403, detail="Access denied to heat telemetry")

    blob_key = f"replays/{rec.tenant_id}/{rec.heat_id}.json"
    try:
        payload = container.blob.get_bytes(blob_key, principal)
    except BlobAccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Replay artifact missing") from exc

    return dict(json.loads(payload.decode("utf-8")))


@router.get("/stream/metrics")
def get_stream_metrics(
    container: ContainerDep,
) -> dict[str, object]:
    return {
        "snapshot": container.stream_engine.snapshot_metrics(),
        "alerts": container.stream_engine.alerts[-20:],
    }
