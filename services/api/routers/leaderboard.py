"""Competitive agent leaderboard endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from services.api.deps import ContainerDep

router = APIRouter(prefix="/v1", tags=["leaderboard"])


@router.get("/leaderboard")
def get_leaderboard(
    container: ContainerDep,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, object]:
    entries = container.repo.get_leaderboard(limit=limit)
    return {"entries": [e.model_dump() for e in entries]}
