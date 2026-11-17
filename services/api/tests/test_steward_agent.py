"""End-to-end tests for Steward Agent typed tools, page citations, and heat correlation."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from packages.python_sim.contracts import (
    AgentRosterEntry,
    EpisodeMetrics,
    HeatTelemetryBatch,
    WorldConfig,
)
from services.api.config import Settings
from services.api.main import create_app


def test_steward_agent_cites_corpus_and_heat_replay(tmp_path: Path) -> None:
    cfg = Settings(
        blob_root=tmp_path / "blobs",
        corpus_work_dir=tmp_path / "corpus",
        bootstrap_corpus=True,
    )
    client = TestClient(create_app(cfg))

    batch = HeatTelemetryBatch(
        heatId="heat-steward-99",
        tenantId="public_league",
        preset="crowded-24",
        config=WorldConfig(numSurfers=24, width=320),
        agentRoster=[
            AgentRosterEntry(
                surferId=0, agentId="ppo-alpha", policyType="neural-onnx", label="PPO"
            ),
        ],
        events=[],
        metrics=EpisodeMetrics(
            seed=99,
            totalSteps=600,
            durationSec=60.0,
            totalWavesSpawned=12,
            surferCatches=[5],
            surferCatchRates=[0.4167],
            surferScores=[210.0],
            surferCollisions=[1],
            surferDropIns=[2],
            surferStaminaUsed=[3.1],
            surferRideTime=[18.0],
        ),
    )
    client.post(
        "/v1/telemetry/batches",
        json=batch.model_dump(),
        headers={"X-Principal-Id": "alice_alpha"},
    )

    res = client.post(
        "/v1/intelligence/query",
        json={
            "query": "Explain drop-in priority penalty and COBALT_PEAK_DRIFT strategy",
            "heat_id": "heat-steward-99",
        },
        headers={"X-Principal-Id": "alice_alpha"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "COBALT_PEAK_DRIFT" in data["answer"]
    assert "heat-steward-99" in data["answer"]
    source_ids = {c["source_id"] for c in data["citations"]}
    assert "doc-alpha-scout-01" in source_ids
    assert "doc-isa-rules-01" in source_ids
    assert "heat-steward-99" in source_ids
    assert set(data["tools_invoked"]) == {
        "search_private_corpus",
        "query_leaderboard",
        "inspect_heat_replay",
    }
