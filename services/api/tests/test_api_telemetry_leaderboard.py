"""API tests for telemetry ingestion, Elo leaderboard updates, and streaming metrics."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from packages.python_sim.contracts import (
    AgentRosterEntry,
    EpisodeMetrics,
    HeatTelemetryBatch,
    TelemetryEvent,
    WorldConfig,
)
from services.api.config import Settings
from services.api.main import create_app


def _make_client(tmp_path: Path) -> TestClient:
    cfg = Settings(
        blob_root=tmp_path / "blobs",
        corpus_work_dir=tmp_path / "corpus",
        bootstrap_corpus=True,
    )
    return TestClient(create_app(cfg))


def test_telemetry_ingestion_updates_leaderboard_and_replay(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["documents"] == 4

    batch = HeatTelemetryBatch(
        heatId="heat-api-1",
        tenantId="public_league",
        preset="crowded-24",
        config=WorldConfig(numSurfers=2, width=320),
        agentRoster=[
            AgentRosterEntry(
                surferId=0, agentId="ppo-gen12", policyType="neural-onnx", label="PPO"
            ),
            AgentRosterEntry(surferId=1, agentId="bot-charger", policyType="rule-bot", label="B1"),
        ],
        events=[
            TelemetryEvent(
                eventId="ev-1",
                heatId="heat-api-1",
                tenantId="public_league",
                eventType="takeoff_caught",
                stepCount=12,
                timeSec=1.2,
                seed=77,
                surferId=0,
                agentPolicyId="ppo-gen12",
            ),
            TelemetryEvent(
                eventId="ev-2",
                heatId="heat-api-1",
                tenantId="public_league",
                eventType="drop_in_violation",
                stepCount=25,
                timeSec=2.5,
                seed=77,
                surferId=1,
                agentPolicyId="bot-charger",
            ),
        ],
        metrics=EpisodeMetrics(
            seed=77,
            totalSteps=100,
            durationSec=10.0,
            totalWavesSpawned=4,
            surferCatches=[2, 0],
            surferCatchRates=[0.5, 0.0],
            surferScores=[120.0, 15.0],
            surferCollisions=[0, 0],
            surferDropIns=[0, 1],
            surferStaminaUsed=[1.2, 1.4],
            surferRideTime=[4.5, 0.0],
        ),
    )

    post_res = client.post(
        "/v1/telemetry/batches",
        json=batch.model_dump(),
        headers={"X-Principal-Id": "alice_alpha"},
    )
    assert post_res.status_code == 200
    assert post_res.json()["accepted_events"] == 2

    lb_res = client.get("/v1/leaderboard")
    assert lb_res.status_code == 200
    entries = lb_res.json()["entries"]
    assert entries[0]["agent_id"] == "ppo-gen12"
    assert entries[0]["elo_rating"] > entries[1]["elo_rating"]

    replay_res = client.get(
        "/v1/heats/heat-api-1/replay",
        headers={"X-Principal-Id": "guest_public"},
    )
    assert replay_res.status_code == 200
    assert replay_res.json()["heatId"] == "heat-api-1"
