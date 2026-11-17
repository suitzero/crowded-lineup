"""Apache Beam batch tests covering document OCR/PDF ingestion and replay analytics."""

from __future__ import annotations

from pathlib import Path

from data.sample_corpus.seed_corpus import generate_sample_corpus
from packages.python_sim.contracts import (
    AgentRosterEntry,
    EpisodeMetrics,
    HeatTelemetryBatch,
    WorldConfig,
)
from pipelines.batch.doc_ingest_beam import run_doc_ingest_beam
from pipelines.batch.replay_analytics_beam import run_replay_analytics_beam
from services.security.acl import DEV_PRINCIPALS
from services.security.audit import AuditLogger
from services.storage.blob import LocalBlobStore
from services.storage.hybrid_index import HybridAclSearchIndex
from services.storage.postgres import InMemoryArenaRepository


def test_beam_doc_ingest_and_acl_index(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    loadfile = generate_sample_corpus(corpus_dir)

    audit = AuditLogger()
    repo = InMemoryArenaRepository(audit)
    index = HybridAclSearchIndex(audit)
    blob = LocalBlobStore(tmp_path / "blobs", audit)

    chunks_indexed = run_doc_ingest_beam(loadfile, repo, index, blob)
    assert chunks_indexed >= 7  # 2+2+2 PDF pages + 1 scanned PNG

    # Verify public rulebook is accessible to guest_public
    guest_hits = index.search(
        "inside priority drop-in penalty", DEV_PRINCIPALS["guest_public"], top_k=3
    )
    assert any(h.doc_id == "doc-isa-rules-01" for h in guest_hits)

    # Verify Team Alpha secret codename is searchable ONLY by alice_alpha, never bob_beta
    alpha_hits = index.search("COBALT_PEAK_DRIFT", DEV_PRINCIPALS["alice_alpha"], top_k=3)
    assert any(h.doc_id == "doc-alpha-scout-01" for h in alpha_hits)

    beta_leak_hits = index.search("COBALT_PEAK_DRIFT", DEV_PRINCIPALS["bob_beta"], top_k=3)
    assert all(h.doc_id != "doc-alpha-scout-01" for h in beta_leak_hits)

    explicit_denied = index.search(
        "COBALT_PEAK_DRIFT",
        DEV_PRINCIPALS["bob_beta"],
        top_k=3,
        doc_id_filter="doc-alpha-scout-01",
    )
    assert explicit_denied == []
    assert any(
        e.decision == "DENY" and e.resource_id == "doc-alpha-scout-01" for e in audit.list_entries()
    )


def test_beam_replay_analytics_aggregation(tmp_path: Path) -> None:
    batches = []
    for idx in range(3):
        b = HeatTelemetryBatch(
            heatId=f"heat-{idx}",
            tenantId="public_league",
            preset="crowded-16",
            config=WorldConfig(numSurfers=2),
            agentRoster=[
                AgentRosterEntry(
                    surferId=0, agentId="ppo-v2", policyType="neural-onnx", label="PPO"
                ),
                AgentRosterEntry(surferId=1, agentId="bot-1", policyType="rule-bot", label="B1"),
            ],
            events=[],
            metrics=EpisodeMetrics(
                seed=100 + idx,
                totalSteps=600,
                durationSec=60.0,
                totalWavesSpawned=10,
                surferCatches=[4, 2],
                surferCatchRates=[0.4, 0.2],
                surferScores=[150.0 + idx * 10, 80.0],
                surferCollisions=[0, 1],
                surferDropIns=[0, 1],
                surferStaminaUsed=[2.5, 3.0],
                surferRideTime=[12.0, 5.0],
            ),
        )
        batches.append(b.model_dump_json())

    out_path = tmp_path / "analytics.json"
    summary = run_replay_analytics_beam(batches, out_path)
    assert out_path.exists()
    assert summary["ppo-v2"]["heats"] == 3
    assert summary["ppo-v2"]["avg_score"] == 160.0
    assert summary["bot-1"]["drop_in_rate"] == 1.0
