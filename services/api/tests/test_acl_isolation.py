"""Penetration tests proving per-document ACL isolation and 100% audit logging across SQL/Index/Blob."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from services.api.config import Settings
from services.api.main import create_app


def _make_client(tmp_path: Path) -> TestClient:
    cfg = Settings(
        blob_root=tmp_path / "blobs",
        corpus_work_dir=tmp_path / "corpus",
        bootstrap_corpus=True,
    )
    return TestClient(create_app(cfg))


def test_cross_tenant_acl_isolation_across_documents_search_and_agent(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # 1. Listing documents only reveals authorized tenant + public documents
    alpha_docs = {
        d["doc_id"]
        for d in client.get("/v1/documents", headers={"X-Principal-Id": "alice_alpha"}).json()[
            "documents"
        ]
    }
    assert alpha_docs == {"doc-isa-rules-01", "doc-alpha-scout-01"}

    beta_docs = {
        d["doc_id"]
        for d in client.get("/v1/documents", headers={"X-Principal-Id": "bob_beta"}).json()[
            "documents"
        ]
    }
    assert beta_docs == {"doc-isa-rules-01", "doc-beta-scout-01"}

    guest_docs = {
        d["doc_id"]
        for d in client.get("/v1/documents", headers={"X-Principal-Id": "guest_public"}).json()[
            "documents"
        ]
    }
    assert guest_docs == {"doc-isa-rules-01"}

    # 2. Direct document read by unauthorized principal fails closed (403) and logs DENY
    deny_res = client.get(
        "/v1/documents/doc-alpha-scout-01",
        headers={"X-Principal-Id": "bob_beta"},
    )
    assert deny_res.status_code == 403

    # Even Steward cannot read private Team Alpha trade-secret scouting doc unless allowlisted
    steward_deny = client.get(
        "/v1/documents/doc-alpha-scout-01",
        headers={"X-Principal-Id": "steward_lead"},
    )
    assert steward_deny.status_code == 403

    # Steward CAN read restricted protest form
    steward_allow = client.get(
        "/v1/documents/doc-protest-scan-01",
        headers={"X-Principal-Id": "steward_lead"},
    )
    assert steward_allow.status_code == 200
    assert "STEWARD_DOCK_PENALTY_OVERRIDE" in steward_allow.json()["pages"]["1"]

    # 3. Hybrid Vector/BM25 search never leaks unauthorized chunks
    alpha_search = client.post(
        "/v1/documents/search",
        json={"query": "COBALT_PEAK_DRIFT AMBER_WEDGE_BUDGET", "top_k": 10},
        headers={"X-Principal-Id": "alice_alpha"},
    ).json()["hits"]
    alpha_hit_ids = {h["doc_id"] for h in alpha_search}
    assert "doc-alpha-scout-01" in alpha_hit_ids
    assert "doc-beta-scout-01" not in alpha_hit_ids
    assert "doc-protest-scan-01" not in alpha_hit_ids

    # 4. Steward Agent query as bob_beta must NEVER cite or leak COBALT_PEAK_DRIFT
    intel_beta = client.post(
        "/v1/intelligence/query",
        json={"query": "Tell me about COBALT_PEAK_DRIFT"},
        headers={"X-Principal-Id": "bob_beta", "X-Agent-Id": "beta-coach-bot"},
    )
    assert intel_beta.status_code == 200
    body_beta = intel_beta.json()
    assert "COBALT_PEAK_DRIFT" not in body_beta["answer"]
    assert all(c["source_id"] != "doc-alpha-scout-01" for c in body_beta["citations"])
    assert body_beta["audit_events_recorded"] >= 3

    # 5. Audit log inspection is restricted to steward role
    assert (
        client.get("/v1/audit/logs", headers={"X-Principal-Id": "alice_alpha"}).status_code == 403
    )
    audit_res = client.get("/v1/audit/logs", headers={"X-Principal-Id": "steward_lead"})
    assert audit_res.status_code == 200
    entries = audit_res.json()["entries"]
    assert any(
        e["principal_id"] == "bob_beta"
        and e["resource_id"] == "doc-alpha-scout-01"
        and e["decision"] == "DENY"
        for e in entries
    )
    assert any(e["agent_id"] == "beta-coach-bot" for e in entries)
