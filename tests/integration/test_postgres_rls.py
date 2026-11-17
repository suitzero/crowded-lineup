"""Real PostgreSQL 16 + pgvector Row-Level Security & Append-Only Audit Log integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest

from services.security.acl import DEV_PRINCIPALS
from services.security.audit import AuditEntry
from services.storage.postgres import PostgresRlsRepository

DB_URL = os.getenv("CROWDED_TEST_DATABASE_URL", "")
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


@pytest.mark.skipif(not DB_URL, reason="CROWDED_TEST_DATABASE_URL not configured")
def test_postgres_rls_and_immutable_audit_log() -> None:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
                cur.execute(sql_file.read_text(encoding="utf-8"))

            cur.execute(
                "DO $$ BEGIN CREATE ROLE app_user NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;"
            )
            cur.execute("GRANT USAGE ON SCHEMA public TO app_user;")
            cur.execute("GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO app_user;")

            cur.execute(
                "INSERT INTO tenants (tenant_id, display_name) VALUES ('public_league','Public'), ('team_alpha','Alpha'), ('team_beta','Beta') ON CONFLICT DO NOTHING;"
            )
            cur.execute(
                """
                INSERT INTO documents (doc_id, tenant_id, title, source_uri, classification, is_public, allowed_principals)
                VALUES
                  ('doc-pub', 'public_league', 'Public Rulebook', 'gs://b/pub.pdf', 0, TRUE, '{}'),
                  ('doc-alpha', 'team_alpha', 'Alpha Secret', 'gs://b/alpha.pdf', 2, FALSE, ARRAY['alice_alpha']),
                  ('doc-beta', 'team_beta', 'Beta Secret', 'gs://b/beta.pdf', 2, FALSE, ARRAY['bob_beta'])
                ON CONFLICT (doc_id) DO NOTHING;
                """
            )
        conn.commit()

        # Verify RLS enforcement as non-superuser `app_user`
        with conn.cursor() as cur:
            cur.execute("SET LOCAL ROLE app_user;")
            cur.execute("SELECT set_config('app.principal_id', 'alice_alpha', true);")
            cur.execute("SELECT set_config('app.tenant_id', 'team_alpha', true);")
            cur.execute("SELECT set_config('app.clearance_level', '2', true);")
            cur.execute("SELECT doc_id FROM documents ORDER BY doc_id;")
            alpha_ids = [r[0] for r in cur.fetchall()]
            assert alpha_ids == ["doc-alpha", "doc-pub"]
        conn.rollback()

        with conn.cursor() as cur:
            cur.execute("SET LOCAL ROLE app_user;")
            cur.execute("SELECT set_config('app.principal_id', 'bob_beta', true);")
            cur.execute("SELECT set_config('app.tenant_id', 'team_beta', true);")
            cur.execute("SELECT set_config('app.clearance_level', '2', true);")
            cur.execute("SELECT doc_id FROM documents ORDER BY doc_id;")
            beta_ids = [r[0] for r in cur.fetchall()]
            assert beta_ids == ["doc-beta", "doc-pub"]
        conn.rollback()

    repo = PostgresRlsRepository(DB_URL)
    entry = AuditEntry(
        trace_id="rls-test-trace",
        principal_id=DEV_PRINCIPALS["alice_alpha"].principal_id,
        tenant_id=DEV_PRINCIPALS["alice_alpha"].tenant_id,
        action="sql_query",
        resource_type="documents",
        resource_id="doc-alpha",
        decision="ALLOW",
        reason="rls_integration_test",
    )
    repo.insert_audit_entry(entry)

    with psycopg.connect(DB_URL) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.errors.RaiseException):
            cur.execute("DELETE FROM audit_logs WHERE audit_id = %s", (entry.audit_id,))
        conn.rollback()
