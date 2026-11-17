-- Crowded Lineup Arena & Intelligence Platform Schema
-- Target: PostgreSQL 16 + pgvector

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS principals (
    principal_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    roles TEXT[] NOT NULL DEFAULT '{}',
    clearance_level INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    policy_type TEXT NOT NULL,
    policy_version INTEGER NOT NULL DEFAULT 2,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS heats (
    heat_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    seed BIGINT NOT NULL,
    preset TEXT NOT NULL,
    num_surfers INTEGER NOT NULL,
    total_waves INTEGER NOT NULL DEFAULT 0,
    total_drop_ins INTEGER NOT NULL DEFAULT 0,
    total_collisions INTEGER NOT NULL DEFAULT 0,
    replay_blob_uri TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS leaderboard_ratings (
    agent_id TEXT PRIMARY KEY,
    policy_type TEXT NOT NULL,
    elo_rating DOUBLE PRECISION NOT NULL DEFAULT 1500.0,
    heats_played INTEGER NOT NULL DEFAULT 0,
    catches INTEGER NOT NULL DEFAULT 0,
    drop_ins INTEGER NOT NULL DEFAULT 0,
    collisions INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
    title TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    classification INTEGER NOT NULL DEFAULT 0,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    allowed_principals TEXT[] NOT NULL DEFAULT '{}',
    custodian TEXT,
    doc_type TEXT NOT NULL DEFAULT 'pdf',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    classification INTEGER NOT NULL DEFAULT 0,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    allowed_principals TEXT[] NOT NULL DEFAULT '{}',
    page_number INTEGER NOT NULL DEFAULT 1,
    content TEXT NOT NULL,
    tags TEXT[] NOT NULL DEFAULT '{}',
    embedding vector(64) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_tenant ON document_chunks (tenant_id, classification);

CREATE TABLE IF NOT EXISTS audit_logs (
    audit_id TEXT PRIMARY KEY,
    timestamp_iso TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trace_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    agent_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('ALLOW', 'DENY')),
    reason TEXT NOT NULL,
    query_hash TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);
