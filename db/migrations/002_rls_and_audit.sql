-- Fine-Grained Row-Level Security (RLS) & Append-Only Audit Log Enforcement

ALTER TABLE heats ENABLE ROW LEVEL SECURITY;
ALTER TABLE heats FORCE ROW LEVEL SECURITY;

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;

-- Session-scoped access predicate:
-- Caller sets `SET LOCAL app.principal_id`, `SET LOCAL app.tenant_id`, `SET LOCAL app.clearance_level`
DROP POLICY IF EXISTS heats_tenant_isolation ON heats;
CREATE POLICY heats_tenant_isolation ON heats
    USING (
        tenant_id = 'public_league'
        OR tenant_id = NULLIF(current_setting('app.tenant_id', TRUE), '')
    );

DROP POLICY IF EXISTS documents_acl_isolation ON documents;
CREATE POLICY documents_acl_isolation ON documents
    USING (
        (is_public = TRUE AND classification = 0)
        OR (
            NULLIF(current_setting('app.principal_id', TRUE), '') = ANY(allowed_principals)
            AND COALESCE(NULLIF(current_setting('app.clearance_level', TRUE), '')::INTEGER, 0) >= classification
        )
        OR (
            tenant_id = NULLIF(current_setting('app.tenant_id', TRUE), '')
            AND COALESCE(NULLIF(current_setting('app.clearance_level', TRUE), '')::INTEGER, 0) >= classification
            AND (cardinality(allowed_principals) = 0 OR NULLIF(current_setting('app.principal_id', TRUE), '') = ANY(allowed_principals))
        )
    );

DROP POLICY IF EXISTS chunks_acl_isolation ON document_chunks;
CREATE POLICY chunks_acl_isolation ON document_chunks
    USING (
        (is_public = TRUE AND classification = 0)
        OR (
            NULLIF(current_setting('app.principal_id', TRUE), '') = ANY(allowed_principals)
            AND COALESCE(NULLIF(current_setting('app.clearance_level', TRUE), '')::INTEGER, 0) >= classification
        )
        OR (
            tenant_id = NULLIF(current_setting('app.tenant_id', TRUE), '')
            AND COALESCE(NULLIF(current_setting('app.clearance_level', TRUE), '')::INTEGER, 0) >= classification
            AND (cardinality(allowed_principals) = 0 OR NULLIF(current_setting('app.principal_id', TRUE), '') = ANY(allowed_principals))
        )
    );

-- Enforce append-only immutability on audit_logs
CREATE OR REPLACE FUNCTION prevent_audit_log_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_logs is strictly append-only (attempted %)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_logs_immutable ON audit_logs;
CREATE TRIGGER trg_audit_logs_immutable
    BEFORE UPDATE OR DELETE ON audit_logs
    FOR EACH ROW
    EXECUTE FUNCTION prevent_audit_log_mutation();
