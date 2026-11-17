# ADR-003: Fine-Grained Per-Document ACL Enforcement & Audit Logging

- **Status**: Accepted
- **Date**: 2026-11-17

## Problem
When an LLM Steward Agent uses RAG over multi-tenant private scouting memos (`team_alpha`, `team_beta`), **post-filtering** top-k search results leaks confidential data in two ways:
1. Unauthorized chunks occupy top-k slots and influence BM25 IDF statistics.
2. Summarization or cross-document tools can inadvertently expose denied document existence or snippets.

## Decision
Enforce **fail-closed Pre-Filter Access Control** uniformly across every storage boundary:
1. **PostgreSQL RLS**: Every transaction binds `app.principal_id`, `app.tenant_id`, and `app.clearance_level` via `SET LOCAL`, forcing Postgres itself to drop unauthorized `documents`, `document_chunks`, and `heats` rows.
2. **Search & Vector Pre-Filter**: `HybridAclSearchIndex.search()` evaluates `can_principal_access(principal, chunk.acl)` **before** building term frequencies or scoring vectors.
3. **Blob ACL Enforcement**: `LocalBlobStore` / `S3MinioBlobStore` validates ACL metadata before returning bytes.
4. **Immutable Audit Trail**: Every allowed and denied read, search, blob fetch, and agent tool invocation records an append-only row (`audit_logs` protected by `BEFORE UPDATE OR DELETE` trigger).
