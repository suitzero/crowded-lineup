# ADR-002: Polyglot Storage Architecture & Trade-offs

- **Status**: Accepted
- **Date**: 2026-11-17

## Decision & Storage Matrix

| Store | Technology | Role in Crowded Lineup | Why Not Use One Store for Everything? |
| :--- | :--- | :--- | :--- |
| **Relational + RLS** | PostgreSQL 16 | Tenants, principals, heat summaries, Elo ratings, document ACL metadata, append-only `audit_logs` | Enforces ACID transactions and **database-engine Row-Level Security (RLS)**; unsuitable for multi-MB raw replay blobs or large binary PDFs. |
| **Blob / Object Store** | GCS / MinIO (S3) | Raw heat JSON/Parquet replays, `.onnx` policy weights, original PDFs and scanned PNGs | Lowest $/GB, immutable versioning, lifecycle tiering (`NEARLINE`); lacks sub-second relational joins or full-text queries. |
| **Hybrid Search + Vector** | `pgvector` + `tsvector` / `HybridAclSearchIndex` | Semantic + BM25 lexical retrieval over chunked rulebooks, OCR incident scans, and scouting memos | Vector similarity captures synonyms ("peak drift" vs "lateral shoulder shift") while lexical BM25 guarantees exact matches on codes (`COBALT_PEAK_DRIFT`, `Rule 2.4`). |
| **Low-Latency Cache / NoSQL** | Redis / Firestore | Real-time session state, sliding-window counters, API rate-limiting | Sub-millisecond key-value updates without bloating Postgres WAL write amplification. |
