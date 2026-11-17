# ADR-001: Batch vs. Streaming Data Pipeline Trade-offs

- **Status**: Accepted
- **Date**: 2026-11-17

## Context
Crowded Lineup generates high-frequency multi-agent simulation telemetry (16–40 surfers at `dt=0.1s`) alongside unstructured league documents (PDF rulebooks, scanned incident sheets, team scouting memos).

## Decision
Adopt a **dual Lambda-style architecture** separating latency-sensitive operational signals from deterministic reprocessing:

1. **Streaming Path (`pipelines/streaming/watermark_engine.py` + `flink_sql_job.sql`)**:
   - Consumes `TelemetryEvent` messages via Redpanda/PubSub using a **5-second event-time watermark** and **30s/10s hopping windows**.
   - Updates real-time lineup congestion alerts and rolling drop-in violation spikes.
   - **Trade-off**: Accepts bounded late-event approximation (events arriving behind the watermark are dropped from real-time alerts and counted in `dropped_late_events`) to guarantee sub-second alerting latency.

2. **Batch Path (`pipelines/batch/replay_analytics_beam.py`, `doc_ingest_beam.py`)**:
   - Reads immutable raw replay batches and document load-files from Blob Storage (`GCS` / `MinIO`) via **Apache Beam** (`DirectRunner` locally, `DataflowRunner` in GCP).
   - Computes exact cross-heat agent evaluation matrices, curated RL trajectories, and OCR/PDF chunk embeddings.
   - **Trade-off**: Higher end-to-end latency (minutes/hours), in exchange for **100% deterministic reprocessability** when reward formulas, Elo parameters, or OCR models change.
