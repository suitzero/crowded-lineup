# On-Call Runbook: Streaming Pipeline Lag & ACL Denial Spikes

## 1. Symptoms & Alerts
- **Alert A (`DropInSpikeAlert`)**: `drop_in_violation >= 3` inside a 30s hopping window.
- **Alert B (`LateEventDropSpike`)**: `dropped_late_events / processed_events > 2%`.
- **Alert C (`AclDenyBurst`)**: Sudden increase in `audit_logs` rows with `decision = 'DENY'`.

## 2. Triage Checklist
1. Check `/metrics` (`crowded_lineup_http_requests_total`) and `/v1/stream/metrics` (`watermark_sec`, `dropped_late_events`).
2. Verify Redpanda/PubSub consumer lag and check `stream-worker` container logs.
3. If `dropped_late_events` spiked due to an upstream batch retry flushing old timestamps:
   - Do **not** loosen streaming watermark indefinitely; replay the affected date partition through the **Apache Beam Batch pipeline** (`pipelines/batch/replay_analytics_beam.py`) to reconcile Postgres rankings deterministically.
4. If `AclDenyBurst` triggers:
   - Query `audit_logs` as `steward_lead` via `/v1/audit/logs` grouped by `principal_id`, `agent_id`, and `resource_id`.
   - Confirm whether an unallowlisted LLM agent or misconfigured `X-Principal-Id` header is probing cross-tenant documents; revoke offending agent credentials immediately if unauthorized.
