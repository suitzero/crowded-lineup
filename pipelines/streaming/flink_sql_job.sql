-- Production Apache Flink SQL topology equivalent to `watermark_engine.py`
-- Consumes Redpanda/Kafka `crowded_lineup.telemetry.v2` topic with 5s event-time watermark.

CREATE TABLE telemetry_events (
    event_id STRING,
    heat_id STRING,
    tenant_id STRING,
    event_type STRING,
    step_count BIGINT,
    time_sec DOUBLE,
    surfer_id INT,
    agent_policy_id STRING,
    event_ts AS TO_TIMESTAMP_LTZ(CAST(time_sec * 1000 AS BIGINT), 3),
    WATERMARK FOR event_ts AS event_ts - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'crowded_lineup.telemetry.v2',
    'properties.bootstrap.servers' = 'redpanda:9092',
    'format' = 'json'
);

CREATE TABLE congestion_alerts (
    tenant_id STRING,
    window_start TIMESTAMP(3),
    window_end TIMESTAMP(3),
    catches BIGINT,
    drop_ins BIGINT,
    collisions BIGINT
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:postgresql://postgres:5432/crowded_lineup',
    'table-name' = 'stream_congestion_windows'
);

INSERT INTO congestion_alerts
SELECT
    tenant_id,
    window_start,
    window_end,
    COUNT(*) FILTER (WHERE event_type = 'takeoff_caught') AS catches,
    COUNT(*) FILTER (WHERE event_type = 'drop_in_violation') AS drop_ins,
    COUNT(*) FILTER (WHERE event_type = 'surfer_collision') AS collisions
FROM TABLE(
    HOP(TABLE telemetry_events, DESCRIPTOR(event_ts), INTERVAL '10' SECOND, INTERVAL '30' SECOND)
)
GROUP BY tenant_id, window_start, window_end
HAVING COUNT(*) FILTER (WHERE event_type = 'drop_in_violation') >= 3;
