"""Streaming worker loop consuming telemetry events and publishing windowed congestion metrics."""

from __future__ import annotations

import json
import os
import time
import urllib.request

from packages.python_sim.contracts import TelemetryEvent
from pipelines.streaming.watermark_engine import WatermarkStreamEngine


def run_jsonl_replay_stream(events_jsonl: str) -> dict[str, object]:
    """Process a JSONL stream of telemetry events through `WatermarkStreamEngine`."""
    engine = WatermarkStreamEngine()
    for line in events_jsonl.splitlines():
        if not line.strip():
            continue
        ev = TelemetryEvent.model_validate_json(line)
        engine.process_event(ev)
    return engine.snapshot_metrics()


def main() -> None:
    api_url = os.getenv("API_BASE_URL", "http://api:8000")
    engine = WatermarkStreamEngine()
    print(
        f"[stream-worker] Started watermark engine (API={api_url}), polling health...", flush=True
    )
    while True:
        try:
            with urllib.request.urlopen(f"{api_url}/healthz", timeout=5) as resp:
                _ = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            print(f"[stream-worker] Waiting for API ({exc})", flush=True)
        print(f"[stream-worker] Snapshot: {engine.snapshot_metrics()}", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    main()
