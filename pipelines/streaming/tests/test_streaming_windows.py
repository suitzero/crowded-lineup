"""Tests for event-time watermarks, late-event handling, and hopping-window alerts."""

from __future__ import annotations

from packages.python_sim.contracts import TelemetryEvent
from pipelines.streaming.watermark_engine import WatermarkStreamEngine


def _ev(idx: int, event_type: str, time_sec: float, surfer_id: int = 1) -> TelemetryEvent:
    return TelemetryEvent(
        eventId=f"ev-{idx}",
        heatId="heat-stream-1",
        tenantId="team_alpha",
        eventType=event_type,  # type: ignore[arg-type]
        stepCount=int(time_sec * 10),
        timeSec=time_sec,
        seed=42,
        surferId=surfer_id,
    )


def test_watermark_hopping_windows_and_late_events() -> None:
    engine = WatermarkStreamEngine(
        window_size_sec=30.0,
        slide_sec=10.0,
        allowed_lateness_sec=5.0,
        drop_in_alert_threshold=2,
    )

    # Normal events advancing watermark to 25.0 - 5.0 = 20.0
    engine.process_event(_ev(1, "takeoff_caught", 12.0, surfer_id=0))
    engine.process_event(_ev(2, "drop_in_violation", 21.0, surfer_id=2))
    engine.process_event(_ev(3, "takeoff_caught", 25.0, surfer_id=3))

    assert engine.watermark_sec == 20.0
    assert len(engine.alerts) == 0

    # Late event STILL within allowed lateness (>= 20.0): accepted and triggers drop-in alert!
    engine.process_event(_ev(4, "drop_in_violation", 22.0, surfer_id=4))
    assert engine.late_accepted_events == 1
    assert engine.dropped_late_events == 0
    assert len(engine.alerts) >= 1

    # Very late event behind watermark (< 20.0): cleanly dropped
    dropped = engine.process_event(_ev(5, "surfer_collision", 14.0, surfer_id=5))
    assert dropped == []
    assert engine.dropped_late_events == 1
