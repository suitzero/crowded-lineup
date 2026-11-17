"""Event-time watermark & hopping window aggregator for real-time lineup congestion & alerts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from packages.python_sim.contracts import TelemetryEvent


@dataclass
class WindowAggregate:
    tenant_id: str
    window_start_sec: float
    window_end_sec: float
    catches: int = 0
    drop_ins: int = 0
    collisions: int = 0
    unique_surfers: set[int] = field(default_factory=set)
    alert_triggered: bool = False


class WatermarkStreamEngine:
    """Deterministic event-time hopping-window processor matching Flink HOP(30s, 10s)."""

    def __init__(
        self,
        window_size_sec: float = 30.0,
        slide_sec: float = 10.0,
        allowed_lateness_sec: float = 5.0,
        drop_in_alert_threshold: int = 3,
    ) -> None:
        self.window_size_sec = window_size_sec
        self.slide_sec = slide_sec
        self.allowed_lateness_sec = allowed_lateness_sec
        self.drop_in_alert_threshold = drop_in_alert_threshold

        self.max_event_time_sec = 0.0
        self.watermark_sec = -allowed_lateness_sec
        self.processed_events = 0
        self.dropped_late_events = 0
        self.late_accepted_events = 0
        self.windows: dict[tuple[str, float], WindowAggregate] = {}
        self.alerts: list[dict[str, object]] = []

    def _window_starts_for(self, ts: float) -> list[float]:
        # Hopping window start times covering `ts`: start <= ts < start + window_size
        first_idx = math.floor((ts - self.window_size_sec) / self.slide_sec) + 1
        last_idx = math.floor(ts / self.slide_sec)
        starts: list[float] = []
        for idx in range(max(0, first_idx), last_idx + 1):
            ws = round(idx * self.slide_sec, 4)
            if ws <= ts < ws + self.window_size_sec:
                starts.append(ws)
        return starts

    def process_event(self, event: TelemetryEvent) -> list[WindowAggregate]:
        """Process one `TelemetryEvent` under event-time watermark semantics."""
        ts = float(event.timeSec)
        if ts < self.watermark_sec:
            self.dropped_late_events += 1
            return []

        if ts < self.max_event_time_sec:
            self.late_accepted_events += 1
        elif ts > self.max_event_time_sec:
            self.max_event_time_sec = ts
            self.watermark_sec = max(
                self.watermark_sec,
                self.max_event_time_sec - self.allowed_lateness_sec,
            )

        self.processed_events += 1
        updated: list[WindowAggregate] = []

        for ws in self._window_starts_for(ts):
            key = (event.tenantId, ws)
            agg = self.windows.get(key)
            if agg is None:
                agg = WindowAggregate(
                    tenant_id=event.tenantId,
                    window_start_sec=ws,
                    window_end_sec=round(ws + self.window_size_sec, 4),
                )
                self.windows[key] = agg

            if event.surferId is not None:
                agg.unique_surfers.add(event.surferId)
            if event.eventType == "takeoff_caught":
                agg.catches += 1
            elif event.eventType == "drop_in_violation":
                agg.drop_ins += 1
            elif event.eventType == "surfer_collision":
                agg.collisions += 1

            if agg.drop_ins >= self.drop_in_alert_threshold and not agg.alert_triggered:
                agg.alert_triggered = True
                self.alerts.append(
                    {
                        "tenant_id": agg.tenant_id,
                        "window_start_sec": agg.window_start_sec,
                        "window_end_sec": agg.window_end_sec,
                        "drop_ins": agg.drop_ins,
                        "collisions": agg.collisions,
                        "reason": "drop_in_spike_threshold_exceeded",
                    }
                )
            updated.append(agg)

        return updated

    def snapshot_metrics(self) -> dict[str, object]:
        return {
            "max_event_time_sec": self.max_event_time_sec,
            "watermark_sec": self.watermark_sec,
            "processed_events": self.processed_events,
            "late_accepted_events": self.late_accepted_events,
            "dropped_late_events": self.dropped_late_events,
            "active_windows": len(self.windows),
            "alerts_count": len(self.alerts),
        }
