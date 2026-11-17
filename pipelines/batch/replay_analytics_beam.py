"""Apache Beam batch replay analytics pipeline aggregating agent performance across heats."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from packages.python_sim.contracts import HeatTelemetryBatch


def extract_agent_heat_records(batch_json: str) -> list[tuple[str, dict[str, Any]]]:
    """Emit `(agent_id, heat_stat)` pairs from a `HeatTelemetryBatch` JSON string."""
    batch = HeatTelemetryBatch.model_validate_json(batch_json)
    if not batch.metrics:
        return []
    m = batch.metrics
    out: list[tuple[str, dict[str, Any]]] = []
    for r in batch.agentRoster:
        sid = r.surferId
        if sid >= len(m.surferScores):
            continue
        out.append(
            (
                r.agentId,
                {
                    "policy_type": r.policyType,
                    "heats": 1,
                    "score": float(m.surferScores[sid]),
                    "catches": int(m.surferCatches[sid]),
                    "drop_ins": int(m.surferDropIns[sid]),
                    "collisions": int(m.surferCollisions[sid]),
                },
            )
        )
    return out


class AggregateAgentStatsFn(beam.CombineFn):
    def create_accumulator(self) -> dict[str, Any]:
        return {
            "policy_type": "unknown",
            "heats": 0,
            "total_score": 0.0,
            "total_catches": 0,
            "total_drop_ins": 0,
            "total_collisions": 0,
        }

    def add_input(self, acc: dict[str, Any], element: dict[str, Any]) -> dict[str, Any]:
        acc["policy_type"] = element.get("policy_type", acc["policy_type"])
        acc["heats"] += int(element["heats"])
        acc["total_score"] += float(element["score"])
        acc["total_catches"] += int(element["catches"])
        acc["total_drop_ins"] += int(element["drop_ins"])
        acc["total_collisions"] += int(element["collisions"])
        return acc

    def merge_accumulators(self, accumulators: list[dict[str, Any]]) -> dict[str, Any]:
        merged = self.create_accumulator()
        for a in accumulators:
            if a["policy_type"] != "unknown":
                merged["policy_type"] = a["policy_type"]
            merged["heats"] += a["heats"]
            merged["total_score"] += a["total_score"]
            merged["total_catches"] += a["total_catches"]
            merged["total_drop_ins"] += a["total_drop_ins"]
            merged["total_collisions"] += a["total_collisions"]
        return merged

    def extract_output(self, acc: dict[str, Any]) -> dict[str, Any]:
        heats = max(1, acc["heats"])
        return {
            "policy_type": acc["policy_type"],
            "heats": acc["heats"],
            "avg_score": round(acc["total_score"] / heats, 2),
            "avg_catches": round(acc["total_catches"] / heats, 3),
            "drop_in_rate": round(acc["total_drop_ins"] / heats, 3),
            "collision_rate": round(acc["total_collisions"] / heats, 3),
        }


def run_replay_analytics_beam(
    batch_payloads: list[str], out_json_path: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Run deterministic Beam DirectRunner aggregation over replay telemetry batches."""
    with tempfile.TemporaryDirectory(prefix="beam_replay_") as tmp_dir:
        out_kv_path = Path(tmp_dir) / "agg_agents.jsonl"
        with beam.Pipeline(options=PipelineOptions(["--runner=DirectRunner"])) as pipeline:
            aggregated = (
                pipeline
                | "ReadBatches" >> beam.Create(batch_payloads)
                | "ExplodeAgents" >> beam.FlatMap(extract_agent_heat_records)
                | "AggregateByAgent" >> beam.CombinePerKey(AggregateAgentStatsFn())
            )
            _ = (
                aggregated
                | "FormatJsonLine"
                >> beam.Map(lambda kv: json.dumps({"agent_id": kv[0], "stats": kv[1]}))
                | "WriteAggregates"
                >> beam.io.WriteToText(
                    str(out_kv_path),
                    shard_name_template="",
                    num_shards=1,
                )
            )

        summary: dict[str, dict[str, Any]] = {}
        for line in out_kv_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            summary[str(row["agent_id"])] = dict(row["stats"])

    if out_json_path is not None:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
