"""Versioned Pydantic v2 contracts mirroring `src/contracts/types.ts` (v2)."""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field

CONTRACT_VERSION = 2
OBSERVATION_DIM = 16
NUM_DISCRETE_ACTIONS = 6


class ActionCode(IntEnum):
    WAIT = 0
    LEFT = 1
    RIGHT = 2
    OUT = 3
    IN = 4
    CATCH = 5


class SurferStatus(IntEnum):
    FREE = 0
    RIDING = 1
    WIPED_OUT = 2


class WorldConfig(BaseModel):
    version: int = CONTRACT_VERSION
    width: float = 320.0
    height: float = 240.0
    dt: float = 0.1
    maxSteps: int = 600
    numSurfers: int = 16
    maxWaves: int = 10
    maxWhitewaters: int = 40
    paddleSpeedMultiplier: float = 1.0
    simplifiedWaves: bool = False
    waveSpawnInterval: float = 5.2


class EpisodeMetrics(BaseModel):
    version: int = CONTRACT_VERSION
    seed: int
    totalSteps: int
    durationSec: float
    totalWavesSpawned: int
    surferCatches: list[int]
    surferCatchRates: list[float]
    surferScores: list[float]
    surferCollisions: list[int]
    surferDropIns: list[int]
    surferStaminaUsed: list[float]
    surferRideTime: list[float]


class ReferenceScenarioStep(BaseModel):
    stepIndex: int
    actions: list[int]
    expectedObservations: list[list[float]]
    expectedRewards: list[float]
    expectedSurferX: list[float]
    expectedSurferY: list[float]
    expectedSurferStamina: list[float]
    expectedSurferState: list[int]


class ReferenceScenario(BaseModel):
    version: int = CONTRACT_VERSION
    name: str
    description: str
    seed: int
    config: WorldConfig
    steps: list[ReferenceScenarioStep]
    expectedMetrics: EpisodeMetrics


class PolicyMetadata(BaseModel):
    version: int = CONTRACT_VERSION
    name: str
    policyType: Literal["human", "rule-bot", "q-table", "neural-onnx"]
    observationVersion: int = CONTRACT_VERSION
    observationDim: int = OBSERVATION_DIM
    actionVersion: int = CONTRACT_VERSION
    numActions: int = NUM_DISCRETE_ACTIONS
    createdAtIso: str
    trainingEpisodes: int = 0
    paddleSpeedMultiplier: float = 1.0
    numCompetitors: int = 0
    avgCatchRate: float = 0.0
    avgReward: float = 0.0


class SerializedPolicy(BaseModel):
    version: int = CONTRACT_VERSION
    metadata: PolicyMetadata
    qTable: dict[str, list[float]] | None = None
    onnxModelBase64: str | None = None


class TelemetryEvent(BaseModel):
    version: int = CONTRACT_VERSION
    eventId: str
    heatId: str
    tenantId: str = "public_league"
    eventType: Literal[
        "heat_started",
        "wave_spawned",
        "takeoff_caught",
        "drop_in_violation",
        "surfer_collision",
        "heat_completed",
    ]
    stepCount: int
    timeSec: float
    seed: int
    surferId: int | None = None
    otherSurferId: int | None = None
    agentPolicyId: str | None = None
    waveId: int | None = None
    x: float | None = None
    y: float | None = None
    rewardDelta: float | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class AgentRosterEntry(BaseModel):
    surferId: int
    agentId: str
    policyType: Literal["human", "rule-bot", "q-table", "neural-onnx"]
    label: str


class HeatTelemetryBatch(BaseModel):
    version: int = CONTRACT_VERSION
    heatId: str
    tenantId: str = "public_league"
    preset: str = "custom"
    config: WorldConfig
    agentRoster: list[AgentRosterEntry]
    events: list[TelemetryEvent]
    metrics: EpisodeMetrics | None = None
