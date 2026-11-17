import { CONTRACT_VERSION } from '../contracts/types.ts';
import type {
  EpisodeMetrics,
  HeatTelemetryBatch,
  LineupPresetId,
  PolicyType,
  StepResult,
  TelemetryEvent,
  WorldState,
} from '../contracts/types.ts';

export interface AgentRosterEntry {
  surferId: number;
  agentId: string;
  policyType: PolicyType;
  label: string;
}

/**
 * Pure deterministic telemetry recorder that observes pre/post simulation step deltas
 * without mutating `WorldState` or introducing non-deterministic timestamps into physics.
 */
export class TelemetryRecorder {
  public readonly heatId: string;
  public readonly tenantId: string;
  public readonly preset: LineupPresetId | 'custom';
  private readonly roster: AgentRosterEntry[];
  private readonly events: TelemetryEvent[] = [];
  private seq = 0;

  private prevWavesSpawned = 0;
  private prevCatches: number[];
  private prevDropIns: number[];
  private prevCollisions: number[];
  private completed = false;

  constructor(
    world: WorldState,
    options: {
      heatId: string;
      tenantId?: string;
      preset?: LineupPresetId | 'custom';
      roster?: AgentRosterEntry[];
    }
  ) {
    this.heatId = options.heatId;
    this.tenantId = options.tenantId ?? 'public-league';
    this.preset = options.preset ?? 'custom';
    this.roster =
      options.roster ??
      Array.from({ length: world.config.numSurfers }, (_, idx) => ({
        surferId: idx,
        agentId: idx === 0 ? 'human-0' : `bot-${idx}`,
        policyType: idx === 0 ? 'human' : 'rule-bot',
        label: idx === 0 ? 'YOU' : `B${idx}`,
      }));

    this.prevWavesSpawned = world.totalWavesSpawned;
    this.prevCatches = [...world.surferCatches];
    this.prevDropIns = [...world.surferDropIns];
    this.prevCollisions = [...world.surferCollisions];

    this.pushEvent({
      eventType: 'heat_started',
      stepCount: world.stepCount,
      timeSec: world.time,
      seed: world.seed,
      metadata: {
        numSurfers: world.config.numSurfers,
        width: world.config.width,
        height: world.config.height,
        preset: this.preset,
      },
    });
  }

  private nextEventId(): string {
    this.seq += 1;
    return `${this.heatId}-ev-${this.seq}`;
  }

  private pushEvent(
    ev: Omit<TelemetryEvent, 'version' | 'eventId' | 'heatId' | 'tenantId'>
  ): void {
    this.events.push({
      version: CONTRACT_VERSION,
      eventId: this.nextEventId(),
      heatId: this.heatId,
      tenantId: this.tenantId,
      ...ev,
    });
  }

  public recordStep(
    world: WorldState,
    stepRes: StepResult,
    metrics?: EpisodeMetrics
  ): TelemetryEvent[] {
    const startLen = this.events.length;

    if (world.totalWavesSpawned > this.prevWavesSpawned) {
      this.pushEvent({
        eventType: 'wave_spawned',
        stepCount: world.stepCount,
        timeSec: world.time,
        seed: world.seed,
        waveId: world.totalWavesSpawned,
      });
      this.prevWavesSpawned = world.totalWavesSpawned;
    }

    for (let i = 0; i < world.config.numSurfers; i++) {
      const policyId = this.roster[i]?.agentId ?? `surfer-${i}`;
      if (world.surferCatches[i] > (this.prevCatches[i] ?? 0)) {
        this.pushEvent({
          eventType: 'takeoff_caught',
          stepCount: world.stepCount,
          timeSec: world.time,
          seed: world.seed,
          surferId: i,
          agentPolicyId: policyId,
          waveId: world.surferRidingWaveId[i],
          x: +world.surferX[i].toFixed(2),
          y: +world.surferY[i].toFixed(2),
          rewardDelta: stepRes.rewards[i],
        });
        this.prevCatches[i] = world.surferCatches[i];
      }

      if (world.surferDropIns[i] > (this.prevDropIns[i] ?? 0)) {
        this.pushEvent({
          eventType: 'drop_in_violation',
          stepCount: world.stepCount,
          timeSec: world.time,
          seed: world.seed,
          surferId: i,
          agentPolicyId: policyId,
          x: +world.surferX[i].toFixed(2),
          y: +world.surferY[i].toFixed(2),
          rewardDelta: stepRes.breakdowns[i]?.dropInPenalty ?? -8.0,
        });
        this.prevDropIns[i] = world.surferDropIns[i];
      }

      if (world.surferCollisions[i] > (this.prevCollisions[i] ?? 0)) {
        this.pushEvent({
          eventType: 'surfer_collision',
          stepCount: world.stepCount,
          timeSec: world.time,
          seed: world.seed,
          surferId: i,
          agentPolicyId: policyId,
          x: +world.surferX[i].toFixed(2),
          y: +world.surferY[i].toFixed(2),
          rewardDelta: stepRes.breakdowns[i]?.collisionPenalty ?? -6.0,
        });
        this.prevCollisions[i] = world.surferCollisions[i];
      }
    }

    if (stepRes.done && !this.completed) {
      this.completed = true;
      this.pushEvent({
        eventType: 'heat_completed',
        stepCount: world.stepCount,
        timeSec: world.time,
        seed: world.seed,
        metadata: {
          totalWavesSpawned: world.totalWavesSpawned,
          totalCatches: world.surferCatches.reduce((a, b) => a + b, 0),
          totalDropIns: world.surferDropIns.reduce((a, b) => a + b, 0),
          totalCollisions: world.surferCollisions.reduce((a, b) => a + b, 0),
        },
      });
      void metrics;
    }

    return this.events.slice(startLen);
  }

  public getEvents(): TelemetryEvent[] {
    return [...this.events];
  }

  public buildBatch(world: WorldState, metrics?: EpisodeMetrics): HeatTelemetryBatch {
    return {
      version: CONTRACT_VERSION,
      heatId: this.heatId,
      tenantId: this.tenantId,
      preset: this.preset,
      config: { ...world.config },
      agentRoster: [...this.roster],
      events: this.getEvents(),
      metrics,
    };
  }

  public toJsonl(): string {
    return this.events.map((ev) => JSON.stringify(ev)).join('\n') + (this.events.length ? '\n' : '');
  }
}
