export const CONTRACT_VERSION = 2 as const;

/**
 * Discrete action codes shared across tabular Q-learning, rule bots,
 * human keyboard controls, and neural/ONNX policy heads.
 */
export const ActionCode = {
  WAIT: 0,
  LEFT: 1,
  RIGHT: 2,
  OUT: 3, // Toward offshore (top, -y)
  IN: 4, // Toward shoreline (bottom, +y)
  CATCH: 5, // Attempt wave takeoff
} as const;

export type ActionCode = (typeof ActionCode)[keyof typeof ActionCode];

export const NUM_DISCRETE_ACTIONS = 6 as const;

export const ACTION_NAMES: Record<ActionCode, string> = {
  [ActionCode.WAIT]: 'WAIT',
  [ActionCode.LEFT]: 'LEFT',
  [ActionCode.RIGHT]: 'RIGHT',
  [ActionCode.OUT]: 'OUT',
  [ActionCode.IN]: 'IN',
  [ActionCode.CATCH]: 'CATCH',
};

/**
 * Surfer physical state codes stored in WorldState.surferState.
 */
export const SurferStatus = {
  FREE: 0,
  RIDING: 1,
  WIPED_OUT: 2,
} as const;

export type SurferStatus = (typeof SurferStatus)[keyof typeof SurferStatus];

export type LineupPresetId =
  | 'classic-4'
  | 'crowded-16'
  | 'crowded-24'
  | 'chaos-40';

/**
 * Action input for a single surfer on one simulation step.
 * Can be specified as a discrete ActionCode (RL / bots / keyboard) or
 * with optional continuous target coordinates (mobile touch target + hold paddle).
 */
export interface SurferAction {
  version: typeof CONTRACT_VERSION;
  action: ActionCode;
  /** Optional continuous target X for touch-to-paddle input */
  targetX?: number;
  /** Optional continuous target Y for touch-to-paddle input */
  targetY?: number;
  /** Whether paddle is held when using continuous touch target */
  paddle?: boolean;
}

export type ActionInput = ActionCode | SurferAction;

/**
 * Simulation configuration contract.
 */
export interface WorldConfig {
  version: typeof CONTRACT_VERSION;
  width: number;
  height: number;
  dt: number;
  maxSteps: number;
  numSurfers: number;
  maxWaves: number;
  maxWhitewaters: number;
  /** Paddling speed multiplier (0.6x - 1.6x, default 1.0) */
  paddleSpeedMultiplier: number;
  /** Simplified wave mode for early curriculum stage */
  simplifiedWaves: boolean;
  /** Wave spawn interval in seconds */
  waveSpawnInterval: number;
}

/**
 * Flat, data-oriented world state representation.
 * All dynamic entity attributes are stored in numeric arrays for fast cloning,
 * JSON serialization, and direct parity with batched Python/CUDA tensor layouts.
 */
export interface WorldState {
  version: typeof CONTRACT_VERSION;
  seed: number;
  rngState: number;
  stepCount: number;
  time: number;
  config: WorldConfig;

  // Global wave counters
  totalWavesSpawned: number;
  nextWaveSpawnTime: number;

  // Surfer arrays (length = config.numSurfers)
  surferX: number[];
  surferY: number[];
  surferVx: number[];
  surferVy: number[];
  surferStamina: number[];
  surferState: number[];
  surferRidingWaveId: number[];
  surferWipeoutTimer: number[];
  surferScore: number[];
  surferCatches: number[];
  surferCollisions: number[];
  surferDropIns: number[];
  surferStaminaUsed: number[];
  surferRideTime: number[];
  surferLastPotential: number[];

  // Wave pool arrays (length = config.maxWaves)
  waveActive: number[];
  waveId: number[];
  waveY: number[];
  waveSpeed: number[];
  wavePeakX: number[];
  wavePeakDriftVx: number[];
  waveQuality: number[];
  waveBreakStartY: number[];
  waveBreakSpreadRate: number[];
  waveBreakWidth: number[];
  waveRiderId: number[];

  // Whitewater pool arrays (length = config.maxWhitewaters)
  wwActive: number[];
  wwX: number[];
  wwY: number[];
  wwWidth: number[];
  wwIntensity: number[];
  wwAge: number[];
  wwMaxAge: number[];
}

/**
 * Canonical continuous normalized observation vector (v2: 16 features in [-1, 1] or [0, 1]).
 * Features [0..9] preserve v1 semantics; [10..15] add spatial crowd sectors & secondary peak awareness.
 */
export const OBSERVATION_DIM = 16 as const;

export interface Observation {
  version: typeof CONTRACT_VERSION;
  surferId: number;
  /**
   * Canonical normalized feature vector (length = OBSERVATION_DIM = 16):
   * [0] relPeakX: relative horizontal position to primary target wave peak in [-1, 1]
   * [1] timeToWave: normalized estimated time until primary wave reaches surfer in [-1, 1]
   * [2] relCrossShore: surfer cross-shore position relative to break zone in [-1, 1]
   * [3] stamina: surfer stamina in [0, 1]
   * [4] competitorPressure: nearby competitor crowding/priority pressure in [0, 1]
   * [5] waveQuality: target wave quality in [0, 1]
   * [6] waveBreakingProgress: how broken the wave is at surfer X in [0, 1]
   * [7] surferVxNorm: normalized surfer horizontal velocity in [-1, 1]
   * [8] surferVyNorm: normalized surfer cross-shore velocity in [-1, 1]
   * [9] surferStatusNorm: 0 = free, 0.5 = riding, 1.0 = wiped out
   * [10] crowdDensityLeft: normalized surfer density to left [-45, -6] in [0, 1]
   * [11] crowdDensityCenter: normalized surfer density in local pocket [-6, +6] in [0, 1]
   * [12] crowdDensityRight: normalized surfer density to right [+6, +45] in [0, 1]
   * [13] insidePriorityThreat: normalized threat from surfers closer to primary peak in [0, 1]
   * [14] secondaryPeakRelX: relative horizontal position to secondary wave peak in [-1, 1]
   * [15] secondaryPeakTimeToWave: normalized time until secondary wave reaches surfer in [-1, 1]
   */
  features: number[];
  /** Named view of features used by Q-table discretization, RuleBots, and debugging */
  named: {
    relPeakX: number;
    timeToWave: number;
    relCrossShore: number;
    stamina: number;
    competitorPressure: number;
    waveQuality: number;
    waveBreakingProgress: number;
    surferVxNorm: number;
    surferVyNorm: number;
    surferStatusNorm: number;
    crowdDensityLeft: number;
    crowdDensityCenter: number;
    crowdDensityRight: number;
    insidePriorityThreat: number;
    secondaryPeakRelX: number;
    secondaryPeakTimeToWave: number;
  };
}

/**
 * Reward breakdown for a single surfer on one step.
 */
export interface RewardBreakdown {
  version: typeof CONTRACT_VERSION;
  total: number;
  catchReward: number;
  qualityBonus: number;
  priorityBonus: number;
  rideReward: number;
  collisionPenalty: number;
  dropInPenalty: number;
  staminaPenalty: number;
  shapingReward: number;
}

/**
 * Result returned by `step(world, actions, dt)`.
 */
export interface StepResult {
  version: typeof CONTRACT_VERSION;
  rewards: number[];
  breakdowns: RewardBreakdown[];
  done: boolean;
}

/**
 * Episode summary metrics contract.
 */
export interface EpisodeMetrics {
  version: typeof CONTRACT_VERSION;
  seed: number;
  totalSteps: number;
  durationSec: number;
  totalWavesSpawned: number;
  surferCatches: number[];
  surferCatchRates: number[];
  surferScores: number[];
  surferCollisions: number[];
  surferDropIns: number[];
  surferStaminaUsed: number[];
  surferRideTime: number[];
}

/**
 * Supported policy types behind the unified Policy interface.
 */
export type PolicyType = 'human' | 'rule-bot' | 'q-table' | 'neural-onnx';

export type BotArchetype =
  | 'local-charger'
  | 'shoulder-hopper'
  | 'patient-picker'
  | 'rookie';

/**
 * Versioned metadata attached to any saved or exported policy.
 */
export interface PolicyMetadata {
  version: typeof CONTRACT_VERSION;
  name: string;
  policyType: PolicyType;
  observationVersion: typeof CONTRACT_VERSION;
  observationDim: number;
  actionVersion: typeof CONTRACT_VERSION;
  numActions: number;
  createdAtIso: string;
  trainingEpisodes: number;
  paddleSpeedMultiplier: number;
  numCompetitors: number;
  avgCatchRate: number;
  avgReward: number;
}

/**
 * Serialized policy payload for localStorage persistence and JSON file export/import.
 */
export interface SerializedPolicy {
  version: typeof CONTRACT_VERSION;
  metadata: PolicyMetadata;
  qTable?: Record<string, number[]>;
  onnxModelBase64?: string;
}

/**
 * Deterministic reference scenario fixture for cross-runtime (TypeScript <-> Python/PyTorch) parity tests.
 */
export interface ReferenceScenarioStep {
  stepIndex: number;
  actions: ActionCode[];
  expectedObservations: number[][];
  expectedRewards: number[];
  expectedSurferX: number[];
  expectedSurferY: number[];
  expectedSurferStamina: number[];
  expectedSurferState: number[];
}

export interface ReferenceScenario {
  version: typeof CONTRACT_VERSION;
  name: string;
  description: string;
  seed: number;
  config: WorldConfig;
  steps: ReferenceScenarioStep[];
  expectedMetrics: EpisodeMetrics;
}

export type TelemetryEventType =
  | 'heat_started'
  | 'wave_spawned'
  | 'takeoff_caught'
  | 'drop_in_violation'
  | 'surfer_collision'
  | 'heat_completed';

export interface TelemetryEvent {
  version: typeof CONTRACT_VERSION;
  eventId: string;
  heatId: string;
  tenantId: string;
  eventType: TelemetryEventType;
  stepCount: number;
  timeSec: number;
  seed: number;
  surferId?: number;
  otherSurferId?: number;
  agentPolicyId?: string;
  waveId?: number;
  x?: number;
  y?: number;
  rewardDelta?: number;
  metadata?: Record<string, string | number | boolean>;
}

export interface HeatTelemetryBatch {
  version: typeof CONTRACT_VERSION;
  heatId: string;
  tenantId: string;
  preset: LineupPresetId | 'custom';
  config: WorldConfig;
  agentRoster: Array<{
    surferId: number;
    agentId: string;
    policyType: PolicyType;
    label: string;
  }>;
  events: TelemetryEvent[];
  metrics?: EpisodeMetrics;
}
