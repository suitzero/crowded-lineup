import {
  ActionCode,
  CONTRACT_VERSION,
  OBSERVATION_DIM,
  SurferStatus,
} from '../contracts/types.ts';
import type {
  ActionInput,
  EpisodeMetrics,
  Observation,
  RewardBreakdown,
  StepResult,
  SurferAction,
  WorldConfig,
  WorldState,
} from '../contracts/types.ts';
import { nextRandom, normalizeSeed } from './prng.ts';

export const DEFAULT_WORLD_CONFIG: WorldConfig = {
  version: CONTRACT_VERSION,
  width: 160,
  height: 240,
  dt: 0.1,
  maxSteps: 600,
  numSurfers: 4,
  maxWaves: 6,
  maxWhitewaters: 24,
  paddleSpeedMultiplier: 1.0,
  simplifiedWaves: false,
  waveSpawnInterval: 7.5,
};

function rand01(world: WorldState): number {
  const [val, nextState] = nextRandom(world.rngState);
  world.rngState = nextState;
  return val;
}

function randRange(world: WorldState, min: number, max: number): number {
  return min + rand01(world) * (max - min);
}

function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}

/**
 * Spawn a single incoming wave in the first available wave slot.
 */
export function spawnWave(world: WorldState, initialY = 10): number {
  const { config } = world;
  let slot = -1;
  for (let i = 0; i < config.maxWaves; i++) {
    if (world.waveActive[i] === 0) {
      slot = i;
      break;
    }
  }
  if (slot === -1) return -1;

  world.totalWavesSpawned += 1;
  world.waveActive[slot] = 1;
  world.waveId[slot] = world.totalWavesSpawned;
  world.waveY[slot] = initialY;

  if (config.simplifiedWaves) {
    // Simplified waves for early curriculum: predictable speed, middle peak, high quality
    world.waveSpeed[slot] = 22.0;
    world.wavePeakX[slot] = randRange(world, config.width * 0.3, config.width * 0.7);
    world.wavePeakDriftVx[slot] = 0.0;
    world.waveQuality[slot] = 0.85;
    world.waveBreakStartY[slot] = 75.0;
    world.waveBreakSpreadRate[slot] = 0.75;
  } else {
    world.waveSpeed[slot] = randRange(world, 20.0, 26.0);
    world.wavePeakX[slot] = randRange(world, 26.0, config.width - 26.0);
    world.wavePeakDriftVx[slot] = randRange(world, -3.2, 3.2);
    world.waveQuality[slot] = randRange(world, 0.4, 1.0);
    // Higher quality waves start breaking slightly further outside with cleaner peel
    world.waveBreakStartY[slot] = 88.0 - world.waveQuality[slot] * 20.0;
    world.waveBreakSpreadRate[slot] = randRange(world, 0.65, 0.95);
  }

  world.waveBreakWidth[slot] =
    initialY >= world.waveBreakStartY[slot]
      ? 6.0 + (initialY - world.waveBreakStartY[slot]) * world.waveBreakSpreadRate[slot]
      : 0.0;
  world.waveRiderId[slot] = -1;
  return slot;
}

function spawnWhitewater(
  world: WorldState,
  x: number,
  y: number,
  width: number,
  intensity = 1.0
): void {
  const maxWw = world.config.maxWhitewaters;
  let slot = -1;
  let oldestSlot = 0;
  let oldestRatio = -1;

  for (let i = 0; i < maxWw; i++) {
    if (world.wwActive[i] === 0) {
      slot = i;
      break;
    }
    const ratio = world.wwAge[i] / Math.max(0.1, world.wwMaxAge[i]);
    if (ratio > oldestRatio) {
      oldestRatio = ratio;
      oldestSlot = i;
    }
  }
  if (slot === -1) {
    slot = oldestSlot;
  }

  world.wwActive[slot] = 1;
  world.wwX[slot] = x;
  world.wwY[slot] = y;
  world.wwWidth[slot] = width;
  world.wwIntensity[slot] = clamp(intensity, 0.1, 1.0);
  world.wwAge[slot] = 0.0;
  world.wwMaxAge[slot] = 3.2;
}

/**
 * Find the best incoming catchable wave for a surfer at (sx, sy).
 * Returns wave slot index or -1 if none active.
 */
export function findTargetWaveSlot(world: WorldState, surferId: number): number {
  const sy = world.surferY[surferId];
  const sx = world.surferX[surferId];
  let bestSlot = -1;
  let bestScore = Infinity;

  for (let w = 0; w < world.config.maxWaves; w++) {
    if (world.waveActive[w] === 0) continue;
    const wy = world.waveY[w];
    // Wave can still be caught if it is behind the surfer or within +8 units ahead of them
    // and has not already reached the shallow shorezone
    if (wy > sy + 8.0 || wy > world.config.height - 40) continue;

    const dy = Math.max(0, sy - wy);
    const timeToReach = dy / Math.max(10, world.waveSpeed[w]);
    const dx = Math.abs(sx - world.wavePeakX[w]);
    const score = timeToReach * 18.0 + dx * 0.25 - world.waveQuality[w] * 4.0;
    if (score < bestScore) {
      bestScore = score;
      bestSlot = w;
    }
  }

  // Fallback: any active wave
  if (bestSlot === -1) {
    for (let w = 0; w < world.config.maxWaves; w++) {
      if (world.waveActive[w] === 1) {
        bestSlot = w;
        break;
      }
    }
  }
  return bestSlot;
}

/**
 * Compute bounded interception potential \Phi(s) in [0, 1.5] for reward shaping.
 */
function computeInterceptionPotential(world: WorldState, surferId: number): number {
  if (world.surferState[surferId] !== SurferStatus.FREE) {
    return 0;
  }
  const slot = findTargetWaveSlot(world, surferId);
  if (slot === -1) return 0;

  const sx = world.surferX[surferId];
  const sy = world.surferY[surferId];
  const peakX = world.wavePeakX[slot];
  // Ideal interception cross-shore Y is just after the wave begins to break
  const targetY = clamp(world.waveBreakStartY[slot] + 10.0, 65.0, 125.0);

  const normDx = Math.abs(sx - peakX) / (world.config.width * 0.5);
  const normDy = Math.abs(sy - targetY) / (world.config.height * 0.5);
  const distNorm = Math.sqrt(normDx * normDx + normDy * normDy);
  return Math.max(0, 1.2 * (1.0 - clamp(distNorm, 0, 1)));
}

/**
 * Create a new deterministic WorldState given partial config and numeric seed.
 */
export function createWorld(
  partialConfig: Partial<WorldConfig> = {},
  seed = 42
): WorldState {
  const config: WorldConfig = {
    ...DEFAULT_WORLD_CONFIG,
    ...partialConfig,
    version: CONTRACT_VERSION,
  };
  config.paddleSpeedMultiplier = clamp(config.paddleSpeedMultiplier, 0.6, 1.6);
  config.numSurfers = Math.max(1, Math.trunc(config.numSurfers));

  const normalizedSeed = normalizeSeed(seed);
  const n = config.numSurfers;
  const mw = config.maxWaves;
  const mww = config.maxWhitewaters;

  const world: WorldState = {
    version: CONTRACT_VERSION,
    seed: normalizedSeed,
    rngState: normalizedSeed,
    stepCount: 0,
    time: 0,
    config,
    totalWavesSpawned: 0,
    nextWaveSpawnTime: 4.5,

    surferX: new Array(n).fill(0),
    surferY: new Array(n).fill(0),
    surferVx: new Array(n).fill(0),
    surferVy: new Array(n).fill(0),
    surferStamina: new Array(n).fill(1.0),
    surferState: new Array(n).fill(SurferStatus.FREE),
    surferRidingWaveId: new Array(n).fill(-1),
    surferWipeoutTimer: new Array(n).fill(0),
    surferScore: new Array(n).fill(0),
    surferCatches: new Array(n).fill(0),
    surferCollisions: new Array(n).fill(0),
    surferDropIns: new Array(n).fill(0),
    surferStaminaUsed: new Array(n).fill(0),
    surferRideTime: new Array(n).fill(0),
    surferLastPotential: new Array(n).fill(0),

    waveActive: new Array(mw).fill(0),
    waveId: new Array(mw).fill(0),
    waveY: new Array(mw).fill(0),
    waveSpeed: new Array(mw).fill(0),
    wavePeakX: new Array(mw).fill(0),
    wavePeakDriftVx: new Array(mw).fill(0),
    waveQuality: new Array(mw).fill(0),
    waveBreakStartY: new Array(mw).fill(0),
    waveBreakSpreadRate: new Array(mw).fill(0),
    waveBreakWidth: new Array(mw).fill(0),
    waveRiderId: new Array(mw).fill(-1),

    wwActive: new Array(mww).fill(0),
    wwX: new Array(mww).fill(0),
    wwY: new Array(mww).fill(0),
    wwWidth: new Array(mww).fill(0),
    wwIntensity: new Array(mww).fill(0),
    wwAge: new Array(mww).fill(0),
    wwMaxAge: new Array(mww).fill(3.2),
  };

  // Position surfers across the lineup zone
  for (let i = 0; i < n; i++) {
    const laneFrac = n === 1 ? 0.5 : (i + 0.5) / n;
    const baseX = 28 + laneFrac * (config.width - 56);
    const jitterX = n === 1 ? randRange(world, -18, 18) : randRange(world, -8, 8);
    const jitterY = randRange(world, -10, 12);
    world.surferX[i] = clamp(baseX + jitterX, 16, config.width - 16);
    world.surferY[i] = clamp(92 + jitterY, 68, 130);
  }

  // Spawn initial ocean swells so the lineup immediately has waves approaching
  spawnWave(world, 34);
  spawnWave(world, 8);

  for (let i = 0; i < n; i++) {
    world.surferLastPotential[i] = computeInterceptionPotential(world, i);
  }

  return world;
}

/**
 * Deep clone a WorldState for replay/rollout verification.
 */
export function cloneWorld(world: WorldState): WorldState {
  return {
    version: CONTRACT_VERSION,
    seed: world.seed,
    rngState: world.rngState,
    stepCount: world.stepCount,
    time: world.time,
    config: { ...world.config },
    totalWavesSpawned: world.totalWavesSpawned,
    nextWaveSpawnTime: world.nextWaveSpawnTime,
    surferX: [...world.surferX],
    surferY: [...world.surferY],
    surferVx: [...world.surferVx],
    surferVy: [...world.surferVy],
    surferStamina: [...world.surferStamina],
    surferState: [...world.surferState],
    surferRidingWaveId: [...world.surferRidingWaveId],
    surferWipeoutTimer: [...world.surferWipeoutTimer],
    surferScore: [...world.surferScore],
    surferCatches: [...world.surferCatches],
    surferCollisions: [...world.surferCollisions],
    surferDropIns: [...world.surferDropIns],
    surferStaminaUsed: [...world.surferStaminaUsed],
    surferRideTime: [...world.surferRideTime],
    surferLastPotential: [...world.surferLastPotential],
    waveActive: [...world.waveActive],
    waveId: [...world.waveId],
    waveY: [...world.waveY],
    waveSpeed: [...world.waveSpeed],
    wavePeakX: [...world.wavePeakX],
    wavePeakDriftVx: [...world.wavePeakDriftVx],
    waveQuality: [...world.waveQuality],
    waveBreakStartY: [...world.waveBreakStartY],
    waveBreakSpreadRate: [...world.waveBreakSpreadRate],
    waveBreakWidth: [...world.waveBreakWidth],
    waveRiderId: [...world.waveRiderId],
    wwActive: [...world.wwActive],
    wwX: [...world.wwX],
    wwY: [...world.wwY],
    wwWidth: [...world.wwWidth],
    wwIntensity: [...world.wwIntensity],
    wwAge: [...world.wwAge],
    wwMaxAge: [...world.wwMaxAge],
  };
}

/**
 * Compute the canonical continuous normalized observation for `surferId`.
 * All features are strictly bounded in [-1, 1] or [0, 1].
 */
export function observe(world: WorldState, surferId: number): Observation {
  const { config } = world;
  const sx = world.surferX[surferId];
  const sy = world.surferY[surferId];
  const vx = world.surferVx[surferId];
  const vy = world.surferVy[surferId];
  const stamina = clamp(world.surferStamina[surferId], 0, 1);
  const status = world.surferState[surferId];

  const waveSlot = findTargetWaveSlot(world, surferId);

  let relPeakX = 0;
  let timeToWave = 1.0;
  let waveQuality = 0.5;
  let waveBreakingProgress = 0;
  let breakRefY = 85.0;

  if (waveSlot !== -1) {
    const peakX = world.wavePeakX[waveSlot];
    const wy = world.waveY[waveSlot];
    const wSpeed = Math.max(10, world.waveSpeed[waveSlot]);
    breakRefY = world.waveBreakStartY[waveSlot];

    // Normalized relative peak X in [-1, 1] (positive means peak is to the right of surfer)
    relPeakX = clamp((peakX - sx) / (config.width * 0.5), -1, 1);

    // Estimated time until wave crest reaches surfer Y, normalized to [-1, 1]
    // Negative if wave has slightly passed surfer
    const rawTime = (sy - wy) / wSpeed;
    timeToWave = clamp(rawTime / 6.0, -1, 1);

    waveQuality = clamp(world.waveQuality[waveSlot], 0, 1);

    if (wy >= breakRefY) {
      const distFromPeak = Math.abs(sx - peakX);
      const breakW = Math.max(1, world.waveBreakWidth[waveSlot]);
      waveBreakingProgress = clamp(1.0 - distFromPeak / (breakW + 18.0), 0, 1);
    }
  }

  // Relative cross-shore position compared to optimal break zone in [-1, 1]
  const relCrossShore = clamp((sy - (breakRefY + 8.0)) / (config.height * 0.4), -1, 1);

  // Competitor pressure in [0, 1]: measures nearby competitors and inside priority threat
  let competitorPressure = 0;
  for (let j = 0; j < config.numSurfers; j++) {
    if (j === surferId) continue;
    const dx = world.surferX[j] - sx;
    const dy = world.surferY[j] - sy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    let p = clamp(1.0 - dist / 45.0, 0, 1);
    if (waveSlot !== -1) {
      const myPeakDist = Math.abs(sx - world.wavePeakX[waveSlot]);
      const otherPeakDist = Math.abs(world.surferX[j] - world.wavePeakX[waveSlot]);
      if (otherPeakDist < myPeakDist) {
        p = clamp(p * 1.35, 0, 1);
      }
    }
    if (p > competitorPressure) {
      competitorPressure = p;
    }
  }

  const maxSpeedRef = 32.0 * config.paddleSpeedMultiplier;
  const surferVxNorm = clamp(vx / maxSpeedRef, -1, 1);
  const surferVyNorm = clamp(vy / maxSpeedRef, -1, 1);
  const surferStatusNorm =
    status === SurferStatus.FREE ? 0 : status === SurferStatus.RIDING ? 0.5 : 1.0;

  const features = [
    relPeakX,
    timeToWave,
    relCrossShore,
    stamina,
    competitorPressure,
    waveQuality,
    waveBreakingProgress,
    surferVxNorm,
    surferVyNorm,
    surferStatusNorm,
  ];

  // Sanity guard for length
  if (features.length !== OBSERVATION_DIM) {
    throw new Error(`Invalid observation length: ${features.length}`);
  }

  return {
    version: CONTRACT_VERSION,
    surferId,
    features,
    named: {
      relPeakX,
      timeToWave,
      relCrossShore,
      stamina,
      competitorPressure,
      waveQuality,
      waveBreakingProgress,
      surferVxNorm,
      surferVyNorm,
      surferStatusNorm,
    },
  };
}

function normalizeAction(input: ActionInput | undefined): SurferAction {
  if (typeof input === 'number') {
    return {
      version: CONTRACT_VERSION,
      action: input,
    };
  }
  if (!input) {
    return {
      version: CONTRACT_VERSION,
      action: ActionCode.WAIT,
    };
  }
  return input;
}

/**
 * Check if a surfer is currently inside active whitewater foam and return foam intensity [0, 1].
 */
export function getWhitewaterFoamAt(world: WorldState, x: number, y: number): number {
  let maxFoam = 0;
  for (let i = 0; i < world.config.maxWhitewaters; i++) {
    if (world.wwActive[i] === 0) continue;
    const dy = Math.abs(y - world.wwY[i]);
    const dx = Math.abs(x - world.wwX[i]);
    if (dy <= 6.5 && dx <= world.wwWidth[i] * 0.55) {
      if (world.wwIntensity[i] > maxFoam) {
        maxFoam = world.wwIntensity[i];
      }
    }
  }
  return maxFoam;
}

/**
 * Evaluate whether surfer `surferId` can catch wave `waveSlot` at the current step.
 */
export function evaluateCatchAttempt(
  world: WorldState,
  surferId: number,
  waveSlot: number
): {
  canCatch: boolean;
  isBreakingOverSurfer: boolean;
  distFromPeak: number;
  quality: number;
} {
  const sx = world.surferX[surferId];
  const sy = world.surferY[surferId];
  const vy = world.surferVy[surferId];
  const stamina = world.surferStamina[surferId];

  const wy = world.waveY[waveSlot];
  const peakX = world.wavePeakX[waveSlot];
  const breakStartY = world.waveBreakStartY[waveSlot];
  const breakWidth = world.waveBreakWidth[waveSlot];
  const quality = world.waveQuality[waveSlot];

  const dy = sy - wy; // Positive when surfer is slightly shoreward of crest
  const distFromPeak = Math.abs(sx - peakX);

  // Crest proximity: surfer must be within [-7.5, +9.5] units of the wave crest
  const inCrestWindow = dy >= -7.5 && dy <= 9.5;

  // Wave must be at or approaching breaking steepness, and not yet on the sandbar shore
  const inCrossShoreWindow = wy >= breakStartY - 8.0 && wy <= world.config.height - 32.0;

  // Horizontal catch window: near the peak or the peeling shoulder
  const maxHorizontalWindow = Math.max(18.0, breakWidth + 15.0);
  const inHorizontalWindow = distFromPeak <= maxHorizontalWindow;

  // Avoid closed-out whitewater soup far behind an old wave's peak
  const isDeepFoamCloseout =
    wy - breakStartY > 52.0 && distFromPeak < breakWidth * 0.45;

  // Physical conditions: stamina, not paddling hard offshore, sufficient paddle speed multiplier
  const hasStamina = stamina >= 0.08;
  const validTravelDirection = vy >= -3.5;

  const canCatch =
    inCrestWindow &&
    inCrossShoreWindow &&
    inHorizontalWindow &&
    !isDeepFoamCloseout &&
    hasStamina &&
    validTravelDirection;

  const isBreakingOverSurfer =
    inCrestWindow && wy >= breakStartY + 4.0 && distFromPeak <= breakWidth + 6.0;

  return {
    canCatch,
    isBreakingOverSurfer,
    distFromPeak,
    quality,
  };
}

/**
 * Advance the simulation by one fixed timestep `dt`.
 */
export function step(
  world: WorldState,
  actions: ActionInput[],
  dt: number = world.config.dt
): StepResult {
  const { config } = world;
  const n = config.numSurfers;
  const effectiveDt = dt > 0 ? dt : config.dt;

  world.stepCount += 1;
  world.time = +(world.time + effectiveDt).toFixed(4);

  const breakdowns: RewardBreakdown[] = new Array(n);
  const rewards: number[] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    breakdowns[i] = {
      version: CONTRACT_VERSION,
      total: 0,
      catchReward: 0,
      qualityBonus: 0,
      priorityBonus: 0,
      rideReward: 0,
      collisionPenalty: 0,
      dropInPenalty: 0,
      staminaPenalty: 0,
      shapingReward: 0,
    };
  }

  // 1. Advance existing waves and spawn whitewater trails
  for (let w = 0; w < config.maxWaves; w++) {
    if (world.waveActive[w] === 0) continue;

    world.waveY[w] += world.waveSpeed[w] * effectiveDt;
    world.wavePeakX[w] = clamp(
      world.wavePeakX[w] + world.wavePeakDriftVx[w] * effectiveDt,
      20,
      config.width - 20
    );

    if (world.waveY[w] >= world.waveBreakStartY[w]) {
      const brokenDist = world.waveY[w] - world.waveBreakStartY[w];
      world.waveBreakWidth[w] = Math.min(
        config.width * 0.85,
        6.0 + brokenDist * world.waveBreakSpreadRate[w]
      );

      // Emit whitewater trail periodically every ~4 steps
      if ((world.stepCount + w) % 4 === 0) {
        spawnWhitewater(
          world,
          world.wavePeakX[w],
          world.waveY[w] - 2.0,
          Math.max(14, world.waveBreakWidth[w] * 1.55),
          0.95
        );
      }
    } else {
      world.waveBreakWidth[w] = 0;
    }

    // Deactivate wave when it hits the shoreline
    if (world.waveY[w] >= config.height - 16) {
      // Release any riding surfers cleanly
      for (let i = 0; i < n; i++) {
        if (
          world.surferState[i] === SurferStatus.RIDING &&
          world.surferRidingWaveId[i] === w
        ) {
          world.surferState[i] = SurferStatus.FREE;
          world.surferRidingWaveId[i] = -1;
        }
      }
      world.waveActive[w] = 0;
      world.waveRiderId[w] = -1;
    }
  }

  // 2. Spawn new waves on schedule
  if (world.time >= world.nextWaveSpawnTime) {
    spawnWave(world, 8.0);
    const jitter = config.simplifiedWaves ? 0 : randRange(world, -1.2, 1.5);
    world.nextWaveSpawnTime = world.time + Math.max(4.5, config.waveSpawnInterval + jitter);
  }

  // 3. Update whitewater lifetime and decay
  for (let i = 0; i < config.maxWhitewaters; i++) {
    if (world.wwActive[i] === 0) continue;
    world.wwAge[i] += effectiveDt;
    world.wwY[i] += 4.5 * effectiveDt; // slight shoreward foam drift
    if (world.wwAge[i] >= world.wwMaxAge[i] || world.wwY[i] >= config.height - 8) {
      world.wwActive[i] = 0;
      world.wwIntensity[i] = 0;
    } else {
      world.wwIntensity[i] = Math.max(0, 1.0 - world.wwAge[i] / world.wwMaxAge[i]);
    }
  }

  // Track catch attempts per wave slot on this step to resolve priority and drop-ins
  const catchAttemptsByWave: Map<
    number,
    Array<{ surferId: number; distFromPeak: number; quality: number }>
  > = new Map();

  // 4. Process surfer actions, movement, and stamina
  const baseMaxSpeed = 26.0 * config.paddleSpeedMultiplier;
  const accelRate = 85.0 * config.paddleSpeedMultiplier;

  for (let i = 0; i < n; i++) {
    const act = normalizeAction(actions[i]);
    const status = world.surferState[i];

    // Handle WIPED_OUT state recovery
    if (status === SurferStatus.WIPED_OUT) {
      world.surferWipeoutTimer[i] -= effectiveDt;
      world.surferVx[i] *= Math.max(0, 1 - 5.0 * effectiveDt);
      world.surferVy[i] = 6.0; // pushed slightly toward shore by foam
      world.surferX[i] = clamp(
        world.surferX[i] + world.surferVx[i] * effectiveDt,
        10,
        config.width - 10
      );
      world.surferY[i] = clamp(
        world.surferY[i] + world.surferVy[i] * effectiveDt,
        12,
        config.height - 18
      );
      world.surferStamina[i] = clamp(world.surferStamina[i] + 0.08 * effectiveDt, 0, 1);
      if (world.surferWipeoutTimer[i] <= 0) {
        world.surferState[i] = SurferStatus.FREE;
        world.surferWipeoutTimer[i] = 0;
      }
      continue;
    }

    // Handle RIDING state
    if (status === SurferStatus.RIDING) {
      const w = world.surferRidingWaveId[i];
      if (w < 0 || w >= config.maxWaves || world.waveActive[w] === 0) {
        world.surferState[i] = SurferStatus.FREE;
        world.surferRidingWaveId[i] = -1;
      } else {
        // Kick out if surfer chooses OUT
        if (act.action === ActionCode.OUT) {
          world.surferState[i] = SurferStatus.FREE;
          world.surferRidingWaveId[i] = -1;
          if (world.waveRiderId[w] === i) {
            world.waveRiderId[w] = -1;
          }
          world.surferVy[i] = -10.0;
        } else {
          // Ride along the breaking shoulder
          const peakX = world.wavePeakX[w];
          const shoulderDir = world.surferX[i] >= peakX ? 1 : -1;
          let lateralSpeed = shoulderDir * 14.0;
          if (act.action === ActionCode.LEFT) lateralSpeed = -18.0;
          else if (act.action === ActionCode.RIGHT) lateralSpeed = 18.0;

          world.surferVx[i] = lateralSpeed;
          world.surferVy[i] = world.waveSpeed[w];
          world.surferX[i] = clamp(
            world.surferX[i] + world.surferVx[i] * effectiveDt,
            10,
            config.width - 10
          );
          world.surferY[i] = clamp(world.waveY[w] + 2.0, 12, config.height - 18);
          world.surferStamina[i] = clamp(
            world.surferStamina[i] + 0.12 * effectiveDt,
            0,
            1
          );
          world.surferRideTime[i] = +(world.surferRideTime[i] + effectiveDt).toFixed(4);

          const stepRideScore = 1.5 * world.waveQuality[w] * effectiveDt * 10;
          world.surferScore[i] = +(world.surferScore[i] + stepRideScore).toFixed(2);
          breakdowns[i].rideReward += 0.15 * world.waveQuality[w];

          // Finish ride cleanly near the beach
          if (world.surferY[i] >= config.height - 26) {
            world.surferState[i] = SurferStatus.FREE;
            world.surferRidingWaveId[i] = -1;
            if (world.waveRiderId[w] === i) {
              world.waveRiderId[w] = -1;
            }
          }
          continue;
        }
      }
    }

    // FREE state: paddling or waiting or attempting CATCH
    let dirX = 0;
    let dirY = 0;
    let isPaddling = false;

    if (
      act.paddle &&
      typeof act.targetX === 'number' &&
      typeof act.targetY === 'number'
    ) {
      const dx = act.targetX - world.surferX[i];
      const dy = act.targetY - world.surferY[i];
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 2.0) {
        dirX = dx / dist;
        dirY = dy / dist;
        isPaddling = true;
      }
    } else {
      switch (act.action) {
        case ActionCode.LEFT:
          dirX = -1;
          isPaddling = true;
          break;
        case ActionCode.RIGHT:
          dirX = 1;
          isPaddling = true;
          break;
        case ActionCode.OUT:
          dirY = -1;
          isPaddling = true;
          break;
        case ActionCode.IN:
          dirY = 1;
          isPaddling = true;
          break;
        case ActionCode.CATCH:
          // Attempting a catch also gives a shoreward takeoff paddle stroke
          dirY = 1;
          isPaddling = true;
          break;
        case ActionCode.WAIT:
        default:
          isPaddling = false;
          break;
      }
    }

    const stamina = world.surferStamina[i];
    // Stamina constraint on speed: below 0.25 stamina, max speed drops sharply
    const staminaFactor =
      stamina <= 0.01
        ? 0.32
        : stamina < 0.25
          ? 0.35 + 0.65 * (stamina / 0.25)
          : 1.0;

    const foamIntensity = getWhitewaterFoamAt(world, world.surferX[i], world.surferY[i]);
    const foamSpeedPenalty = foamIntensity > 0 ? 1.0 - 0.35 * foamIntensity : 1.0;
    const effectiveMaxSpeed = baseMaxSpeed * staminaFactor * foamSpeedPenalty;

    if (isPaddling) {
      const targetVx = dirX * effectiveMaxSpeed;
      const targetVy = dirY * effectiveMaxSpeed;

      const dvx = targetVx - world.surferVx[i];
      const dvy = targetVy - world.surferVy[i];
      const maxDv = accelRate * effectiveDt;
      const dvMag = Math.sqrt(dvx * dvx + dvy * dvy);

      if (dvMag <= maxDv || dvMag === 0) {
        world.surferVx[i] = targetVx;
        world.surferVy[i] = targetVy;
      } else {
        world.surferVx[i] += (dvx / dvMag) * maxDv;
        world.surferVy[i] += (dvy / dvMag) * maxDv;
      }

      // Stamina drain
      const extraFoamDrain = dirY < 0 && foamIntensity > 0 ? 0.09 * foamIntensity : 0;
      const drain = (0.11 + extraFoamDrain) * effectiveDt;
      const prevStamina = world.surferStamina[i];
      world.surferStamina[i] = clamp(prevStamina - drain, 0, 1);
      const actualUsed = prevStamina - world.surferStamina[i];
      world.surferStaminaUsed[i] = +(world.surferStaminaUsed[i] + actualUsed).toFixed(4);
      breakdowns[i].staminaPenalty -= 0.015;
    } else {
      // Water drag & stamina recovery when resting
      const drag = Math.max(0, 1.0 - 4.5 * effectiveDt);
      world.surferVx[i] *= drag;
      world.surferVy[i] *= drag;
      if (Math.abs(world.surferVx[i]) < 0.05) world.surferVx[i] = 0;
      if (Math.abs(world.surferVy[i]) < 0.05) world.surferVy[i] = 0;

      world.surferStamina[i] = clamp(
        world.surferStamina[i] + 0.18 * effectiveDt,
        0,
        1
      );
    }

    // Apply shoreward whitewater push if caught in foam while idle
    if (foamIntensity > 0.35 && !isPaddling) {
      world.surferVy[i] += 4.0 * foamIntensity * effectiveDt;
    }

    // Integrate position
    world.surferX[i] = clamp(
      world.surferX[i] + world.surferVx[i] * effectiveDt,
      10,
      config.width - 10
    );
    world.surferY[i] = clamp(
      world.surferY[i] + world.surferVy[i] * effectiveDt,
      12,
      config.height - 18
    );

    // Evaluate CATCH action
    if (act.action === ActionCode.CATCH) {
      let bestWave = -1;
      let minCrestDist = Infinity;
      for (let w = 0; w < config.maxWaves; w++) {
        if (world.waveActive[w] === 0) continue;
        const d = Math.abs(world.surferY[i] - world.waveY[w]);
        if (d < minCrestDist) {
          minCrestDist = d;
          bestWave = w;
        }
      }

      if (bestWave !== -1) {
        const evalRes = evaluateCatchAttempt(world, i, bestWave);
        if (evalRes.canCatch) {
          const list = catchAttemptsByWave.get(bestWave) ?? [];
          list.push({
            surferId: i,
            distFromPeak: evalRes.distFromPeak,
            quality: evalRes.quality,
          });
          catchAttemptsByWave.set(bestWave, list);
        } else {
          // Missed takeoff penalty if spamming catch or wiped out by breaking lip
          breakdowns[i].staminaPenalty -= 0.05;
          if (evalRes.isBreakingOverSurfer && world.surferStamina[i] < 0.12) {
            world.surferState[i] = SurferStatus.WIPED_OUT;
            world.surferWipeoutTimer[i] = 1.0;
          }
        }
      } else {
        breakdowns[i].staminaPenalty -= 0.04;
      }
    }
  }

  // 5. Resolve wave priority and drop-ins for all catch attempts on this step
  for (const [waveSlot, attempts] of catchAttemptsByWave.entries()) {
    // Sort attempts by closeness to peak (inside surfer has priority)
    attempts.sort((a, b) => a.distFromPeak - b.distFromPeak);

    const existingRider = world.waveRiderId[waveSlot];
    const hasExistingPriorityRider =
      existingRider !== -1 &&
      world.surferState[existingRider] === SurferStatus.RIDING &&
      world.surferRidingWaveId[waveSlot] !== undefined;

    for (let idx = 0; idx < attempts.length; idx++) {
      const candidate = attempts[idx];
      const sid = candidate.surferId;

      // Check if an existing rider already holds priority closer to the peak, or
      // another surfer in the same step is closer to the peak
      let droppedIn = false;
      if (hasExistingPriorityRider && existingRider !== sid) {
        const existingDist = Math.abs(
          world.surferX[existingRider] - world.wavePeakX[waveSlot]
        );
        if (existingDist <= candidate.distFromPeak + 6.0) {
          droppedIn = true;
        }
      }
      if (idx > 0) {
        droppedIn = true;
      }

      if (droppedIn) {
        // Drop-in violation!
        world.surferDropIns[sid] += 1;
        breakdowns[sid].dropInPenalty -= 8.0;
        world.surferState[sid] = SurferStatus.WIPED_OUT;
        world.surferWipeoutTimer[sid] = 1.2;
        world.surferRidingWaveId[sid] = -1;
      } else {
        // Valid wave catch with priority!
        world.surferState[sid] = SurferStatus.RIDING;
        world.surferRidingWaveId[sid] = waveSlot;
        world.waveRiderId[waveSlot] = sid;
        world.surferCatches[sid] += 1;

        const catchPts = 25.0 + Math.round(candidate.quality * 25.0);
        world.surferScore[sid] = +(world.surferScore[sid] + catchPts).toFixed(2);

        breakdowns[sid].catchReward += 12.0;
        breakdowns[sid].qualityBonus += +(6.0 * candidate.quality).toFixed(4);
        breakdowns[sid].priorityBonus += 3.0;
      }
    }
  }

  // 6. Check pairwise surfer collisions
  const collisionRadius = 6.5;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (
        world.surferState[i] === SurferStatus.WIPED_OUT ||
        world.surferState[j] === SurferStatus.WIPED_OUT
      ) {
        continue;
      }
      const dx = world.surferX[j] - world.surferX[i];
      const dy = world.surferY[j] - world.surferY[i];
      const distSq = dx * dx + dy * dy;
      if (distSq < collisionRadius * collisionRadius) {
        const dist = Math.max(0.1, Math.sqrt(distSq));
        const nx = dx / dist;
        const ny = dy / dist;
        const overlap = (collisionRadius - dist) * 0.5 + 0.5;

        world.surferX[i] = clamp(world.surferX[i] - nx * overlap, 10, config.width - 10);
        world.surferY[i] = clamp(world.surferY[i] - ny * overlap, 12, config.height - 18);
        world.surferX[j] = clamp(world.surferX[j] + nx * overlap, 10, config.width - 10);
        world.surferY[j] = clamp(world.surferY[j] + ny * overlap, 12, config.height - 18);

        world.surferVx[i] *= -0.3;
        world.surferVy[i] *= -0.3;
        world.surferVx[j] *= -0.3;
        world.surferVy[j] *= -0.3;

        world.surferCollisions[i] += 1;
        world.surferCollisions[j] += 1;
        breakdowns[i].collisionPenalty -= 6.0;
        breakdowns[j].collisionPenalty -= 6.0;

        // If either was riding, collision knocks them off
        if (world.surferState[i] === SurferStatus.RIDING) {
          const w = world.surferRidingWaveId[i];
          if (w >= 0 && world.waveRiderId[w] === i) world.waveRiderId[w] = -1;
          world.surferState[i] = SurferStatus.WIPED_OUT;
          world.surferWipeoutTimer[i] = 1.0;
          world.surferRidingWaveId[i] = -1;
        }
        if (world.surferState[j] === SurferStatus.RIDING) {
          const w = world.surferRidingWaveId[j];
          if (w >= 0 && world.waveRiderId[w] === j) world.waveRiderId[w] = -1;
          world.surferState[j] = SurferStatus.WIPED_OUT;
          world.surferWipeoutTimer[j] = 1.0;
          world.surferRidingWaveId[j] = -1;
        }
      }
    }
  }

  // 7. Compute bounded potential-based shaping and total step reward
  const gamma = 0.98;
  for (let i = 0; i < n; i++) {
    const newPotential = computeInterceptionPotential(world, i);
    const rawShaping = gamma * newPotential - world.surferLastPotential[i];
    world.surferLastPotential[i] = newPotential;

    // Limit shaping so terminal / catch rewards strictly dominate
    const shaping =
      breakdowns[i].catchReward > 0 ? 0 : clamp(rawShaping * 0.5, -0.06, 0.06);
    breakdowns[i].shapingReward = +shaping.toFixed(4);

    const b = breakdowns[i];
    const total =
      b.catchReward +
      b.qualityBonus +
      b.priorityBonus +
      b.rideReward +
      b.collisionPenalty +
      b.dropInPenalty +
      b.staminaPenalty +
      b.shapingReward;
    b.total = +total.toFixed(4);
    rewards[i] = b.total;
  }

  return {
    version: CONTRACT_VERSION,
    rewards,
    breakdowns,
    done: isEpisodeDone(world),
  };
}

/**
 * Check whether the simulation episode has finished.
 */
export function isEpisodeDone(world: WorldState): boolean {
  return world.stepCount >= world.config.maxSteps;
}

/**
 * Extract versioned summary metrics for the episode.
 */
export function getEpisodeMetrics(world: WorldState): EpisodeMetrics {
  const waves = Math.max(1, world.totalWavesSpawned);
  const catchRates = world.surferCatches.map((c) => +(c / waves).toFixed(4));
  return {
    version: CONTRACT_VERSION,
    seed: world.seed,
    totalSteps: world.stepCount,
    durationSec: +world.time.toFixed(2),
    totalWavesSpawned: world.totalWavesSpawned,
    surferCatches: [...world.surferCatches],
    surferCatchRates: catchRates,
    surferScores: [...world.surferScore],
    surferCollisions: [...world.surferCollisions],
    surferDropIns: [...world.surferDropIns],
    surferStaminaUsed: [...world.surferStaminaUsed],
    surferRideTime: [...world.surferRideTime],
  };
}
