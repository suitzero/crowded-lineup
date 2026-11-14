import {
  ActionCode,
  CONTRACT_VERSION,
  NUM_DISCRETE_ACTIONS,
  OBSERVATION_DIM,
} from '../contracts/types.ts';
import type {
  Observation,
  PolicyMetadata,
  PolicyType,
  SerializedPolicy,
} from '../contracts/types.ts';
import { SeededRng } from '../sim/prng.ts';
import type { Policy } from './policy.ts';

/**
 * Discretize the canonical continuous normalized Observation into a compact
 * state key ONLY inside the Q-table agent (5 x 5 x 3 x 3 x 2 = 450 states).
 *
 * Uses the 5 spec-required features:
 *  - relPeakX: relative peak horizontal position (5 bins)
 *  - timeToWave: estimated time until wave reaches surfer (5 bins)
 *  - relCrossShore: relative cross-shore position (3 bins)
 *  - stamina: surfer stamina (3 bins)
 *  - competitorPressure: nearby competitor pressure (2 bins)
 */
export function discretizeObservation(obs: Observation): string {
  const relPeakX = obs.features[0];
  const timeToWave = obs.features[1];
  const relCrossShore = obs.features[2];
  const stamina = obs.features[3];
  const competitorPressure = obs.features[4];

  let peakBin = 2;
  if (relPeakX < -0.2) peakBin = 0;
  else if (relPeakX < -0.055) peakBin = 1;
  else if (relPeakX <= 0.055) peakBin = 2;
  else if (relPeakX <= 0.2) peakBin = 3;
  else peakBin = 4;

  let timeBin = 2;
  if (timeToWave < -0.05) timeBin = 0; // passed
  else if (timeToWave <= 0.075) timeBin = 1; // in takeoff catch window!
  else if (timeToWave <= 0.2) timeBin = 2; // imminent
  else if (timeToWave <= 0.45) timeBin = 3; // approaching
  else timeBin = 4; // far outside

  let crossBin = 1;
  if (relCrossShore < -0.1) crossBin = 0; // too far outside
  else if (relCrossShore <= 0.14) crossBin = 1; // prime takeoff zone
  else crossBin = 2; // too far inside

  let stamBin = 2;
  if (stamina < 0.22) stamBin = 0;
  else if (stamina < 0.6) stamBin = 1;
  else stamBin = 2;

  const pressureBin = competitorPressure >= 0.45 ? 1 : 0;

  return `${peakBin}:${timeBin}:${crossBin}:${stamBin}:${pressureBin}`;
}

export interface QLearningConfig {
  alpha: number; // learning rate
  gamma: number; // discount factor
  epsilon: number; // exploration rate
}

export const DEFAULT_Q_CONFIG: QLearningConfig = {
  alpha: 0.18,
  gamma: 0.96,
  epsilon: 0.15,
};

/**
 * Tabular Q-learning policy implementing the shared Policy interface.
 */
export class QTablePolicy implements Policy {
  public readonly id: string;
  public name: string;
  public readonly type: PolicyType = 'q-table';

  private table: Map<string, number[]> = new Map();
  private metadata: PolicyMetadata;

  constructor(name = 'Trained Q-Surfer', initialMeta?: Partial<PolicyMetadata>) {
    this.id = 'q-table-policy';
    this.name = name;
    this.metadata = {
      version: CONTRACT_VERSION,
      name,
      policyType: 'q-table',
      observationVersion: CONTRACT_VERSION,
      observationDim: OBSERVATION_DIM,
      actionVersion: CONTRACT_VERSION,
      numActions: NUM_DISCRETE_ACTIONS,
      createdAtIso: '2026-11-14T00:00:00.000Z',
      trainingEpisodes: 0,
      paddleSpeedMultiplier: 1.0,
      numCompetitors: 0,
      avgCatchRate: 0,
      avgReward: 0,
      ...initialMeta,
    };
  }

  public getStateCount(): number {
    return this.table.size;
  }

  public getQValues(stateKey: string): number[] {
    let q = this.table.get(stateKey);
    if (!q) {
      q = new Array(NUM_DISCRETE_ACTIONS).fill(0);
      this.table.set(stateKey, q);
    }
    return q;
  }

  /**
   * Select greedy best action (used during gameplay / evaluation).
   */
  public selectAction(observation: Observation): ActionCode {
    // If currently riding a wave (statusNorm == 0.5), stay on the wave face
    if (observation.features[9] >= 0.4 && observation.features[9] <= 0.6) {
      return ActionCode.WAIT;
    }

    const key = discretizeObservation(observation);
    const q = this.table.get(key);
    if (!q) {
      // Unseen state fallback: move toward peak or wait
      if (observation.features[1] >= -0.05 && observation.features[1] <= 0.075 && Math.abs(observation.features[0]) <= 0.2) {
        return ActionCode.CATCH;
      }
      if (observation.features[0] > 0.07) return ActionCode.RIGHT;
      if (observation.features[0] < -0.07) return ActionCode.LEFT;
      return ActionCode.WAIT;
    }

    let bestAction: ActionCode = ActionCode.WAIT;
    let bestVal = -Infinity;
    for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
      if (q[a] > bestVal) {
        bestVal = q[a];
        bestAction = a as ActionCode;
      }
    }
    return bestAction;
  }

  /**
   * Select action with epsilon-greedy exploration using a deterministic SeededRng.
   */
  public selectActionEpsilonGreedy(
    observation: Observation,
    epsilon: number,
    rng: SeededRng
  ): ActionCode {
    if (rng.next() < epsilon) {
      return rng.int(0, NUM_DISCRETE_ACTIONS) as ActionCode;
    }
    const key = discretizeObservation(observation);
    const q = this.getQValues(key);
    let bestAction: ActionCode = ActionCode.WAIT;
    let bestVal = -Infinity;
    for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
      if (q[a] > bestVal) {
        bestVal = q[a];
        bestAction = a as ActionCode;
      }
    }
    return bestAction;
  }

  /**
   * Perform a single standard Bellman Q-learning update:
   * Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
   */
  public update(
    obs: Observation,
    action: ActionCode,
    reward: number,
    nextObs: Observation,
    done: boolean,
    alpha = DEFAULT_Q_CONFIG.alpha,
    gamma = DEFAULT_Q_CONFIG.gamma
  ): number {
    const stateKey = discretizeObservation(obs);
    const nextKey = discretizeObservation(nextObs);

    const q = this.getQValues(stateKey);
    const nextQ = this.getQValues(nextKey);

    let maxNextQ = 0;
    if (!done) {
      maxNextQ = -Infinity;
      for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
        if (nextQ[a] > maxNextQ) {
          maxNextQ = nextQ[a];
        }
      }
    }

    const currentQ = q[action];
    const target = reward + (done ? 0 : gamma * maxNextQ);
    const tdError = target - currentQ;
    q[action] = +(currentQ + alpha * tdError).toFixed(6);
    return tdError;
  }

  public updateMetadata(partial: Partial<PolicyMetadata>): void {
    this.metadata = {
      ...this.metadata,
      ...partial,
      version: CONTRACT_VERSION,
      policyType: 'q-table',
      observationVersion: CONTRACT_VERSION,
      observationDim: OBSERVATION_DIM,
      actionVersion: CONTRACT_VERSION,
      numActions: NUM_DISCRETE_ACTIONS,
    };
    this.name = this.metadata.name;
  }

  public getMetadata(): PolicyMetadata {
    return { ...this.metadata };
  }

  /**
   * Serialize policy to a versioned JSON-compatible object.
   */
  public toJSON(): SerializedPolicy {
    const qTableObj: Record<string, number[]> = {};
    for (const [key, values] of this.table.entries()) {
      qTableObj[key] = values.map((v) => +v.toFixed(5));
    }
    return {
      version: CONTRACT_VERSION,
      metadata: this.getMetadata(),
      qTable: qTableObj,
    };
  }

  /**
   * Restore a QTablePolicy from a versioned SerializedPolicy payload.
   * Performs strict contract version and observation compatibility checks.
   */
  public static fromJSON(payload: SerializedPolicy): QTablePolicy {
    if (!payload || payload.version !== CONTRACT_VERSION) {
      throw new Error(
        `Incompatible policy version: expected ${CONTRACT_VERSION}, got ${
          (payload as { version?: unknown })?.version
        }`
      );
    }
    if (!payload.metadata || payload.metadata.version !== CONTRACT_VERSION) {
      throw new Error('Incompatible or missing policy metadata version.');
    }
    if (payload.metadata.observationVersion !== CONTRACT_VERSION) {
      throw new Error(
        `Incompatible observationVersion: expected ${CONTRACT_VERSION}, got ${payload.metadata.observationVersion}`
      );
    }
    if (payload.metadata.observationDim !== OBSERVATION_DIM) {
      throw new Error(
        `Incompatible observationDim: expected ${OBSERVATION_DIM}, got ${payload.metadata.observationDim}`
      );
    }
    if (payload.metadata.numActions !== NUM_DISCRETE_ACTIONS) {
      throw new Error(
        `Incompatible numActions: expected ${NUM_DISCRETE_ACTIONS}, got ${payload.metadata.numActions}`
      );
    }

    const policy = new QTablePolicy(payload.metadata.name, payload.metadata);
    if (payload.qTable) {
      for (const [key, arr] of Object.entries(payload.qTable)) {
        if (Array.isArray(arr) && arr.length === NUM_DISCRETE_ACTIONS) {
          policy.table.set(
            key,
            arr.map((n) => Number(n) || 0)
          );
        }
      }
    }
    return policy;
  }
}
