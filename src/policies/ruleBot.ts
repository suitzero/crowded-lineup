import {
  ActionCode,
  CONTRACT_VERSION,
  OBSERVATION_DIM,
  SurferStatus,
} from '../contracts/types.ts';
import type {
  Observation,
  PolicyMetadata,
  PolicyType,
  WorldState,
} from '../contracts/types.ts';
import type { Policy } from './policy.ts';

/**
 * Deterministic rule-based competitor bot implementing the shared Policy interface.
 * Uses the canonical continuous normalized observation vector (plus surferId lane offset
 * to spread competitors across peak & shoulders).
 */
export class RuleBotPolicy implements Policy {
  public readonly id: string;
  public readonly name: string;
  public readonly type: PolicyType = 'rule-bot';
  private readonly aggression: number;

  constructor(botIndex = 1, aggression = 0.75) {
    this.id = `rule-bot-${botIndex}`;
    this.name = `Lineup Bot #${botIndex}`;
    this.aggression = aggression;
  }

  public selectAction(
    observation: Observation,
    world?: WorldState,
    surferId = observation.surferId
  ): ActionCode {
    const {
      relPeakX,
      timeToWave,
      relCrossShore,
      stamina,
      competitorPressure,
      surferStatusNorm,
    } = observation.named;

    // While riding, stay on the wave face until near shore
    if (surferStatusNorm >= 0.4 && surferStatusNorm <= 0.6) {
      if (world && surferId !== undefined && world.surferState[surferId] === SurferStatus.RIDING) {
        return ActionCode.WAIT;
      }
      return ActionCode.WAIT;
    }

    // If wiped out, wait to recover
    if (surferStatusNorm > 0.8) {
      return ActionCode.WAIT;
    }

    // Deterministic slight lateral offset per bot so multiple bots don't stack on top of each other
    const shoulderOffset = ((surferId % 3) - 1) * 0.045;
    const adjustedPeakErr = relPeakX + shoulderOffset;

    // 1. Check if the wave is inside the takeoff catch window right now
    if (
      timeToWave >= -0.05 &&
      timeToWave <= 0.075 &&
      Math.abs(relPeakX) <= 0.22 &&
      relCrossShore >= -0.18 &&
      relCrossShore <= 0.25 &&
      stamina >= 0.1
    ) {
      // Yield if heavy inside competitor pressure and bot has moderate aggression
      if (competitorPressure > 0.88 && this.aggression < 0.85) {
        return ActionCode.OUT;
      }
      return ActionCode.CATCH;
    }

    // 2. If wave is very close (imminent) and we are horizontally aligned, paddle IN to match speed
    if (
      timeToWave > 0.075 &&
      timeToWave <= 0.16 &&
      Math.abs(adjustedPeakErr) <= 0.09 &&
      stamina >= 0.18
    ) {
      return ActionCode.IN;
    }

    // 3. Conserve stamina when low and wave is not imminent
    if (stamina < 0.25 && timeToWave > 0.18) {
      return ActionCode.WAIT;
    }

    // 4. Cross-shore positioning: return to the takeoff zone if pushed too far inside or outside
    if (relCrossShore > 0.14) {
      return ActionCode.OUT;
    }
    if (relCrossShore < -0.14 && timeToWave > 0.12) {
      return ActionCode.IN;
    }

    // 5. Horizontal interception toward the incoming wave peak
    if (adjustedPeakErr > 0.055) {
      return ActionCode.RIGHT;
    }
    if (adjustedPeakErr < -0.055) {
      return ActionCode.LEFT;
    }

    // 6. Fine cross-shore adjustment when aligned with peak
    if (relCrossShore > 0.06) {
      return ActionCode.OUT;
    }
    if (relCrossShore < -0.06) {
      return ActionCode.IN;
    }

    // Rest to keep stamina topped up for takeoff
    return ActionCode.WAIT;
  }

  public getMetadata(): PolicyMetadata {
    return {
      version: CONTRACT_VERSION,
      name: this.name,
      policyType: 'rule-bot',
      observationVersion: CONTRACT_VERSION,
      observationDim: OBSERVATION_DIM,
      actionVersion: CONTRACT_VERSION,
      numActions: 6,
      createdAtIso: '2026-01-01T00:00:00.000Z',
      trainingEpisodes: 0,
      paddleSpeedMultiplier: 1.0,
      numCompetitors: 0,
      avgCatchRate: 0,
      avgReward: 0,
    };
  }
}
