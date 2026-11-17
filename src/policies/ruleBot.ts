import {
  ActionCode,
  CONTRACT_VERSION,
  NUM_DISCRETE_ACTIONS,
  OBSERVATION_DIM,
  SurferStatus,
} from '../contracts/types.ts';
import type {
  BotArchetype,
  Observation,
  PolicyMetadata,
  PolicyType,
  WorldState,
} from '../contracts/types.ts';
import type { Policy } from './policy.ts';

const ARCHETYPE_BY_MOD: BotArchetype[] = [
  'local-charger',
  'shoulder-hopper',
  'patient-picker',
  'rookie',
];

/**
 * Deterministic rule-based competitor bot implementing the shared Policy interface.
 * Uses the canonical continuous normalized observation vector (including 16D spatial
 * crowd-density and secondary-peak features) plus deterministic archetype personalities.
 */
export class RuleBotPolicy implements Policy {
  public readonly id: string;
  public readonly name: string;
  public readonly type: PolicyType = 'rule-bot';
  public readonly archetype: BotArchetype;
  private readonly aggression: number;

  constructor(
    botIndex = 1,
    aggression = 0.75,
    archetype: BotArchetype = ARCHETYPE_BY_MOD[(Math.max(0, botIndex - 1)) % ARCHETYPE_BY_MOD.length]
  ) {
    this.id = `rule-bot-${botIndex}-${archetype}`;
    this.archetype = archetype;
    this.name = `Lineup Bot #${botIndex} (${archetype})`;
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
      waveQuality,
      surferStatusNorm,
      crowdDensityLeft,
      crowdDensityCenter,
      crowdDensityRight,
      insidePriorityThreat,
      secondaryPeakRelX,
      secondaryPeakTimeToWave,
    } = observation.named;

    // While riding, carve toward less crowded shoulder or finish near shore
    if (surferStatusNorm >= 0.4 && surferStatusNorm <= 0.6) {
      if (world && surferId !== undefined && world.surferState[surferId] === SurferStatus.RIDING) {
        if (crowdDensityLeft < crowdDensityRight - 0.15) return ActionCode.LEFT;
        if (crowdDensityRight < crowdDensityLeft - 0.15) return ActionCode.RIGHT;
      }
      return ActionCode.WAIT;
    }

    // If wiped out, wait to recover
    if (surferStatusNorm > 0.8) {
      return ActionCode.WAIT;
    }

    // Spread bots deterministically across peaks and shoulders instead of 3 fixed columns
    const spreadIndex = ((surferId * 7 + 3) % 11) - 5;
    const shoulderOffset = spreadIndex * 0.025;

    // Shoulder-hoppers pivot to a secondary wave peak if the primary pocket is jammed
    let desiredRelX = relPeakX + shoulderOffset;
    if (
      this.archetype === 'shoulder-hopper' &&
      (crowdDensityCenter >= 0.5 || insidePriorityThreat >= 0.6) &&
      Math.abs(secondaryPeakRelX) > 0.05 &&
      secondaryPeakTimeToWave >= 0.05 &&
      secondaryPeakTimeToWave <= 0.55
    ) {
      desiredRelX = secondaryPeakRelX + shoulderOffset;
    }

    // 1. Check if the wave is inside the takeoff catch window right now
    const timingSlack = this.archetype === 'rookie' ? 0.02 : 0;
    const minQuality = this.archetype === 'patient-picker' ? 0.58 : 0.0;
    if (
      timeToWave >= -0.05 - timingSlack &&
      timeToWave <= 0.075 + timingSlack &&
      Math.abs(relPeakX) <= 0.22 &&
      relCrossShore >= -0.18 &&
      relCrossShore <= 0.25 &&
      stamina >= 0.1 &&
      waveQuality >= minQuality
    ) {
      // Respect inside priority unless archetype is rookie/aggressive charger
      const yieldsToInside =
        this.archetype !== 'rookie' &&
        (insidePriorityThreat > 0.55 || (competitorPressure > 0.86 && this.aggression < 0.85));
      if (yieldsToInside) {
        return crowdDensityLeft <= crowdDensityRight ? ActionCode.LEFT : ActionCode.OUT;
      }
      return ActionCode.CATCH;
    }

    // 2. If wave is imminent and horizontally aligned, paddle IN to match wave speed
    if (
      timeToWave > 0.075 &&
      timeToWave <= 0.16 &&
      Math.abs(desiredRelX) <= 0.09 &&
      stamina >= 0.18
    ) {
      return ActionCode.IN;
    }

    // 3. Conserve stamina when low and wave is not imminent
    if (stamina < 0.25 && timeToWave > 0.18) {
      return ActionCode.WAIT;
    }

    // 4. Cross-shore positioning
    if (relCrossShore > 0.14) {
      return ActionCode.OUT;
    }
    if (relCrossShore < -0.14 && timeToWave > 0.12) {
      return ActionCode.IN;
    }

    // 5. Horizontal interception toward target peak/open shoulder
    if (desiredRelX > 0.05) {
      return ActionCode.RIGHT;
    }
    if (desiredRelX < -0.05) {
      return ActionCode.LEFT;
    }

    // 6. Fine cross-shore adjustment when aligned
    if (relCrossShore > 0.06) {
      return ActionCode.OUT;
    }
    if (relCrossShore < -0.06) {
      return ActionCode.IN;
    }

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
      numActions: NUM_DISCRETE_ACTIONS,
      createdAtIso: '2026-11-17T00:00:00.000Z',
      trainingEpisodes: 0,
      paddleSpeedMultiplier: 1.0,
      numCompetitors: 0,
      avgCatchRate: 0,
      avgReward: 0,
    };
  }
}
