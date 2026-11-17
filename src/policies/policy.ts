import {
  ActionCode,
  CONTRACT_VERSION,
  NUM_DISCRETE_ACTIONS,
  OBSERVATION_DIM,
} from '../contracts/types.ts';
import type {
  ActionInput,
  Observation,
  PolicyMetadata,
  PolicyType,
  SurferAction,
  WorldState,
} from '../contracts/types.ts';

/**
 * Shared Policy interface so gameplay, evaluation, and training treat
 * Human input, Rule-based bots, Tabular Q-learning, and future ONNX neural
 * networks identically.
 */
export interface Policy {
  readonly id: string;
  readonly name: string;
  readonly type: PolicyType;
  selectAction(
    observation: Observation,
    world?: WorldState,
    surferId?: number
  ): ActionInput;
  getMetadata(): PolicyMetadata;
}

/**
 * Human controller policy that translates touch target + PADDLE hold, CATCH button,
 * or desktop keyboard state into a `SurferAction`.
 */
export class HumanPolicy implements Policy {
  public readonly id = 'human-player';
  public readonly name = 'Human Player';
  public readonly type: PolicyType = 'human';

  private targetX: number | undefined = undefined;
  private targetY: number | undefined = undefined;
  private holdingPaddle = false;
  private pendingCatch = false;
  private keyboardAction: ActionCode | null = null;

  public setTouchTarget(x: number, y: number): void {
    this.targetX = x;
    this.targetY = y;
  }

  public getTouchTarget(): { x: number; y: number } | null {
    if (typeof this.targetX === 'number' && typeof this.targetY === 'number') {
      return { x: this.targetX, y: this.targetY };
    }
    return null;
  }

  public setPaddleHeld(held: boolean): void {
    this.holdingPaddle = held;
  }

  public isPaddleHeld(): boolean {
    return this.holdingPaddle;
  }

  public triggerCatch(): void {
    this.pendingCatch = true;
  }

  public setKeyboardAction(action: ActionCode | null): void {
    this.keyboardAction = action;
  }

  public selectAction(_observation: Observation): SurferAction {
    if (this.pendingCatch) {
      this.pendingCatch = false;
      return {
        version: CONTRACT_VERSION,
        action: ActionCode.CATCH,
      };
    }

    if (this.keyboardAction !== null) {
      return {
        version: CONTRACT_VERSION,
        action: this.keyboardAction,
      };
    }

    if (
      this.holdingPaddle &&
      typeof this.targetX === 'number' &&
      typeof this.targetY === 'number'
    ) {
      return {
        version: CONTRACT_VERSION,
        action: ActionCode.WAIT,
        targetX: this.targetX,
        targetY: this.targetY,
        paddle: true,
      };
    }

    return {
      version: CONTRACT_VERSION,
      action: ActionCode.WAIT,
    };
  }

  public getMetadata(): PolicyMetadata {
    return {
      version: CONTRACT_VERSION,
      name: this.name,
      policyType: 'human',
      observationVersion: CONTRACT_VERSION,
      observationDim: OBSERVATION_DIM,
      actionVersion: CONTRACT_VERSION,
      numActions: NUM_DISCRETE_ACTIONS,
      createdAtIso: '2026-01-01T00:00:00.000Z',
      trainingEpisodes: 0,
      paddleSpeedMultiplier: 1.0,
      numCompetitors: 0,
      avgCatchRate: 0,
      avgReward: 0,
    };
  }
}
