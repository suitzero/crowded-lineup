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
import type { Policy } from './policy.ts';

/**
 * Reserved interface for future ONNX neural policy inference (`onnxruntime-web`).
 * Per SPEC Part 2 (#6, #7), `onnxruntime-web` is intentionally NOT added as a dependency
 * in the MVP release. This adapter validates contract & observation compatibility and
 * exposes the exact synchronous/warmed-cache inference hook for future GPU-trained PPO models.
 */
export class NeuralOnnxPolicyPlaceholder implements Policy {
  public readonly id: string;
  public readonly name: string;
  public readonly type: PolicyType = 'neural-onnx';
  private readonly metadata: PolicyMetadata;
  private isLoaded = false;

  constructor(name = 'Future ONNX PPO Policy', metadata?: Partial<PolicyMetadata>) {
    this.id = 'neural-onnx-policy';
    this.name = name;
    this.metadata = {
      version: CONTRACT_VERSION,
      name,
      policyType: 'neural-onnx',
      observationVersion: CONTRACT_VERSION,
      observationDim: OBSERVATION_DIM,
      actionVersion: CONTRACT_VERSION,
      numActions: NUM_DISCRETE_ACTIONS,
      createdAtIso: '2026-11-14T00:00:00.000Z',
      trainingEpisodes: 0,
      paddleSpeedMultiplier: 1.0,
      numCompetitors: 3,
      avgCatchRate: 0,
      avgReward: 0,
      ...metadata,
    };
  }

  /**
   * Validate an exported ONNX model metadata envelope before loading.
   */
  public validateCompatibility(envelope: SerializedPolicy): void {
    if (envelope.version !== CONTRACT_VERSION) {
      throw new Error(
        `ONNX policy version mismatch: expected ${CONTRACT_VERSION}, got ${envelope.version}`
      );
    }
    if (envelope.metadata.observationDim !== OBSERVATION_DIM) {
      throw new Error(
        `ONNX observationDim mismatch: expected ${OBSERVATION_DIM}, got ${envelope.metadata.observationDim}`
      );
    }
    if (envelope.metadata.numActions !== NUM_DISCRETE_ACTIONS) {
      throw new Error(
        `ONNX numActions mismatch: expected ${NUM_DISCRETE_ACTIONS}, got ${envelope.metadata.numActions}`
      );
    }
  }

  public async loadOnnxModel(envelope: SerializedPolicy): Promise<void> {
    this.validateCompatibility(envelope);
    throw new Error(
      'ONNX runtime (onnxruntime-web) is reserved for Phase 2 GPU RL integration. See docs/gpu-rl-roadmap.md.'
    );
  }

  public selectAction(observation: Observation): ActionCode {
    if (observation.features.length !== OBSERVATION_DIM) {
      throw new Error(`Expected ${OBSERVATION_DIM} continuous features`);
    }
    if (!this.isLoaded) {
      // Safe deterministic fallback if invoked without a loaded ONNX session
      return ActionCode.WAIT;
    }
    return ActionCode.WAIT;
  }

  public getMetadata(): PolicyMetadata {
    return { ...this.metadata };
  }
}
