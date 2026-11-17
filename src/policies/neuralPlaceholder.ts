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

function decodeBase64ToBytes(b64: string): Uint8Array {
  if (typeof atob === 'function') {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      out[i] = bin.charCodeAt(i);
    }
    return out;
  }
  return Uint8Array.from(Buffer.from(b64, 'base64'));
}

interface OrtSessionLike {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: ArrayLike<number> }>>;
}

interface OrtModuleLike {
  InferenceSession: {
    create(buffer: Uint8Array): Promise<OrtSessionLike>;
  };
  Tensor: new (type: string, data: Float32Array, dims: number[]) => unknown;
}

/**
 * Real ONNX neural policy adapter backed by `onnxruntime-web`.
 * Validates v2 contracts (`OBSERVATION_DIM = 16`, `NUM_DISCRETE_ACTIONS = 6`)
 * and runs PPO-exported models exported by `packages/python_rl/export_onnx.py`.
 */
export class NeuralOnnxPolicyPlaceholder implements Policy {
  public readonly id: string;
  public name: string;
  public readonly type: PolicyType = 'neural-onnx';
  private metadata: PolicyMetadata;
  private isLoaded = false;
  private session: OrtSessionLike | null = null;
  private ortMod: OrtModuleLike | null = null;
  private lastAction: ActionCode = ActionCode.WAIT;
  private inferInFlight = false;

  constructor(name = 'PPO ONNX Neural Policy', metadata?: Partial<PolicyMetadata>) {
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
      createdAtIso: '2026-11-17T00:00:00.000Z',
      trainingEpisodes: 0,
      paddleSpeedMultiplier: 1.0,
      numCompetitors: 15,
      avgCatchRate: 0,
      avgReward: 0,
      ...metadata,
    };
  }

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
    if (!envelope.onnxModelBase64) {
      throw new Error('Missing onnxModelBase64 payload in SerializedPolicy');
    }
    const ort = (await import('onnxruntime-web')) as unknown as OrtModuleLike;
    const bytes = decodeBase64ToBytes(envelope.onnxModelBase64);
    this.session = await ort.InferenceSession.create(bytes);
    this.ortMod = ort;
    this.metadata = { ...envelope.metadata };
    this.name = envelope.metadata.name;
    this.isLoaded = true;
  }

  public isSessionLoaded(): boolean {
    return this.isLoaded && this.session !== null;
  }

  public async predictActionAsync(observation: Observation): Promise<ActionCode> {
    if (observation.features.length !== OBSERVATION_DIM) {
      throw new Error(`Expected ${OBSERVATION_DIM} continuous features`);
    }
    if (!this.session || !this.ortMod) {
      return ActionCode.WAIT;
    }
    const input = new this.ortMod.Tensor(
      'float32',
      Float32Array.from(observation.features),
      [1, OBSERVATION_DIM]
    );
    const outputs = await this.session.run({ obs: input });
    const logitsOut = outputs.logits ?? Object.values(outputs)[0];
    const data = logitsOut?.data;
    if (!data || data.length < NUM_DISCRETE_ACTIONS) {
      return ActionCode.WAIT;
    }
    let bestAct: ActionCode = ActionCode.WAIT;
    let bestScore = -Infinity;
    for (let a = 0; a < NUM_DISCRETE_ACTIONS; a++) {
      if (data[a] > bestScore) {
        bestScore = data[a];
        bestAct = a as ActionCode;
      }
    }
    this.lastAction = bestAct;
    return bestAct;
  }

  public selectAction(observation: Observation): ActionCode {
    if (observation.features.length !== OBSERVATION_DIM) {
      throw new Error(`Expected ${OBSERVATION_DIM} continuous features`);
    }
    if (!this.isLoaded || !this.session) {
      return ActionCode.WAIT;
    }
    if (!this.inferInFlight) {
      this.inferInFlight = true;
      void this.predictActionAsync(observation).finally(() => {
        this.inferInFlight = false;
      });
    }
    return this.lastAction;
  }

  public getMetadata(): PolicyMetadata {
    return { ...this.metadata };
  }
}
