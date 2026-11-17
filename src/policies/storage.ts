import type { SerializedPolicy } from '../contracts/types.ts';
import { NeuralOnnxPolicyPlaceholder } from './neuralPlaceholder.ts';
import type { Policy } from './policy.ts';
import { QTablePolicy } from './qTable.ts';

const STORAGE_KEY = 'crowded_lineup_saved_policy_v2';

export function savePolicyToLocalStorage(policy: QTablePolicy): void {
  if (typeof localStorage === 'undefined') return;
  const serialized = JSON.stringify(policy.toJSON());
  localStorage.setItem(STORAGE_KEY, serialized);
}

export function loadPolicyFromLocalStorage(): QTablePolicy | null {
  if (typeof localStorage === 'undefined') return null;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  const parsed = JSON.parse(raw) as SerializedPolicy;
  return QTablePolicy.fromJSON(parsed);
}

export function hasSavedPolicyInLocalStorage(): boolean {
  if (typeof localStorage === 'undefined') return false;
  return localStorage.getItem(STORAGE_KEY) !== null;
}

export function serializePolicyToJsonString(policy: QTablePolicy): string {
  return JSON.stringify(policy.toJSON(), null, 2);
}

export function deserializePolicyFromJsonString(jsonStr: string): QTablePolicy {
  const parsed = JSON.parse(jsonStr) as SerializedPolicy;
  return QTablePolicy.fromJSON(parsed);
}

export async function deserializeAnyPolicyFromJsonString(
  jsonStr: string
): Promise<Policy> {
  const parsed = JSON.parse(jsonStr) as SerializedPolicy;
  if (parsed.metadata?.policyType === 'neural-onnx' && parsed.onnxModelBase64) {
    const onnxPolicy = new NeuralOnnxPolicyPlaceholder(parsed.metadata.name);
    await onnxPolicy.loadOnnxModel(parsed);
    return onnxPolicy;
  }
  return QTablePolicy.fromJSON(parsed);
}
