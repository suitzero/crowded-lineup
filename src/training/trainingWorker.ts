import { CONTRACT_VERSION } from '../contracts/types.ts';
import type { SerializedPolicy } from '../contracts/types.ts';
import { QTablePolicy } from '../policies/qTable.ts';
import { QTrainer } from './trainer.ts';
import type { TrainingParams, TrainingProgressPoint } from './trainer.ts';

export type WorkerInboundMessage =
  | {
      version: typeof CONTRACT_VERSION;
      type: 'start';
      params: TrainingParams;
      existingPolicy?: SerializedPolicy;
    }
  | {
      version: typeof CONTRACT_VERSION;
      type: 'pause';
    }
  | {
      version: typeof CONTRACT_VERSION;
      type: 'resume';
    }
  | {
      version: typeof CONTRACT_VERSION;
      type: 'cancel';
    };

export type WorkerOutboundMessage =
  | {
      version: typeof CONTRACT_VERSION;
      type: 'progress';
      point: TrainingProgressPoint;
      totalEpisodes: number;
      policy: SerializedPolicy;
    }
  | {
      version: typeof CONTRACT_VERSION;
      type: 'paused';
      point: TrainingProgressPoint;
      policy: SerializedPolicy;
    }
  | {
      version: typeof CONTRACT_VERSION;
      type: 'cancelled';
    }
  | {
      version: typeof CONTRACT_VERSION;
      type: 'completed';
      point: TrainingProgressPoint;
      policy: SerializedPolicy;
    };

let trainer: QTrainer | null = null;
let paused = false;
let timerId: ReturnType<typeof setTimeout> | null = null;

function clearLoop(): void {
  if (timerId !== null) {
    clearTimeout(timerId);
    timerId = null;
  }
}

function scheduleNextChunk(): void {
  clearLoop();
  if (!trainer || paused) return;

  timerId = setTimeout(() => {
    if (!trainer || paused) return;

    const total = trainer.params.totalEpisodes;
    // Choose a responsive chunk size (20 to 50 episodes per tick) so messages are handled promptly
    const batchSize = total <= 1000 ? 25 : total <= 5000 ? 50 : 100;
    const point = trainer.stepBatch(batchSize);

    if (trainer.isDone()) {
      const doneMsg: WorkerOutboundMessage = {
        version: CONTRACT_VERSION,
        type: 'completed',
        point,
        policy: trainer.policy.toJSON(),
      };
      self.postMessage(doneMsg);
      trainer = null;
      return;
    }

    const progressMsg: WorkerOutboundMessage = {
      version: CONTRACT_VERSION,
      type: 'progress',
      point,
      totalEpisodes: total,
      policy: trainer.policy.toJSON(),
    };
    self.postMessage(progressMsg);
    scheduleNextChunk();
  }, 0);
}

self.onmessage = (event: MessageEvent<WorkerInboundMessage>) => {
  const msg = event.data;
  if (!msg || msg.version !== CONTRACT_VERSION) return;

  switch (msg.type) {
    case 'start': {
      clearLoop();
      paused = false;
      const basePolicy = msg.existingPolicy
        ? QTablePolicy.fromJSON(msg.existingPolicy)
        : undefined;
      trainer = new QTrainer(msg.params, basePolicy);
      scheduleNextChunk();
      break;
    }
    case 'pause': {
      paused = true;
      clearLoop();
      if (trainer) {
        const lastPoint =
          trainer.history[trainer.history.length - 1] ?? {
            episode: trainer.currentEpisode,
            rollingCatchRate: trainer.getRollingCatchRate(),
            rollingReward: trainer.getRollingReward(),
            epsilon: 0.2,
            stateCount: trainer.policy.getStateCount(),
          };
        const pausedMsg: WorkerOutboundMessage = {
          version: CONTRACT_VERSION,
          type: 'paused',
          point: lastPoint,
          policy: trainer.policy.toJSON(),
        };
        self.postMessage(pausedMsg);
      }
      break;
    }
    case 'resume': {
      if (trainer && paused) {
        paused = false;
        scheduleNextChunk();
      }
      break;
    }
    case 'cancel': {
      paused = false;
      clearLoop();
      trainer = null;
      const cancelMsg: WorkerOutboundMessage = {
        version: CONTRACT_VERSION,
        type: 'cancelled',
      };
      self.postMessage(cancelMsg);
      break;
    }
  }
};
