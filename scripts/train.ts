import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { serializePolicyToJsonString } from '../src/policies/storage.ts';
import { QTrainer } from '../src/training/trainer.ts';

function parseArg(args: string[], flag: string, fallback: string): string {
  const idx = args.indexOf(flag);
  if (idx !== -1 && idx + 1 < args.length) {
    return args[idx + 1];
  }
  const prefixed = args.find((a) => a.startsWith(`${flag}=`));
  if (prefixed) {
    return prefixed.split('=')[1];
  }
  return fallback;
}

const argv = process.argv.slice(2);
const totalEpisodes = Math.max(10, parseInt(parseArg(argv, '--episodes', '300'), 10));
const numCompetitors = Math.max(0, parseInt(parseArg(argv, '--competitors', '2'), 10));
const paddleSpeedMultiplier = parseFloat(parseArg(argv, '--speed', '1.0'));
const seed = parseInt(parseArg(argv, '--seed', '2026'), 10);
const outFile = parseArg(argv, '--out', '');

console.log(
  `[Crowded Lineup] Starting headless Q-learning training: episodes=${totalEpisodes}, competitors=${numCompetitors}, speed=${paddleSpeedMultiplier}x, seed=${seed}`
);

const trainer = new QTrainer({
  totalEpisodes,
  numCompetitors,
  paddleSpeedMultiplier,
  seed,
});

const reportInterval = Math.max(25, Math.floor(totalEpisodes / 6));
const startMs = performance.now();

while (!trainer.isDone()) {
  const pt = trainer.stepBatch(reportInterval);
  console.log(
    `  Episode ${String(pt.episode).padStart(5, ' ')} / ${totalEpisodes} | ` +
      `Catch Rate: ${(pt.rollingCatchRate * 100).toFixed(1)}% | ` +
      `Avg Reward: ${pt.rollingReward.toFixed(2)} | ` +
      `Epsilon: ${pt.epsilon.toFixed(3)} | ` +
      `Q-States: ${pt.stateCount}`
  );
}

const elapsedSec = ((performance.now() - startMs) / 1000).toFixed(2);
const meta = trainer.policy.getMetadata();

console.log(
  `[Crowded Lineup] Training complete in ${elapsedSec}s! Final Rolling Catch Rate: ${(
    meta.avgCatchRate * 100
  ).toFixed(1)}%, Avg Reward: ${meta.avgReward.toFixed(
    2
  )}, Learned States: ${trainer.policy.getStateCount()}`
);

if (outFile) {
  const absOut = resolve(process.cwd(), outFile);
  writeFileSync(absOut, serializePolicyToJsonString(trainer.policy) + '\n', 'utf-8');
  console.log(`[Crowded Lineup] Saved versioned policy JSON to ${absOut}`);
}
