import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ActionCode } from '../src/contracts/types.ts';
import type { ReferenceScenario } from '../src/contracts/types.ts';
import { RuleBotPolicy } from '../src/policies/ruleBot.ts';
import { buildReferenceScenario } from '../src/sim/scenarios.ts';
import { observe } from '../src/sim/world.ts';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = resolve(__dirname, '../tests/fixtures/reference-scenarios.json');

const bot1 = new RuleBotPolicy(1, 0.85);
const bot2 = new RuleBotPolicy(2, 0.85);
const bot3 = new RuleBotPolicy(3, 0.95);

const scenarios: ReferenceScenario[] = [
  buildReferenceScenario(
    'solo-wave-catch',
    'Single surfer intercepts peak and catches incoming simplified wave',
    101,
    {
      numSurfers: 1,
      simplifiedWaves: true,
      paddleSpeedMultiplier: 1.0,
    },
    45,
    (world) => {
      const obs = observe(world, 0);
      return [bot1.selectAction(obs, world, 0)];
    }
  ),
  buildReferenceScenario(
    'crowded-lineup-three-surfers',
    'Three surfers competing for wave priority in standard wave conditions',
    202,
    {
      numSurfers: 3,
      simplifiedWaves: false,
      paddleSpeedMultiplier: 1.1,
    },
    50,
    (world) => {
      return [
        bot1.selectAction(observe(world, 0), world, 0),
        bot2.selectAction(observe(world, 1), world, 1),
        bot3.selectAction(observe(world, 2), world, 2),
      ];
    }
  ),
  buildReferenceScenario(
    'stamina-depletion-and-recovery',
    'Surfer paddles offshore for 25 steps depleting stamina, then rests for 15 steps',
    303,
    {
      numSurfers: 1,
      simplifiedWaves: true,
      paddleSpeedMultiplier: 1.2,
    },
    40,
    (_world, stepIdx) => {
      return [stepIdx < 25 ? ActionCode.OUT : ActionCode.WAIT];
    }
  ),
];

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(scenarios, null, 2) + '\n', 'utf-8');
console.log(`Generated ${scenarios.length} reference scenarios at ${outPath}`);
