import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ActionCode } from '../src/contracts/types.ts';
import type { BotArchetype, ReferenceScenario } from '../src/contracts/types.ts';
import { RuleBotPolicy } from '../src/policies/ruleBot.ts';
import { buildReferenceScenario } from '../src/sim/scenarios.ts';
import { LINEUP_PRESETS, observe } from '../src/sim/world.ts';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outPath = resolve(__dirname, '../tests/fixtures/reference-scenarios.json');

const ARCHETYPES: BotArchetype[] = [
  'local-charger',
  'shoulder-hopper',
  'patient-picker',
  'rookie',
];
const bots = Array.from({ length: 24 }, (_, idx) => {
  return new RuleBotPolicy(idx + 1, 0.78, ARCHETYPES[idx % ARCHETYPES.length]);
});

const scenarios: ReferenceScenario[] = [
  buildReferenceScenario(
    'solo-wave-catch',
    'Single surfer intercepts peak and catches incoming simplified wave',
    101,
    {
      width: 160,
      numSurfers: 1,
      simplifiedWaves: true,
      paddleSpeedMultiplier: 1.0,
    },
    45,
    (world) => {
      const obs = observe(world, 0);
      return [bots[0].selectAction(obs, world, 0)];
    }
  ),
  buildReferenceScenario(
    'crowded-lineup-three-surfers',
    'Three surfers competing for wave priority in standard wave conditions',
    202,
    {
      width: 160,
      numSurfers: 3,
      simplifiedWaves: false,
      paddleSpeedMultiplier: 1.1,
    },
    50,
    (world) => {
      return [0, 1, 2].map((i) => bots[i].selectAction(observe(world, i), world, i));
    }
  ),
  buildReferenceScenario(
    'stamina-depletion-and-recovery',
    'Surfer paddles offshore for 25 steps depleting stamina, then rests for 15 steps',
    303,
    {
      width: 160,
      numSurfers: 1,
      simplifiedWaves: true,
      paddleSpeedMultiplier: 1.2,
    },
    40,
    (_world, stepIdx) => {
      return [stepIdx < 25 ? ActionCode.OUT : ActionCode.WAIT];
    }
  ),
  buildReferenceScenario(
    'crowded-lineup-twenty-four-surfers',
    'Twenty-four archetype bots competing across a 320x240 multi-peak lineup',
    404,
    {
      ...LINEUP_PRESETS['crowded-24'],
      simplifiedWaves: false,
      paddleSpeedMultiplier: 1.0,
    },
    35,
    (world) => {
      const actions: ActionCode[] = [];
      for (let i = 0; i < world.config.numSurfers; i++) {
        actions.push(bots[i].selectAction(observe(world, i), world, i));
      }
      return actions;
    }
  ),
];

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(scenarios, null, 2) + '\n', 'utf-8');
console.log(`Generated ${scenarios.length} reference scenarios at ${outPath}`);
