import {
  CONTRACT_VERSION,
} from '../contracts/types.ts';
import type {
  ActionCode,
  ReferenceScenario,
  ReferenceScenarioStep,
  WorldConfig,
  WorldState,
} from '../contracts/types.ts';
import {
  createWorld,
  getEpisodeMetrics,
  observe,
  step,
} from './world.ts';

/**
 * Record a deterministic ReferenceScenario by running the simulation for `numSteps`
 * with a deterministic action callback.
 */
export function buildReferenceScenario(
  name: string,
  description: string,
  seed: number,
  partialConfig: Partial<WorldConfig>,
  numSteps: number,
  actionSelector: (world: WorldState, stepIdx: number) => ActionCode[]
): ReferenceScenario {
  const world = createWorld(
    {
      ...partialConfig,
      maxSteps: numSteps,
    },
    seed
  );

  const steps: ReferenceScenarioStep[] = [];

  for (let s = 0; s < numSteps; s++) {
    const actions = actionSelector(world, s);
    const stepRes = step(world, actions, world.config.dt);

    const expectedObservations: number[][] = [];
    for (let i = 0; i < world.config.numSurfers; i++) {
      const obs = observe(world, i);
      expectedObservations.push(obs.features.map((f) => +f.toFixed(6)));
    }

    steps.push({
      stepIndex: s + 1,
      actions: [...actions],
      expectedObservations,
      expectedRewards: stepRes.rewards.map((r) => +r.toFixed(4)),
      expectedSurferX: world.surferX.map((x) => +x.toFixed(4)),
      expectedSurferY: world.surferY.map((y) => +y.toFixed(4)),
      expectedSurferStamina: world.surferStamina.map((st) => +st.toFixed(4)),
      expectedSurferState: [...world.surferState],
    });
  }

  return {
    version: CONTRACT_VERSION,
    name,
    description,
    seed,
    config: { ...world.config },
    steps,
    expectedMetrics: getEpisodeMetrics(world),
  };
}

/**
 * Replay a ReferenceScenario against the TypeScript simulation and return any mismatches.
 */
export function replayAndVerifyScenario(
  scenario: ReferenceScenario,
  tolerance = 1e-4
): { passed: boolean; mismatches: string[] } {
  const mismatches: string[] = [];

  if (scenario.version !== CONTRACT_VERSION) {
    mismatches.push(
      `Scenario version mismatch: expected ${CONTRACT_VERSION}, got ${scenario.version}`
    );
    return { passed: false, mismatches };
  }

  const world = createWorld(scenario.config, scenario.seed);

  for (let idx = 0; idx < scenario.steps.length; idx++) {
    const rec = scenario.steps[idx];
    const stepRes = step(world, rec.actions, world.config.dt);

    for (let i = 0; i < world.config.numSurfers; i++) {
      const obs = observe(world, i);
      for (let f = 0; f < obs.features.length; f++) {
        const diff = Math.abs(obs.features[f] - rec.expectedObservations[i][f]);
        if (diff > tolerance) {
          mismatches.push(
            `Step ${rec.stepIndex} surfer ${i} obs[${f}] mismatch: got ${obs.features[f]}, expected ${rec.expectedObservations[i][f]}`
          );
        }
      }

      if (Math.abs(stepRes.rewards[i] - rec.expectedRewards[i]) > tolerance) {
        mismatches.push(
          `Step ${rec.stepIndex} surfer ${i} reward mismatch: got ${stepRes.rewards[i]}, expected ${rec.expectedRewards[i]}`
        );
      }

      if (Math.abs(world.surferX[i] - rec.expectedSurferX[i]) > tolerance) {
        mismatches.push(
          `Step ${rec.stepIndex} surfer ${i} X mismatch: got ${world.surferX[i]}, expected ${rec.expectedSurferX[i]}`
        );
      }

      if (Math.abs(world.surferY[i] - rec.expectedSurferY[i]) > tolerance) {
        mismatches.push(
          `Step ${rec.stepIndex} surfer ${i} Y mismatch: got ${world.surferY[i]}, expected ${rec.expectedSurferY[i]}`
        );
      }

      if (Math.abs(world.surferStamina[i] - rec.expectedSurferStamina[i]) > tolerance) {
        mismatches.push(
          `Step ${rec.stepIndex} surfer ${i} stamina mismatch: got ${world.surferStamina[i]}, expected ${rec.expectedSurferStamina[i]}`
        );
      }

      if (world.surferState[i] !== rec.expectedSurferState[i]) {
        mismatches.push(
          `Step ${rec.stepIndex} surfer ${i} state mismatch: got ${world.surferState[i]}, expected ${rec.expectedSurferState[i]}`
        );
      }
    }
  }

  const finalMetrics = getEpisodeMetrics(world);
  if (
    JSON.stringify(finalMetrics.surferCatches) !==
    JSON.stringify(scenario.expectedMetrics.surferCatches)
  ) {
    mismatches.push(
      `Final catches mismatch: got ${JSON.stringify(
        finalMetrics.surferCatches
      )}, expected ${JSON.stringify(scenario.expectedMetrics.surferCatches)}`
    );
  }

  return {
    passed: mismatches.length === 0,
    mismatches,
  };
}
