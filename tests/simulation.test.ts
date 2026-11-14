import { describe, expect, it } from 'vitest';
import referenceScenariosData from './fixtures/reference-scenarios.json';
import {
  ActionCode,
  CONTRACT_VERSION,
  OBSERVATION_DIM,
  SurferStatus,
} from '../src/contracts/types.ts';
import type { ReferenceScenario, SerializedPolicy } from '../src/contracts/types.ts';
import { NeuralOnnxPolicyPlaceholder } from '../src/policies/neuralPlaceholder.ts';
import { QTablePolicy, discretizeObservation } from '../src/policies/qTable.ts';
import { RuleBotPolicy } from '../src/policies/ruleBot.ts';
import {
  deserializePolicyFromJsonString,
  serializePolicyToJsonString,
} from '../src/policies/storage.ts';
import { SeededRng } from '../src/sim/prng.ts';
import { replayAndVerifyScenario } from '../src/sim/scenarios.ts';
import {
  cloneWorld,
  createWorld,
  getEpisodeMetrics,
  isEpisodeDone,
  observe,
  spawnWave,
  step,
} from '../src/sim/world.ts';
import { QTrainer } from '../src/training/trainer.ts';

describe('Crowded Lineup — Deterministic Simulation & RL Suite', () => {
  it('1. Seed reproducibility: identical seeds & actions produce identical trajectories and metrics', () => {
    const w1 = createWorld({ numSurfers: 3, maxSteps: 100 }, 1337);
    const w2 = createWorld({ numSurfers: 3, maxSteps: 100 }, 1337);
    const wDiff = createWorld({ numSurfers: 3, maxSteps: 100 }, 9999);

    const bot = new RuleBotPolicy(1);

    for (let s = 0; s < 100; s++) {
      const a1 = [
        bot.selectAction(observe(w1, 0), w1, 0),
        bot.selectAction(observe(w1, 1), w1, 1),
        bot.selectAction(observe(w1, 2), w1, 2),
      ];
      const a2 = [
        bot.selectAction(observe(w2, 0), w2, 0),
        bot.selectAction(observe(w2, 1), w2, 1),
        bot.selectAction(observe(w2, 2), w2, 2),
      ];
      const r1 = step(w1, a1);
      const r2 = step(w2, a2);
      step(wDiff, a1);

      expect(r1.rewards).toEqual(r2.rewards);
      expect(w1.surferX).toEqual(w2.surferX);
      expect(w1.surferY).toEqual(w2.surferY);
      expect(w1.rngState).toEqual(w2.rngState);
    }

    expect(getEpisodeMetrics(w1)).toEqual(getEpisodeMetrics(w2));
    expect(w1.surferX).not.toEqual(wDiff.surferX);
  });

  it('2. Wave propagation: crest moves toward shore (+y), breaking begins at waveBreakStartY and spreads laterally', () => {
    const world = createWorld({ numSurfers: 1, simplifiedWaves: true }, 42);
    // Clear existing waves and spawn one controlled wave at y = 60
    world.waveActive.fill(0);
    const slot = spawnWave(world, 60);
    expect(slot).toBeGreaterThanOrEqual(0);

    world.waveBreakStartY[slot] = 75;
    world.waveSpeed[slot] = 20;
    world.waveBreakSpreadRate[slot] = 0.8;

    const initialY = world.waveY[slot];
    expect(world.waveBreakWidth[slot]).toBe(0);

    // Step 5 times (dt = 0.1 -> +10 units -> y = 70, still unbroken)
    for (let i = 0; i < 5; i++) {
      step(world, [ActionCode.WAIT]);
    }
    expect(world.waveY[slot]).toBeGreaterThan(initialY);
    expect(world.waveY[slot]).toBeCloseTo(70, 3);
    expect(world.waveBreakWidth[slot]).toBe(0);

    // Step 10 more times (-> y = 90 > waveBreakStartY = 75, breaking spreads laterally)
    for (let i = 0; i < 10; i++) {
      step(world, [ActionCode.WAIT]);
    }
    expect(world.waveY[slot]).toBeCloseTo(90, 3);
    const widthAt90 = world.waveBreakWidth[slot];
    expect(widthAt90).toBeGreaterThan(10);

    // Step 5 more times -> breaking width increases further
    for (let i = 0; i < 5; i++) {
      step(world, [ActionCode.WAIT]);
    }
    expect(world.waveBreakWidth[slot]).toBeGreaterThan(widthAt90);
  });

  it('3. Whitewater lifetime: breaking waves spawn whitewater trails that fade and expire', () => {
    const world = createWorld({ numSurfers: 1, simplifiedWaves: true }, 77);
    world.waveActive.fill(0);
    const slot = spawnWave(world, 80);
    world.waveBreakStartY[slot] = 70;

    // Step until whitewater is emitted
    for (let i = 0; i < 6; i++) {
      step(world, [ActionCode.WAIT]);
    }

    const activeIdx = world.wwActive.findIndex((a) => a === 1);
    expect(activeIdx).toBeGreaterThanOrEqual(0);
    const initialIntensity = world.wwIntensity[activeIdx];
    expect(initialIntensity).toBeGreaterThan(0.8);

    // Prevent further wave emissions so we track this specific whitewater slot
    world.waveActive.fill(0);
    world.nextWaveSpawnTime = 9999;

    // Advance 15 steps (1.5s) -> intensity fades
    for (let i = 0; i < 15; i++) {
      step(world, [ActionCode.WAIT]);
    }
    expect(world.wwActive[activeIdx]).toBe(1);
    expect(world.wwIntensity[activeIdx]).toBeLessThan(initialIntensity);
    expect(world.wwIntensity[activeIdx]).toBeGreaterThan(0);

    // Advance another 25 steps (2.5s, total > 3.2s maxAge) -> whitewater deactivates
    for (let i = 0; i < 25; i++) {
      step(world, [ActionCode.WAIT]);
    }
    expect(world.wwActive[activeIdx]).toBe(0);
    expect(world.wwIntensity[activeIdx]).toBe(0);
  });

  it('4. Surfer movement: directional actions and touch targets accelerate surfer as expected', () => {
    const world = createWorld({ numSurfers: 1, paddleSpeedMultiplier: 1.0 }, 55);
    world.surferX[0] = 80;
    world.surferY[0] = 100;

    step(world, [ActionCode.LEFT]);
    expect(world.surferVx[0]).toBeLessThan(0);
    expect(world.surferX[0]).toBeLessThan(80);

    const xAfterLeft = world.surferX[0];
    step(world, [ActionCode.RIGHT]);
    step(world, [ActionCode.RIGHT]);
    step(world, [ActionCode.RIGHT]);
    expect(world.surferVx[0]).toBeGreaterThan(0);
    expect(world.surferX[0]).toBeGreaterThan(xAfterLeft);

    const yBeforeOut = world.surferY[0];
    step(world, [ActionCode.OUT]);
    step(world, [ActionCode.OUT]);
    expect(world.surferY[0]).toBeLessThan(yBeforeOut);

    // Continuous touch target movement
    const xBeforeTouch = world.surferX[0];
    const yBeforeTouch = world.surferY[0];
    for (let i = 0; i < 5; i++) {
      step(world, [
        {
          version: CONTRACT_VERSION,
          action: ActionCode.WAIT,
          targetX: 140,
          targetY: 160,
          paddle: true,
        },
      ]);
    }
    expect(world.surferX[0]).toBeGreaterThan(xBeforeTouch);
    expect(world.surferY[0]).toBeGreaterThan(yBeforeTouch);
  });

  it('5. Stamina constraints: paddling drains stamina, low stamina reduces max speed, waiting recovers stamina', () => {
    const world = createWorld({ numSurfers: 1, paddleSpeedMultiplier: 1.0 }, 66);
    world.surferX[0] = 80;
    world.surferY[0] = 180;

    // Paddle OUT continuously to measure full-stamina speed and drain stamina
    for (let i = 0; i < 10; i++) {
      step(world, [ActionCode.OUT]);
    }
    const fullStaminaSpeed = Math.abs(world.surferVy[0]);
    expect(world.surferStamina[0]).toBeLessThan(0.92);

    // Force stamina near zero and step to verify speed constraint
    world.surferStamina[0] = 0.02;
    for (let i = 0; i < 8; i++) {
      step(world, [ActionCode.OUT]);
    }
    const exhaustedSpeed = Math.abs(world.surferVy[0]);
    expect(exhaustedSpeed).toBeLessThan(fullStaminaSpeed * 0.55);

    // Wait to recover stamina
    const lowStam = world.surferStamina[0];
    for (let i = 0; i < 20; i++) {
      step(world, [ActionCode.WAIT]);
    }
    expect(world.surferStamina[0]).toBeGreaterThan(lowStam + 0.25);
  });

  it('6. Catch success and failure: valid position/timing catches wave; bad position or exhausted stamina fails', () => {
    // Valid catch setup
    const world = createWorld({ numSurfers: 1, simplifiedWaves: true }, 88);
    world.waveActive.fill(0);
    const w = spawnWave(world, 80);
    world.wavePeakX[w] = 80;
    world.waveBreakStartY[w] = 75;
    world.waveBreakWidth[w] = 12;
    world.waveSpeed[w] = 20;

    // Place surfer right at the peak crest with good stamina
    world.surferX[0] = 80;
    world.surferY[0] = 82;
    world.surferVy[0] = 5;
    world.surferStamina[0] = 0.9;

    const resSuccess = step(world, [ActionCode.CATCH]);
    expect(world.surferState[0]).toBe(SurferStatus.RIDING);
    expect(world.surferCatches[0]).toBe(1);
    expect(resSuccess.breakdowns[0].catchReward).toBeGreaterThan(10);
    expect(resSuccess.rewards[0]).toBeGreaterThan(15);

    // Failure case 1: surfer far away from wave crest
    const worldFailDist = createWorld({ numSurfers: 1, simplifiedWaves: true }, 89);
    worldFailDist.waveActive.fill(0);
    const w2 = spawnWave(worldFailDist, 40);
    worldFailDist.wavePeakX[w2] = 80;
    worldFailDist.surferX[0] = 80;
    worldFailDist.surferY[0] = 120;
    const resFail1 = step(worldFailDist, [ActionCode.CATCH]);
    expect(worldFailDist.surferState[0]).toBe(SurferStatus.FREE);
    expect(worldFailDist.surferCatches[0]).toBe(0);
    expect(resFail1.breakdowns[0].catchReward).toBe(0);

    // Failure case 2: surfer at crest but zero stamina
    const worldFailStam = createWorld({ numSurfers: 1, simplifiedWaves: true }, 90);
    worldFailStam.waveActive.fill(0);
    const w3 = spawnWave(worldFailStam, 80);
    worldFailStam.wavePeakX[w3] = 80;
    worldFailStam.waveBreakStartY[w3] = 75;
    worldFailStam.surferX[0] = 80;
    worldFailStam.surferY[0] = 82;
    worldFailStam.surferStamina[0] = 0.01;
    step(worldFailStam, [ActionCode.CATCH]);
    expect(worldFailStam.surferCatches[0]).toBe(0);
  });

  it('7. Priority and collisions: inside surfer closest to peak wins priority; outside surfer gets drop-in penalty; close surfers collide', () => {
    const world = createWorld({ numSurfers: 2, simplifiedWaves: true }, 120);
    world.waveActive.fill(0);
    const w = spawnWave(world, 80);
    world.wavePeakX[w] = 80;
    world.waveBreakStartY[w] = 72;
    world.waveBreakWidth[w] = 20;
    world.waveSpeed[w] = 20;

    // Surfer 0 is right on the peak (x = 80); Surfer 1 is on the shoulder (x = 96)
    world.surferX[0] = 80;
    world.surferY[0] = 82;
    world.surferX[1] = 96;
    world.surferY[1] = 82;

    const res = step(world, [ActionCode.CATCH, ActionCode.CATCH]);
    // Surfer 0 gets the wave with priority
    expect(world.surferState[0]).toBe(SurferStatus.RIDING);
    expect(world.surferCatches[0]).toBe(1);
    expect(res.breakdowns[0].priorityBonus).toBeGreaterThan(0);

    // Surfer 1 committed a drop-in and is penalized + wiped out
    expect(world.surferDropIns[1]).toBe(1);
    expect(res.breakdowns[1].dropInPenalty).toBeLessThan(0);
    expect(world.surferState[1]).toBe(SurferStatus.WIPED_OUT);

    // Test pairwise collision
    const worldCol = createWorld({ numSurfers: 2 }, 121);
    worldCol.waveActive.fill(0);
    worldCol.surferX[0] = 80;
    worldCol.surferY[0] = 100;
    worldCol.surferX[1] = 82; // distance 2 < collisionRadius 6.5
    worldCol.surferY[1] = 100;

    const colRes = step(worldCol, [ActionCode.RIGHT, ActionCode.LEFT]);
    expect(worldCol.surferCollisions[0]).toBe(1);
    expect(worldCol.surferCollisions[1]).toBe(1);
    expect(colRes.breakdowns[0].collisionPenalty).toBeLessThan(0);
    expect(colRes.breakdowns[1].collisionPenalty).toBeLessThan(0);
  });

  it('8. Q-table update & continuous observation normalization: Bellman update shifts Q-values toward target', () => {
    const world = createWorld({ numSurfers: 2 }, 500);
    const obs0 = observe(world, 0);
    expect(obs0.version).toBe(CONTRACT_VERSION);
    expect(obs0.features.length).toBe(OBSERVATION_DIM);
    for (const f of obs0.features) {
      expect(f).toBeGreaterThanOrEqual(-1.0);
      expect(f).toBeLessThanOrEqual(1.0);
    }

    const qPolicy = new QTablePolicy('TestLearner');
    const key = discretizeObservation(obs0);
    expect(qPolicy.getQValues(key)[ActionCode.CATCH]).toBe(0);

    step(world, [ActionCode.CATCH, ActionCode.WAIT]);
    const obs1 = observe(world, 0);

    const tdError = qPolicy.update(obs0, ActionCode.CATCH, 15.0, obs1, false, 0.25, 0.95);
    expect(tdError).toBeCloseTo(15.0, 4);
    expect(qPolicy.getQValues(key)[ActionCode.CATCH]).toBeCloseTo(3.75, 4);
    expect(qPolicy.selectAction(obs0)).toBe(ActionCode.CATCH);

    // Deterministic epsilon-greedy exploration
    const rng1 = new SeededRng(42);
    const rng2 = new SeededRng(42);
    const a1 = qPolicy.selectActionEpsilonGreedy(obs0, 0.5, rng1);
    const a2 = qPolicy.selectActionEpsilonGreedy(obs0, 0.5, rng2);
    expect(a1).toBe(a2);
  });

  it('9. Policy serialization & version compatibility checks: round-trip JSON preserves Q-table and metadata', () => {
    const policy = new QTablePolicy('ChampBot', {
      trainingEpisodes: 1000,
      avgCatchRate: 0.42,
      avgReward: 28.5,
    });
    const world = createWorld({ numSurfers: 1 }, 777);
    const obs = observe(world, 0);
    policy.update(obs, ActionCode.RIGHT, 4.0, obs, false, 0.5, 0.9);

    const jsonStr = serializePolicyToJsonString(policy);
    const restored = deserializePolicyFromJsonString(jsonStr);

    expect(restored.getMetadata()).toEqual(policy.getMetadata());
    expect(restored.selectAction(obs)).toBe(ActionCode.RIGHT);
    expect(restored.getStateCount()).toBe(policy.getStateCount());

    // Incompatible version should throw
    const badPayload = JSON.parse(jsonStr) as SerializedPolicy;
    (badPayload as { version: number }).version = 999;
    expect(() => QTablePolicy.fromJSON(badPayload)).toThrow(/Incompatible policy version/);

    // Neural ONNX placeholder validates contract compatibility
    const neural = new NeuralOnnxPolicyPlaceholder();
    expect(neural.getMetadata().version).toBe(CONTRACT_VERSION);
    expect(() => neural.validateCompatibility( JSON.parse(jsonStr) )).not.toThrow();
  });

  it('10. Headless simulation & QTrainer run without any DOM globals and replay reference JSON fixtures', () => {
    // Verify cloneWorld and full episode completion
    const world = createWorld({ numSurfers: 2, maxSteps: 30 }, 314);
    const cloned = cloneWorld(world);
    expect(cloned).toEqual(world);

    while (!isEpisodeDone(world)) {
      step(world, [ActionCode.WAIT, ActionCode.OUT]);
    }
    expect(isEpisodeDone(world)).toBe(true);
    const metrics = getEpisodeMetrics(world);
    expect(metrics.version).toBe(CONTRACT_VERSION);
    expect(metrics.totalSteps).toBe(30);

    // Short headless trainer smoke test
    const trainer = new QTrainer({
      totalEpisodes: 8,
      paddleSpeedMultiplier: 1.0,
      numCompetitors: 1,
      seed: 1234,
      maxStepsPerEpisode: 60,
    });
    const progress = trainer.stepBatch(8);
    expect(trainer.isDone()).toBe(true);
    expect(progress.episode).toBe(8);
    expect(progress.stateCount).toBeGreaterThan(0);

    // Replay shipped reference scenarios from tests/fixtures/reference-scenarios.json
    const scenarios = referenceScenariosData as unknown as ReferenceScenario[];
    expect(scenarios.length).toBeGreaterThanOrEqual(3);

    for (const scenario of scenarios) {
      const result = replayAndVerifyScenario(scenario);
      expect(result.mismatches).toEqual([]);
      expect(result.passed).toBe(true);
    }
  });
});
