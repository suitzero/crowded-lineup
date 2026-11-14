import {
  ActionCode,
  CONTRACT_VERSION,
} from '../contracts/types.ts';
import type { WorldConfig } from '../contracts/types.ts';
import { QTablePolicy } from '../policies/qTable.ts';
import { RuleBotPolicy } from '../policies/ruleBot.ts';
import { SeededRng } from '../sim/prng.ts';
import {
  createWorld,
  getEpisodeMetrics,
  isEpisodeDone,
  observe,
  step,
} from '../sim/world.ts';

export interface TrainingParams {
  totalEpisodes: number;
  paddleSpeedMultiplier: number;
  numCompetitors: number;
  seed: number;
  maxStepsPerEpisode?: number;
  alpha?: number;
  gamma?: number;
  epsilonStart?: number;
  epsilonEnd?: number;
}

export interface TrainingProgressPoint {
  episode: number;
  rollingCatchRate: number;
  rollingReward: number;
  epsilon: number;
  stateCount: number;
}

/**
 * Run a single training episode for surfer 0 (the QTablePolicy learner)
 * alongside optional deterministic RuleBot competitors.
 */
export function runTrainingEpisode(
  policy: QTablePolicy,
  bots: RuleBotPolicy[],
  config: Partial<WorldConfig>,
  episodeSeed: number,
  rng: SeededRng,
  epsilon: number,
  alpha: number,
  gamma: number
): { catchRate: number; totalReward: number; catches: number } {
  const world = createWorld(config, episodeSeed);
  const numSurfers = world.config.numSurfers;
  const actions: ActionCode[] = new Array(numSurfers).fill(ActionCode.WAIT);

  let totalReward = 0;
  let prevObs = observe(world, 0);

  while (!isEpisodeDone(world)) {
    // Learner action (surfer 0)
    const learnerAction = policy.selectActionEpsilonGreedy(prevObs, epsilon, rng);
    actions[0] = learnerAction;

    // Competitor bot actions (surfers 1..N-1)
    for (let i = 1; i < numSurfers; i++) {
      const botObs = observe(world, i);
      const bot = bots[(i - 1) % bots.length];
      actions[i] = bot.selectAction(botObs, world, i);
    }

    const stepRes = step(world, actions, world.config.dt);
    const nextObs = observe(world, 0);
    const reward = stepRes.rewards[0];
    totalReward += reward;

    policy.update(
      prevObs,
      learnerAction,
      reward,
      nextObs,
      stepRes.done,
      alpha,
      gamma
    );

    prevObs = nextObs;
  }

  const metrics = getEpisodeMetrics(world);
  return {
    catchRate: metrics.surferCatchRates[0],
    totalReward: +totalReward.toFixed(3),
    catches: metrics.surferCatches[0],
  };
}

/**
 * Stateful trainer class that runs episodes in chunks so Web Workers or CLI scripts
 * can pause, resume, cancel, and stream progress updates.
 */
export class QTrainer {
  public readonly params: Required<TrainingParams>;
  public readonly policy: QTablePolicy;
  private readonly rng: SeededRng;
  private readonly bots: RuleBotPolicy[];

  public currentEpisode = 0;
  private recentCatchRates: number[] = [];
  private recentRewards: number[] = [];
  public history: TrainingProgressPoint[] = [];

  constructor(params: TrainingParams, existingPolicy?: QTablePolicy) {
    this.params = {
      maxStepsPerEpisode: 420,
      alpha: 0.2,
      gamma: 0.96,
      epsilonStart: 0.35,
      epsilonEnd: 0.04,
      ...params,
    };

    this.rng = new SeededRng(this.params.seed);
    this.policy =
      existingPolicy ??
      new QTablePolicy('Trained Q-Surfer', {
        paddleSpeedMultiplier: this.params.paddleSpeedMultiplier,
        numCompetitors: this.params.numCompetitors,
      });

    this.bots = [];
    for (let i = 0; i < Math.max(1, this.params.numCompetitors); i++) {
      this.bots.push(new RuleBotPolicy(i + 1, 0.65 + (i % 3) * 0.1));
    }
  }

  public isDone(): boolean {
    return this.currentEpisode >= this.params.totalEpisodes;
  }

  /**
   * Run a batch of episodes synchronously and return the latest progress snapshot.
   */
  public stepBatch(batchSize: number): TrainingProgressPoint {
    const targetEp = Math.min(
      this.params.totalEpisodes,
      this.currentEpisode + Math.max(1, batchSize)
    );

    while (this.currentEpisode < targetEp) {
      this.currentEpisode += 1;
      const progressFrac = this.currentEpisode / Math.max(1, this.params.totalEpisodes);
      const epsilon =
        this.params.epsilonStart +
        (this.params.epsilonEnd - this.params.epsilonStart) * progressFrac;

      // Curriculum stage 1: first 25% of training (or when 0 competitors selected)
      // uses single surfer + simplified waves so catching policy forms rapidly.
      const rollingAvg = this.getRollingCatchRate();
      const inStage1 =
        progressFrac < 0.25 && rollingAvg < 0.22;

      const activeCompetitors = inStage1 ? 0 : this.params.numCompetitors;
      const simplifiedWaves = inStage1 || this.params.numCompetitors === 0;

      const episodeSeed = (this.params.seed + this.currentEpisode * 7919) >>> 0;
      const res = runTrainingEpisode(
        this.policy,
        this.bots,
        {
          numSurfers: 1 + activeCompetitors,
          paddleSpeedMultiplier: this.params.paddleSpeedMultiplier,
          simplifiedWaves,
          maxSteps: this.params.maxStepsPerEpisode,
          waveSpawnInterval: 6.5,
        },
        episodeSeed,
        this.rng,
        epsilon,
        this.params.alpha,
        this.params.gamma
      );

      this.recentCatchRates.push(res.catchRate);
      if (this.recentCatchRates.length > 50) this.recentCatchRates.shift();

      this.recentRewards.push(res.totalReward);
      if (this.recentRewards.length > 50) this.recentRewards.shift();
    }

    const point: TrainingProgressPoint = {
      episode: this.currentEpisode,
      rollingCatchRate: this.getRollingCatchRate(),
      rollingReward: this.getRollingReward(),
      epsilon: +(
        this.params.epsilonStart +
        (this.params.epsilonEnd - this.params.epsilonStart) *
          (this.currentEpisode / Math.max(1, this.params.totalEpisodes))
      ).toFixed(4),
      stateCount: this.policy.getStateCount(),
    };

    this.history.push(point);
    this.policy.updateMetadata({
      version: CONTRACT_VERSION,
      trainingEpisodes: this.currentEpisode,
      paddleSpeedMultiplier: this.params.paddleSpeedMultiplier,
      numCompetitors: this.params.numCompetitors,
      avgCatchRate: point.rollingCatchRate,
      avgReward: point.rollingReward,
    });

    return point;
  }

  public getRollingCatchRate(): number {
    if (this.recentCatchRates.length === 0) return 0;
    const sum = this.recentCatchRates.reduce((a, b) => a + b, 0);
    return +(sum / this.recentCatchRates.length).toFixed(4);
  }

  public getRollingReward(): number {
    if (this.recentRewards.length === 0) return 0;
    const sum = this.recentRewards.reduce((a, b) => a + b, 0);
    return +(sum / this.recentRewards.length).toFixed(2);
  }
}
