import './style.css';
import {
  initAudio,
  isMuted,
  playSfx,
  setMuted,
  startBgm,
} from './audio/audioEngine.ts';
import {
  ActionCode,
  CONTRACT_VERSION,
  SurferStatus,
} from './contracts/types.ts';
import type { ActionInput, SurferAction, WorldState } from './contracts/types.ts';
import { HumanPolicy } from './policies/policy.ts';
import type { Policy } from './policies/policy.ts';
import { QTablePolicy } from './policies/qTable.ts';
import { RuleBotPolicy } from './policies/ruleBot.ts';
import {
  deserializePolicyFromJsonString,
  hasSavedPolicyInLocalStorage,
  loadPolicyFromLocalStorage,
  savePolicyToLocalStorage,
  serializePolicyToJsonString,
} from './policies/storage.ts';
import { OceanCanvasRenderer } from './render/canvasRenderer.ts';
import { TrainingChartRenderer } from './render/chartRenderer.ts';
import {
  createWorld,
  isEpisodeDone,
  observe,
  step,
} from './sim/world.ts';
import type { TrainingProgressPoint } from './training/trainer.ts';
import type {
  WorkerInboundMessage,
  WorkerOutboundMessage,
} from './training/trainingWorker.ts';

function getEl<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing DOM element #${id}`);
  return el as T;
}

// --- Application State ---
let heatSeed = 1042;
let playPaddleSpeed = 1.0;
let playVsTrainedAi = false;

const humanPolicy = new HumanPolicy();
const ruleBots: RuleBotPolicy[] = [
  new RuleBotPolicy(1, 0.8),
  new RuleBotPolicy(2, 0.72),
  new RuleBotPolicy(3, 0.65),
];

let activeQPolicy: QTablePolicy =
  loadPolicyFromLocalStorage() ?? new QTablePolicy('Trained Q-Surfer');

let world: WorldState = createWorld(
  {
    numSurfers: 4,
    paddleSpeedMultiplier: playPaddleSpeed,
    maxSteps: 1200,
  },
  heatSeed
);

// --- DOM Elements ---
const tabBtnPlay = getEl<HTMLButtonElement>('tab-btn-play');
const tabBtnTrain = getEl<HTMLButtonElement>('tab-btn-train');
const panelPlay = getEl<HTMLElement>('panel-play');
const panelTrain = getEl<HTMLElement>('panel-train');

const hudWaves = getEl<HTMLElement>('hud-waves');
const hudCatches = getEl<HTMLElement>('hud-catches');
const hudStaminaFill = getEl<HTMLElement>('hud-stamina-fill');
const hudScore = getEl<HTMLElement>('hud-score');
const btnMute = getEl<HTMLButtonElement>('btn-mute');
const playModeBadge = getEl<HTMLElement>('play-mode-badge');
const playStatusMsg = getEl<HTMLElement>('play-status-msg');
const btnResetHeat = getEl<HTMLButtonElement>('btn-reset-heat');

const oceanCanvas = getEl<HTMLCanvasElement>('ocean-canvas');
const btnPaddle = getEl<HTMLButtonElement>('btn-paddle');
const btnCatch = getEl<HTMLButtonElement>('btn-catch');
const playSpeedSlider = getEl<HTMLInputElement>('play-speed-slider');
const playSpeedVal = getEl<HTMLElement>('play-speed-val');
const btnModeBots = getEl<HTMLButtonElement>('btn-mode-bots');
const btnModeTrained = getEl<HTMLButtonElement>('btn-mode-trained');

// Training Tab Elements
const epButtons = Array.from(
  document.querySelectorAll<HTMLButtonElement>('.ep-btn')
);
const trainSpeedSlider = getEl<HTMLInputElement>('train-speed-slider');
const trainSpeedVal = getEl<HTMLElement>('train-speed-val');
const trainCompSlider = getEl<HTMLInputElement>('train-comp-slider');
const trainCompVal = getEl<HTMLElement>('train-comp-val');
const btnTrainStart = getEl<HTMLButtonElement>('btn-train-start');
const btnTrainPause = getEl<HTMLButtonElement>('btn-train-pause');
const btnTrainCancel = getEl<HTMLButtonElement>('btn-train-cancel');
const trainProgressFill = getEl<HTMLElement>('train-progress-fill');
const statEpisodes = getEl<HTMLElement>('stat-episodes');
const statCatchRate = getEl<HTMLElement>('stat-catch-rate');
const statReward = getEl<HTMLElement>('stat-reward');
const statStates = getEl<HTMLElement>('stat-states');
const trainChartCanvas = getEl<HTMLCanvasElement>('train-chart');
const policySummaryText = getEl<HTMLElement>('policy-summary-text');
const btnSaveLocal = getEl<HTMLButtonElement>('btn-save-local');
const btnLoadLocal = getEl<HTMLButtonElement>('btn-load-local');
const btnExportJson = getEl<HTMLButtonElement>('btn-export-json');
const inputImportJson = getEl<HTMLInputElement>('input-import-json');
const btnPlayVsAi = getEl<HTMLButtonElement>('btn-play-vs-ai');

const oceanRenderer = new OceanCanvasRenderer(oceanCanvas);
const chartRenderer = new TrainingChartRenderer(trainChartCanvas);

// --- Audio & Mute UI ---
function updateMuteButtonUi(): void {
  const mutedState = isMuted();
  btnMute.textContent = mutedState ? '🔇' : '🔊';
  btnMute.classList.toggle('muted', mutedState);
  btnMute.setAttribute('aria-pressed', String(mutedState));
}

btnMute.addEventListener('pointerdown', (e) => {
  e.stopPropagation();
});

btnMute.addEventListener('click', (e) => {
  e.stopPropagation();
  const nextMuted = !isMuted();
  setMuted(nextMuted);
  updateMuteButtonUi();
  if (!nextMuted) {
    playSfx('click');
  }
});

// Resume AudioContext and start BGM after the first pointer interaction
window.addEventListener(
  'pointerdown',
  (e) => {
    if (e.target instanceof Node && btnMute.contains(e.target)) {
      return;
    }
    initAudio();
    if (!isMuted()) {
      startBgm();
    }
  },
  { passive: true }
);

// --- Helper: Update Policy Summary UI ---
function refreshPolicySummaryUi(): void {
  const meta = activeQPolicy.getMetadata();
  const states = activeQPolicy.getStateCount();
  const catchPct = (meta.avgCatchRate * 100).toFixed(1);
  policySummaryText.textContent = `Policy: ${meta.name} (${meta.trainingEpisodes} eps, ${catchPct}% catch, ${states} states)`;
}

// --- Tab Navigation ---
function switchTab(tab: 'play' | 'train'): void {
  const isPlay = tab === 'play';
  tabBtnPlay.classList.toggle('active', isPlay);
  tabBtnPlay.setAttribute('aria-selected', String(isPlay));
  tabBtnTrain.classList.toggle('active', !isPlay);
  tabBtnTrain.setAttribute('aria-selected', String(!isPlay));

  panelPlay.classList.toggle('active', isPlay);
  panelPlay.hidden = !isPlay;
  panelTrain.classList.toggle('active', !isPlay);
  panelTrain.hidden = isPlay;

  if (!isPlay) {
    chartRenderer.render(trainingHistory, selectedEpisodes);
    refreshPolicySummaryUi();
  }
}

tabBtnPlay.addEventListener('click', () => {
  playSfx('click');
  switchTab('play');
});
tabBtnTrain.addEventListener('click', () => {
  playSfx('click');
  switchTab('train');
});

// --- Start / Reset Gameplay Heat ---
function startNewHeat(nextSeed?: number): void {
  heatSeed = nextSeed ?? (heatSeed + 17) >>> 0;
  world = createWorld(
    {
      numSurfers: 4,
      paddleSpeedMultiplier: playPaddleSpeed,
      maxSteps: 1200,
    },
    heatSeed
  );
  humanPolicy.setTouchTarget(world.surferX[0], world.surferY[0] - 12);
  updatePlayModeUi();
}

function updatePlayModeUi(): void {
  btnModeBots.classList.toggle('active', !playVsTrainedAi);
  btnModeTrained.classList.toggle('active', playVsTrainedAi);
  if (playVsTrainedAi) {
    const eps = activeQPolicy.getMetadata().trainingEpisodes;
    playModeBadge.textContent = `VS TRAINED AI (${eps} EPS)`;
  } else {
    playModeBadge.textContent = 'VS 3 RULE BOTS';
  }
}

btnResetHeat.addEventListener('click', () => {
  playSfx('click');
  startNewHeat();
  playStatusMsg.textContent = 'New ocean heat started!';
});

btnModeBots.addEventListener('click', () => {
  playSfx('click');
  playVsTrainedAi = false;
  updatePlayModeUi();
  playStatusMsg.textContent = 'Rivals: Deterministic Rule Bots';
});

btnModeTrained.addEventListener('click', () => {
  playSfx('click');
  playVsTrainedAi = true;
  updatePlayModeUi();
  playStatusMsg.textContent =
    activeQPolicy.getStateCount() > 0
      ? 'Surfer #2 is now your Trained Q-AI!'
      : 'Tip: Train AI in TRAIN tab for stronger rival!';
});

playSpeedSlider.addEventListener('input', () => {
  playPaddleSpeed = parseFloat(playSpeedSlider.value);
  playSpeedVal.textContent = `${playPaddleSpeed.toFixed(1)}x`;
  world.config.paddleSpeedMultiplier = playPaddleSpeed;
});

// --- Canvas Touch / Pointer Target Controls ---
function updateCanvasTargetFromPointer(clientX: number, clientY: number): void {
  const rect = oceanCanvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const normX = (clientX - rect.left) / rect.width;
  const normY = (clientY - rect.top) / rect.height;
  const simX = Math.max(8, Math.min(world.config.width - 8, normX * world.config.width));
  const simY = Math.max(10, Math.min(world.config.height - 14, normY * world.config.height));
  humanPolicy.setTouchTarget(simX, simY);
}

let pointerDownOnCanvas = false;
oceanCanvas.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  pointerDownOnCanvas = true;
  updateCanvasTargetFromPointer(e.clientX, e.clientY);
  playStatusMsg.textContent = 'Target set — hold PADDLE to move';
});

oceanCanvas.addEventListener('pointermove', (e) => {
  if (!pointerDownOnCanvas) return;
  e.preventDefault();
  updateCanvasTargetFromPointer(e.clientX, e.clientY);
});

window.addEventListener('pointerup', () => {
  pointerDownOnCanvas = false;
});

// --- Hold PADDLE Button & CATCH Button ---
function setPaddleActive(active: boolean): void {
  humanPolicy.setPaddleHeld(active);
  btnPaddle.classList.toggle('active', active);
}

btnPaddle.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  if (!humanPolicy.getTouchTarget()) {
    // Default target slightly offshore toward peak if user hasn't tapped canvas yet
    humanPolicy.setTouchTarget(world.surferX[0], Math.max(60, world.surferY[0] - 25));
  }
  setPaddleActive(true);
  if (world.surferState[0] === SurferStatus.FREE) {
    playSfx('splash');
  }
});

btnPaddle.addEventListener('pointerup', () => setPaddleActive(false));
btnPaddle.addEventListener('pointerleave', () => setPaddleActive(false));
btnPaddle.addEventListener('pointercancel', () => setPaddleActive(false));

btnCatch.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  humanPolicy.triggerCatch();
  btnCatch.classList.add('active');
  setTimeout(() => btnCatch.classList.remove('active'), 140);
});

// --- Desktop Keyboard Controls ---
const heldKeys = new Set<string>();

function updateKeyboardAction(): void {
  if (heldKeys.has('ArrowLeft') || heldKeys.has('KeyA')) {
    humanPolicy.setKeyboardAction(ActionCode.LEFT);
  } else if (heldKeys.has('ArrowRight') || heldKeys.has('KeyD')) {
    humanPolicy.setKeyboardAction(ActionCode.RIGHT);
  } else if (heldKeys.has('ArrowUp') || heldKeys.has('KeyW')) {
    humanPolicy.setKeyboardAction(ActionCode.OUT);
  } else if (heldKeys.has('ArrowDown') || heldKeys.has('KeyS')) {
    humanPolicy.setKeyboardAction(ActionCode.IN);
  } else {
    humanPolicy.setKeyboardAction(null);
  }
}

window.addEventListener('keydown', (e) => {
  if (panelPlay.hidden) return;
  if (e.code === 'Space' || e.code === 'Enter') {
    e.preventDefault();
    humanPolicy.triggerCatch();
    return;
  }
  if (
    ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'KeyW', 'KeyA', 'KeyS', 'KeyD'].includes(
      e.code
    )
  ) {
    e.preventDefault();
    heldKeys.add(e.code);
    updateKeyboardAction();
  }
});

window.addEventListener('keyup', (e) => {
  if (heldKeys.has(e.code)) {
    heldKeys.delete(e.code);
    updateKeyboardAction();
  }
});

// --- Simulation & Render Loop ---
let lastFrameTime = performance.now();
let simAccumulator = 0;

function updateHud(): void {
  hudWaves.textContent = String(world.totalWavesSpawned);
  hudCatches.textContent = String(world.surferCatches[0]);
  const stamPct = Math.round(world.surferStamina[0] * 100);
  hudStaminaFill.style.width = `${stamPct}%`;
  hudStaminaFill.style.backgroundColor =
    stamPct > 30 ? 'var(--accent-cyan)' : 'var(--accent-coral)';
  hudScore.textContent = String(Math.round(world.surferScore[0]));
}

let bestHeatScore = 0;

function advanceGameplayStep(): void {
  if (isEpisodeDone(world)) {
    const finalScore = Math.round(world.surferScore[0]);
    const isNewBest = finalScore > bestHeatScore;
    if (isNewBest) {
      bestHeatScore = finalScore;
    }
    playSfx('jingle');
    startNewHeat();
    playStatusMsg.textContent = isNewBest
      ? `🏆 Heat complete! New high score: ${finalScore}!`
      : `Heat complete! Score: ${finalScore} — New heat started`;
    updateHud();
    return;
  }

  const numSurfers = world.config.numSurfers;
  const actions: ActionInput[] = new Array(numSurfers);

  // Surfer 0: Human player
  const obs0 = observe(world, 0);
  const humanAct: SurferAction = humanPolicy.selectAction(obs0);
  actions[0] = humanAct;

  // Surfers 1..N-1: Either Trained Q-AI (surfer 1) + RuleBots, or all RuleBots
  for (let i = 1; i < numSurfers; i++) {
    const obsI = observe(world, i);
    const policy: Policy =
      playVsTrainedAi && i === 1
        ? activeQPolicy
        : ruleBots[(i - 1) % ruleBots.length];
    actions[i] = policy.selectAction(obsI, world, i);
  }

  const prevState = world.surferState[0];
  const prevCatches = world.surferCatches[0];
  const prevDropIns = world.surferDropIns[0];
  const prevCollisions = world.surferCollisions[0];
  const attemptedCatch =
    prevState === SurferStatus.FREE && humanAct.action === ActionCode.CATCH;
  const isPaddling =
    prevState === SurferStatus.FREE &&
    (Boolean(humanAct.paddle) ||
      humanAct.action === ActionCode.LEFT ||
      humanAct.action === ActionCode.RIGHT ||
      humanAct.action === ActionCode.OUT ||
      humanAct.action === ActionCode.IN);

  const stepRes = step(world, actions, world.config.dt);

  const caughtWave = world.surferCatches[0] > prevCatches;
  const droppedIn = world.surferDropIns[0] > prevDropIns;
  const collided = world.surferCollisions[0] > prevCollisions;

  if (caughtWave) {
    playSfx('catch');
    playStatusMsg.textContent = `🌊 Caught wave! +${Math.round(stepRes.rewards[0])} reward`;
  } else if (droppedIn) {
    playSfx('collision');
    playStatusMsg.textContent = '⚠️ Drop-in penalty! Inside surfer had priority';
  } else if (collided) {
    playSfx('collision');
    playStatusMsg.textContent = '💥 Surfer collision! (-6 penalty)';
  } else if (attemptedCatch) {
    playSfx('fail');
    playStatusMsg.textContent = 'Missed wave — time takeoff near breaking peak!';
  } else if (world.surferState[0] === SurferStatus.WIPED_OUT) {
    playStatusMsg.textContent = '💥 Wiped out! Recovering...';
  } else if (isPaddling) {
    playSfx('splash');
  }

  updateHud();
}

function gameLoop(now: number): void {
  const elapsedSec = Math.min(0.25, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  if (!panelPlay.hidden) {
    simAccumulator += elapsedSec;
    const dt = world.config.dt;
    while (simAccumulator >= dt) {
      advanceGameplayStep();
      simAccumulator -= dt;
    }

    const surferLabels = [
      'YOU',
      playVsTrainedAi ? 'Q-AI' : 'B1',
      'B2',
      'B3',
    ];
    oceanRenderer.render(world, {
      touchTarget: humanPolicy.getTouchTarget(),
      paddleHeld: humanPolicy.isPaddleHeld(),
      surferLabels,
    });
  }

  requestAnimationFrame(gameLoop);
}

// --- Web Worker Training Controller ---
let selectedEpisodes = 1000;
let trainPaddleSpeed = 1.0;
let trainCompetitors = 2;
let trainingWorker: Worker | null = null;
let isTraining = false;
let isPaused = false;
let trainingHistory: TrainingProgressPoint[] = [];

epButtons.forEach((btn) => {
  btn.addEventListener('click', () => {
    if (isTraining) return;
    playSfx('click');
    epButtons.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    selectedEpisodes = Number(btn.dataset.episodes) || 1000;
    statEpisodes.textContent = `0 / ${selectedEpisodes}`;
    chartRenderer.render(trainingHistory, selectedEpisodes);
  });
});

trainSpeedSlider.addEventListener('input', () => {
  trainPaddleSpeed = parseFloat(trainSpeedSlider.value);
  trainSpeedVal.textContent = `${trainPaddleSpeed.toFixed(1)}x`;
});

trainCompSlider.addEventListener('input', () => {
  trainCompetitors = parseInt(trainCompSlider.value, 10);
  trainCompVal.textContent = String(trainCompetitors);
});

function ensureWorker(): Worker {
  if (!trainingWorker) {
    trainingWorker = new Worker(
      new URL('./training/trainingWorker.ts', import.meta.url),
      { type: 'module' }
    );
    trainingWorker.onmessage = (event: MessageEvent<WorkerOutboundMessage>) => {
      handleWorkerMessage(event.data);
    };
  }
  return trainingWorker;
}

function updateTrainingStatsUi(point: TrainingProgressPoint, total: number): void {
  const pct = Math.min(100, Math.round((point.episode / Math.max(1, total)) * 100));
  trainProgressFill.style.width = `${pct}%`;
  statEpisodes.textContent = `${point.episode} / ${total}`;
  statCatchRate.textContent = `${(point.rollingCatchRate * 100).toFixed(1)}%`;
  statReward.textContent = point.rollingReward.toFixed(1);
  statStates.textContent = String(point.stateCount);
}

function setTrainingButtonsState(running: boolean, paused: boolean): void {
  isTraining = running;
  isPaused = paused;
  btnTrainStart.disabled = running;
  btnTrainPause.disabled = !running;
  btnTrainPause.textContent = paused ? 'RESUME' : 'PAUSE';
  btnTrainCancel.disabled = !running;
}

function handleWorkerMessage(msg: WorkerOutboundMessage): void {
  if (!msg || msg.version !== CONTRACT_VERSION) return;

  switch (msg.type) {
    case 'progress': {
      trainingHistory.push(msg.point);
      activeQPolicy = QTablePolicy.fromJSON(msg.policy);
      updateTrainingStatsUi(msg.point, msg.totalEpisodes);
      chartRenderer.render(trainingHistory, msg.totalEpisodes);
      refreshPolicySummaryUi();
      break;
    }
    case 'paused': {
      activeQPolicy = QTablePolicy.fromJSON(msg.policy);
      setTrainingButtonsState(true, true);
      refreshPolicySummaryUi();
      break;
    }
    case 'cancelled': {
      setTrainingButtonsState(false, false);
      break;
    }
    case 'completed': {
      trainingHistory.push(msg.point);
      activeQPolicy = QTablePolicy.fromJSON(msg.policy);
      savePolicyToLocalStorage(activeQPolicy);
      updateTrainingStatsUi(msg.point, selectedEpisodes);
      chartRenderer.render(trainingHistory, selectedEpisodes);
      setTrainingButtonsState(false, false);
      refreshPolicySummaryUi();
      break;
    }
  }
}

btnTrainStart.addEventListener('click', () => {
  playSfx('click');
  trainingHistory = [];
  trainProgressFill.style.width = '0%';
  setTrainingButtonsState(true, false);

  const worker = ensureWorker();
  const startMsg: WorkerInboundMessage = {
    version: CONTRACT_VERSION,
    type: 'start',
    params: {
      totalEpisodes: selectedEpisodes,
      paddleSpeedMultiplier: trainPaddleSpeed,
      numCompetitors: trainCompetitors,
      seed: 2026,
    },
  };
  worker.postMessage(startMsg);
});

btnTrainPause.addEventListener('click', () => {
  if (!trainingWorker || !isTraining) return;
  playSfx('click');
  if (isPaused) {
    setTrainingButtonsState(true, false);
    trainingWorker.postMessage({
      version: CONTRACT_VERSION,
      type: 'resume',
    } satisfies WorkerInboundMessage);
  } else {
    trainingWorker.postMessage({
      version: CONTRACT_VERSION,
      type: 'pause',
    } satisfies WorkerInboundMessage);
  }
});

btnTrainCancel.addEventListener('click', () => {
  if (!trainingWorker || !isTraining) return;
  playSfx('click');
  trainingWorker.postMessage({
    version: CONTRACT_VERSION,
    type: 'cancel',
  } satisfies WorkerInboundMessage);
});

// --- Policy Persistence (Save/Load Local, Export/Import JSON, Play vs AI) ---
btnSaveLocal.addEventListener('click', () => {
  playSfx('click');
  savePolicyToLocalStorage(activeQPolicy);
  policySummaryText.textContent = `Saved to localStorage! (${
    activeQPolicy.getMetadata().trainingEpisodes
  } eps)`;
});

btnLoadLocal.addEventListener('click', () => {
  playSfx('click');
  if (!hasSavedPolicyInLocalStorage()) {
    policySummaryText.textContent = 'No saved policy found in localStorage.';
    return;
  }
  const loaded = loadPolicyFromLocalStorage();
  if (loaded) {
    activeQPolicy = loaded;
    refreshPolicySummaryUi();
  }
});

btnExportJson.addEventListener('click', () => {
  playSfx('click');
  const json = serializePolicyToJsonString(activeQPolicy);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `crowded-lineup-policy-v${CONTRACT_VERSION}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

inputImportJson.addEventListener('change', async () => {
  const file = inputImportJson.files?.[0];
  if (!file) return;
  playSfx('click');
  try {
    const text = await file.text();
    const imported = deserializePolicyFromJsonString(text);
    activeQPolicy = imported;
    savePolicyToLocalStorage(activeQPolicy);
    refreshPolicySummaryUi();
  } catch (err) {
    policySummaryText.textContent = `Import error: ${(err as Error).message}`;
  } finally {
    inputImportJson.value = '';
  }
});

btnPlayVsAi.addEventListener('click', () => {
  playSfx('click');
  playVsTrainedAi = true;
  playPaddleSpeed = trainPaddleSpeed;
  playSpeedSlider.value = String(playPaddleSpeed);
  playSpeedVal.textContent = `${playPaddleSpeed.toFixed(1)}x`;
  startNewHeat();
  switchTab('play');
  playStatusMsg.textContent = 'Competing against your Trained Q-AI!';
});

// --- Initialize App & PWA Service Worker ---
updateMuteButtonUi();
updateHud();
updatePlayModeUi();
refreshPolicySummaryUi();
chartRenderer.render(trainingHistory, selectedEpisodes);
requestAnimationFrame(gameLoop);

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {
      // Service worker registration optional in non-HTTPS preview environments
    });
  });
}
