/**
 * Procedural 8-bit (PICO-8 style) Web Audio engine for Crowded Lineup.
 *
 * - Pure Web Audio API synthesis (square, triangle, white-noise buffer).
 * - BGM: "겨울바람" (백순진 작사·작곡) 8-bit arrangement at 132 BPM.
 * - Lookahead scheduler (35ms interval, 0.12s lookahead window).
 * - Pure note/SFX data and helpers exported for headless unit tests.
 */

export const MUTE_STORAGE_KEY = 'crowded-lineup-muted';
export const BGM_TEMPO_BPM = 132;
export const BGM_STEP_DURATION_SEC = 60 / BGM_TEMPO_BPM / 2; // Eighth-note step (~0.2273s)
export const BGM_STEPS_PER_BAR = 8;
export const BGM_BAR_COUNT = 8;
export const BGM_TOTAL_STEPS = BGM_STEPS_PER_BAR * BGM_BAR_COUNT; // 64 steps
export const SPLASH_MIN_INTERVAL_MS = 220;

export type WaveformType = 'square' | 'triangle' | 'noise';

export type SfxName =
  | 'click'
  | 'splash'
  | 'catch'
  | 'fail'
  | 'collision'
  | 'jingle';

export interface MelodyNoteSpec {
  /** MIDI note number (e.g. 60 = C4) or null for a rest ('R') */
  readonly midi: number | null;
  /** Duration in eighth-note steps */
  readonly steps: number;
}

export interface SfxEventSpec {
  readonly waveform: WaveformType;
  readonly startTime: number;
  readonly duration: number;
  readonly gain: number;
  readonly midi?: number;
  readonly freqStart?: number;
  readonly freqEnd?: number;
  readonly filterFreq?: number;
}

export interface SfxDefinition {
  readonly name: SfxName;
  readonly totalDuration: number;
  readonly events: readonly SfxEventSpec[];
}

/**
 * Convert a MIDI note number to frequency in Hz (A4 = MIDI 69 = 440 Hz).
 */
export function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/**
 * Lead melody for "겨울바람" (작사·작곡 백순진), C major, 8 bars = 64 eighth-note steps.
 *
 * Bar 1 (손이 시려워): (64,2) (67,1) (65,1) (67,2) (69,2)
 * Bar 2 (발이 시려워): (65,2) (64,2) (62,2) (60,2)
 * Bar 3 (겨울바람 때문에): (60,1) (60,1) (60,1) (60,1) (62,1) (62,1) (64,2)
 * Bar 4 (꽁꽁꽁 chant): (60,1) (60,1) (60,1) (R,1) (60,1) (60,1) (60,1) (R,1)
 * Bar 5 (손이 꽁꽁꽁): (64,1) (65,1) (67,1) (67,1) (67,2) (69,2)
 * Bar 6 (발이 꽁꽁꽁): (65,1) (64,1) (62,1) (62,1) (60,2) (60,2)
 * Bar 7 (겨울바람 때문에): (60,1) (60,1) (62,1) (62,1) (60,1) (59,1) (60,2)
 * Bar 8 (꽁꽁꽁 chant): (60,1) (60,1) (60,1) (R,1) (60,1) (60,1) (60,1) (R,1)
 */
export const BGM_LEAD_MELODY: readonly MelodyNoteSpec[] = [
  // Bar 1 (손이 시려워)
  { midi: 64, steps: 2 },
  { midi: 67, steps: 1 },
  { midi: 65, steps: 1 },
  { midi: 67, steps: 2 },
  { midi: 69, steps: 2 },
  // Bar 2 (발이 시려워)
  { midi: 65, steps: 2 },
  { midi: 64, steps: 2 },
  { midi: 62, steps: 2 },
  { midi: 60, steps: 2 },
  // Bar 3 (겨울바람 때문에)
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 62, steps: 1 },
  { midi: 62, steps: 1 },
  { midi: 64, steps: 2 },
  // Bar 4 (꽁꽁꽁 chant)
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: null, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: null, steps: 1 },
  // Bar 5 (손이 꽁꽁꽁)
  { midi: 64, steps: 1 },
  { midi: 65, steps: 1 },
  { midi: 67, steps: 1 },
  { midi: 67, steps: 1 },
  { midi: 67, steps: 2 },
  { midi: 69, steps: 2 },
  // Bar 6 (발이 꽁꽁꽁)
  { midi: 65, steps: 1 },
  { midi: 64, steps: 1 },
  { midi: 62, steps: 1 },
  { midi: 62, steps: 1 },
  { midi: 60, steps: 2 },
  { midi: 60, steps: 2 },
  // Bar 7 (겨울바람 때문에)
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 62, steps: 1 },
  { midi: 62, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 59, steps: 1 },
  { midi: 60, steps: 2 },
  // Bar 8 (꽁꽁꽁 chant)
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: null, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: 60, steps: 1 },
  { midi: null, steps: 1 },
] as const;

/**
 * Bass bar roots (1 root per bar across 8 bars): C2 = 36, G2 = 43.
 */
export const BGM_BASS_ROOTS: readonly number[] = [
  36, 43, 36, 43, 36, 43, 36, 36,
] as const;

/**
 * Compute the bass MIDI note for a given eighth-note step (0..63).
 * Alternates root and root+7 on each eighth step of the bar.
 */
export function getBassMidiForStep(stepIndex: number): number {
  const normalizedStep = ((stepIndex % BGM_TOTAL_STEPS) + BGM_TOTAL_STEPS) % BGM_TOTAL_STEPS;
  const barIndex = Math.floor(normalizedStep / BGM_STEPS_PER_BAR);
  const stepInBar = normalizedStep % BGM_STEPS_PER_BAR;
  const root = BGM_BASS_ROOTS[barIndex];
  return stepInBar % 2 === 0 ? root : root + 7;
}

/**
 * Whether an eighth-note step index (0..63) has an off-beat percussion tick (steps 1,3,5,7 of each bar).
 */
export function isPercussionStep(stepIndex: number): boolean {
  const stepInBar = ((stepIndex % BGM_STEPS_PER_BAR) + BGM_STEPS_PER_BAR) % BGM_STEPS_PER_BAR;
  return stepInBar % 2 === 1;
}

/**
 * Expand `BGM_LEAD_MELODY` into a 64-entry lookup table indexed by eighth-note step.
 * Entries where a note starts carry the `MelodyNoteSpec`; continuation steps are `null`.
 */
export function buildLeadStepTriggers(
  melody: readonly MelodyNoteSpec[] = BGM_LEAD_MELODY
): ReadonlyArray<MelodyNoteSpec | null> {
  const triggers: Array<MelodyNoteSpec | null> = new Array(BGM_TOTAL_STEPS).fill(null);
  let cursor = 0;
  for (const item of melody) {
    if (cursor < BGM_TOTAL_STEPS) {
      triggers[cursor] = item;
    }
    cursor += item.steps;
  }
  return triggers;
}

export const BGM_LEAD_STEP_TRIGGERS = buildLeadStepTriggers(BGM_LEAD_MELODY);

/**
 * Pure SFX definition table for all named sound effects.
 */
export const SFX_DEFINITIONS: Readonly<Record<SfxName, SfxDefinition>> = {
  click: {
    name: 'click',
    totalDuration: 0.05,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.05,
        gain: 0.16,
        freqStart: 880,
        freqEnd: 660,
      },
    ],
  },
  splash: {
    name: 'splash',
    totalDuration: 0.06,
    events: [
      {
        waveform: 'noise',
        startTime: 0,
        duration: 0.06,
        gain: 0.07,
        filterFreq: 1300,
      },
    ],
  },
  catch: {
    name: 'catch',
    totalDuration: 0.28,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.07,
        gain: 0.18,
        midi: 72, // C5
      },
      {
        waveform: 'square',
        startTime: 0.07,
        duration: 0.07,
        gain: 0.18,
        midi: 76, // E5
      },
      {
        waveform: 'square',
        startTime: 0.14,
        duration: 0.07,
        gain: 0.18,
        midi: 79, // G5
      },
      {
        waveform: 'square',
        startTime: 0.21,
        duration: 0.07,
        gain: 0.2,
        midi: 84, // C6
      },
    ],
  },
  fail: {
    name: 'fail',
    totalDuration: 0.25,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.25,
        gain: 0.18,
        freqStart: 220,
        freqEnd: 110,
      },
    ],
  },
  collision: {
    name: 'collision',
    totalDuration: 0.12,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: 0.12,
        gain: 0.24,
        freqStart: 90,
        freqEnd: 55,
      },
      {
        waveform: 'noise',
        startTime: 0,
        duration: 0.12,
        gain: 0.16,
        filterFreq: 650,
      },
    ],
  },
  jingle: {
    name: 'jingle',
    totalDuration: BGM_STEP_DURATION_SEC * 5,
    events: [
      {
        waveform: 'square',
        startTime: 0,
        duration: BGM_STEP_DURATION_SEC,
        gain: 0.18,
        midi: 72, // C5 (1 step)
      },
      {
        waveform: 'square',
        startTime: BGM_STEP_DURATION_SEC * 1,
        duration: BGM_STEP_DURATION_SEC,
        gain: 0.18,
        midi: 76, // E5 (1 step)
      },
      {
        waveform: 'square',
        startTime: BGM_STEP_DURATION_SEC * 2,
        duration: BGM_STEP_DURATION_SEC,
        gain: 0.18,
        midi: 79, // G5 (1 step)
      },
      {
        waveform: 'square',
        startTime: BGM_STEP_DURATION_SEC * 3,
        duration: BGM_STEP_DURATION_SEC * 2,
        gain: 0.2,
        midi: 84, // C6 (2 steps)
      },
    ],
  },
};

// --- Runtime Web Audio Engine State ---

const LOOKAHEAD_INTERVAL_MS = 35;
const SCHEDULE_AHEAD_SEC = 0.12;
const BGM_LEAD_GAIN = 0.15;
const BGM_BASS_GAIN = 0.12;
const BGM_NOISE_GAIN = 0.03;

function readInitialMutedState(): boolean {
  try {
    if (typeof localStorage !== 'undefined') {
      return localStorage.getItem(MUTE_STORAGE_KEY) === 'true';
    }
  } catch {
    // Ignore storage access errors in restricted/headless contexts
  }
  return false;
}

let muted: boolean = readInitialMutedState();
let audioCtx: AudioContext | null = null;
let masterGain: GainNode | null = null;
let noiseBuffer: AudioBuffer | null = null;

let schedulerTimer: ReturnType<typeof setInterval> | null = null;
let currentStep = 0;
let nextNoteTime = 0;
let lastSplashTimeMs = -Infinity;

function getAudioContextConstructor(): typeof AudioContext | null {
  if (typeof globalThis === 'undefined') return null;
  const g = globalThis as unknown as {
    AudioContext?: typeof AudioContext;
    webkitAudioContext?: typeof AudioContext;
  };
  return g.AudioContext ?? g.webkitAudioContext ?? null;
}

function createWhiteNoiseBuffer(ctx: AudioContext): AudioBuffer {
  const durationSec = 0.5;
  const length = Math.max(1, Math.floor(ctx.sampleRate * durationSec));
  const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  // Deterministic LCG noise so buffer creation is fast and consistent
  let seed = 0x12345678;
  for (let i = 0; i < length; i++) {
    seed = (1664525 * seed + 1013904223) >>> 0;
    data[i] = (seed / 0xffffffff) * 2 - 1;
  }
  return buffer;
}

/**
 * Lazily create and resume the Web Audio API context and master gain node.
 * Safe to call multiple times or in headless environments without Web Audio.
 */
export function initAudio(): AudioContext | null {
  if (!audioCtx) {
    const Ctor = getAudioContextConstructor();
    if (!Ctor) return null;
    try {
      audioCtx = new Ctor();
      masterGain = audioCtx.createGain();
      masterGain.gain.value = muted ? 0 : 1;
      masterGain.connect(audioCtx.destination);
      noiseBuffer = createWhiteNoiseBuffer(audioCtx);
    } catch {
      audioCtx = null;
      masterGain = null;
      return null;
    }
  }

  if (audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {
      // Ignore autoplay rejection until a valid user gesture occurs
    });
  }

  if (masterGain) {
    masterGain.gain.setValueAtTime(muted ? 0 : 1, audioCtx.currentTime);
  }

  return audioCtx;
}

/**
 * Return whether audio is currently muted.
 */
export function isMuted(): boolean {
  return muted;
}

/**
 * Set the mute state, persist it in localStorage (`crowded-lineup-muted`),
 * and start/stop BGM accordingly.
 */
export function setMuted(m: boolean): void {
  muted = Boolean(m);
  try {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem(MUTE_STORAGE_KEY, String(muted));
    }
  } catch {
    // Ignore localStorage quota/privacy errors
  }

  if (muted) {
    if (masterGain && audioCtx) {
      masterGain.gain.setValueAtTime(0, audioCtx.currentTime);
    }
    stopBgm();
  } else {
    initAudio();
    if (masterGain && audioCtx) {
      masterGain.gain.setValueAtTime(1, audioCtx.currentTime);
    }
    startBgm();
  }
}

function scheduleOscNote(
  ctx: AudioContext,
  dest: AudioNode,
  waveform: 'square' | 'triangle',
  freqStart: number,
  freqEnd: number | undefined,
  startTime: number,
  duration: number,
  peakGain: number
): void {
  const osc = ctx.createOscillator();
  const gainNode = ctx.createGain();

  osc.type = waveform;
  osc.frequency.setValueAtTime(freqStart, startTime);
  if (typeof freqEnd === 'number' && freqEnd !== freqStart) {
    osc.frequency.exponentialRampToValueAtTime(
      Math.max(20, freqEnd),
      startTime + duration
    );
  }

  const attack = Math.min(0.008, duration * 0.15);
  const releaseTime = startTime + duration;
  gainNode.gain.setValueAtTime(0.0001, startTime);
  gainNode.gain.linearRampToValueAtTime(peakGain, startTime + attack);
  gainNode.gain.exponentialRampToValueAtTime(0.0001, releaseTime);

  osc.connect(gainNode);
  gainNode.connect(dest);

  osc.start(startTime);
  osc.stop(releaseTime + 0.005);
}

function scheduleNoiseBurst(
  ctx: AudioContext,
  dest: AudioNode,
  startTime: number,
  duration: number,
  peakGain: number,
  filterFreq = 1800
): void {
  if (!noiseBuffer) {
    noiseBuffer = createWhiteNoiseBuffer(ctx);
  }

  const source = ctx.createBufferSource();
  source.buffer = noiseBuffer;
  source.loop = true;

  const filter = ctx.createBiquadFilter();
  filter.type = 'bandpass';
  filter.frequency.setValueAtTime(filterFreq, startTime);
  filter.Q.setValueAtTime(1.2, startTime);

  const gainNode = ctx.createGain();
  const attack = Math.min(0.005, duration * 0.15);
  const releaseTime = startTime + duration;
  gainNode.gain.setValueAtTime(0.0001, startTime);
  gainNode.gain.linearRampToValueAtTime(peakGain, startTime + attack);
  gainNode.gain.exponentialRampToValueAtTime(0.0001, releaseTime);

  source.connect(filter);
  filter.connect(gainNode);
  gainNode.connect(dest);

  source.start(startTime);
  source.stop(releaseTime + 0.005);
}

function scheduleBgmStep(stepIndex: number, stepTime: number): void {
  if (!audioCtx || !masterGain || muted) return;

  // 1. Lead melody (square wave, short decay per note)
  const leadTrigger = BGM_LEAD_STEP_TRIGGERS[stepIndex];
  if (leadTrigger && leadTrigger.midi !== null) {
    const freq = midiToFreq(leadTrigger.midi);
    const noteDur = leadTrigger.steps * BGM_STEP_DURATION_SEC * 0.88;
    scheduleOscNote(
      audioCtx,
      masterGain,
      'square',
      freq,
      undefined,
      stepTime,
      noteDur,
      BGM_LEAD_GAIN
    );
  }

  // 2. Bass (triangle wave, alternating root / root+7 each eighth step)
  const bassMidi = getBassMidiForStep(stepIndex);
  const bassFreq = midiToFreq(bassMidi);
  scheduleOscNote(
    audioCtx,
    masterGain,
    'triangle',
    bassFreq,
    undefined,
    stepTime,
    BGM_STEP_DURATION_SEC * 0.78,
    BGM_BASS_GAIN
  );

  // 3. Off-beat percussion noise tick (steps 1,3,5,7 of each bar)
  if (isPercussionStep(stepIndex)) {
    scheduleNoiseBurst(
      audioCtx,
      masterGain,
      stepTime,
      0.032,
      BGM_NOISE_GAIN,
      2800
    );
  }
}

function schedulerTick(): void {
  if (!audioCtx || !masterGain || muted) return;
  if (audioCtx.state !== 'running') return;

  const now = audioCtx.currentTime;
  if (nextNoteTime < now - 0.05) {
    nextNoteTime = now + 0.01;
  }

  while (nextNoteTime < now + SCHEDULE_AHEAD_SEC) {
    scheduleBgmStep(currentStep, nextNoteTime);
    nextNoteTime += BGM_STEP_DURATION_SEC;
    currentStep = (currentStep + 1) % BGM_TOTAL_STEPS;
  }
}

/**
 * Start the looping "겨울바람" chiptune BGM using a lookahead scheduler.
 */
export function startBgm(): void {
  if (muted) return;
  const ctx = initAudio();
  if (!ctx) return;

  if (schedulerTimer !== null) {
    return;
  }

  currentStep = 0;
  nextNoteTime = ctx.currentTime + 0.02;
  schedulerTick();
  schedulerTimer = setInterval(schedulerTick, LOOKAHEAD_INTERVAL_MS);
}

/**
 * Stop the BGM lookahead scheduler.
 */
export function stopBgm(): void {
  if (schedulerTimer !== null) {
    clearInterval(schedulerTimer);
    schedulerTimer = null;
  }
}

/**
 * Play one of the named 8-bit sound effects (`click`, `splash`, `catch`, `fail`, `collision`, `jingle`).
 */
export function playSfx(name: SfxName): void {
  if (muted) return;

  const def = SFX_DEFINITIONS[name];
  if (!def) return;

  const nowMs =
    typeof performance !== 'undefined' ? performance.now() : Date.now();
  if (name === 'splash') {
    if (nowMs - lastSplashTimeMs < SPLASH_MIN_INTERVAL_MS) {
      return;
    }
    lastSplashTimeMs = nowMs;
  }

  const ctx = initAudio();
  if (!ctx || !masterGain) return;

  const baseTime = ctx.currentTime + 0.005;
  for (const ev of def.events) {
    const start = baseTime + ev.startTime;
    if (ev.waveform === 'noise') {
      scheduleNoiseBurst(
        ctx,
        masterGain,
        start,
        ev.duration,
        ev.gain,
        ev.filterFreq ?? 1200
      );
    } else {
      const freqStart =
        typeof ev.midi === 'number'
          ? midiToFreq(ev.midi)
          : (ev.freqStart ?? 440);
      scheduleOscNote(
        ctx,
        masterGain,
        ev.waveform,
        freqStart,
        ev.freqEnd,
        start,
        ev.duration,
        ev.gain
      );
    }
  }
}
