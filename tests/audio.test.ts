import { describe, expect, it } from 'vitest';
import {
  BGM_BAR_COUNT,
  BGM_BASS_ROOTS,
  BGM_LEAD_MELODY,
  BGM_LEAD_STEP_TRIGGERS,
  BGM_STEPS_PER_BAR,
  BGM_TEMPO_BPM,
  BGM_TOTAL_STEPS,
  getBassMidiForStep,
  isMuted,
  isPercussionStep,
  midiToFreq,
  playSfx,
  setMuted,
  SFX_DEFINITIONS,
  startBgm,
  stopBgm,
} from '../src/audio/audioEngine.ts';
import type { SfxName } from '../src/audio/audioEngine.ts';

describe('8-bit Chiptune Audio Engine & 겨울바람 BGM Data', () => {
  it('computes midiToFreq accurately (MIDI 69 -> 440 Hz, octave ratios)', () => {
    expect(midiToFreq(69)).toBeCloseTo(440, 6);
    expect(midiToFreq(60)).toBeCloseTo(261.625565, 5); // C4
    expect(midiToFreq(72)).toBeCloseTo(523.251131, 5); // C5
    expect(midiToFreq(81)).toBeCloseTo(880, 5); // A5
    expect(midiToFreq(57)).toBeCloseTo(220, 5); // A3
  });

  it('validates 겨울바람 lead melody data integrity (64 eighth-note steps, MIDI 55..72 or rest)', () => {
    expect(BGM_TEMPO_BPM).toBe(132);
    expect(BGM_TOTAL_STEPS).toBe(64);
    expect(BGM_BAR_COUNT).toBe(8);
    expect(BGM_STEPS_PER_BAR).toBe(8);

    const totalSteps = BGM_LEAD_MELODY.reduce((acc, note) => acc + note.steps, 0);
    expect(totalSteps).toBe(64);

    for (const note of BGM_LEAD_MELODY) {
      expect(note.steps).toBeGreaterThan(0);
      expect(Number.isInteger(note.steps)).toBe(true);
      if (note.midi !== null) {
        expect(note.midi).toBeGreaterThanOrEqual(55);
        expect(note.midi).toBeLessThanOrEqual(72);
      }
    }

    expect(BGM_LEAD_STEP_TRIGGERS.length).toBe(64);
    const triggeredCount = BGM_LEAD_STEP_TRIGGERS.filter((t) => t !== null).length;
    expect(triggeredCount).toBe(BGM_LEAD_MELODY.length);
  });

  it('validates bass bar roots and alternating root / root+7 pattern', () => {
    expect(BGM_BASS_ROOTS).toHaveLength(8);
    expect(BGM_BASS_ROOTS).toEqual([36, 43, 36, 43, 36, 43, 36, 36]);

    // Bar 1 (root 36): step 0 -> 36, step 1 -> 43
    expect(getBassMidiForStep(0)).toBe(36);
    expect(getBassMidiForStep(1)).toBe(43);

    // Bar 2 (root 43): step 8 -> 43, step 9 -> 50
    expect(getBassMidiForStep(8)).toBe(43);
    expect(getBassMidiForStep(9)).toBe(50);

    // Percussion ticks only on off-beat eighth steps (1, 3, 5, 7 of each bar)
    for (let step = 0; step < 64; step++) {
      expect(isPercussionStep(step)).toBe(step % 2 === 1);
    }
  });

  it('validates SFX definition table sanity for all named SFX', () => {
    const expectedNames: SfxName[] = [
      'click',
      'splash',
      'catch',
      'fail',
      'collision',
      'jingle',
    ];

    for (const name of expectedNames) {
      const def = SFX_DEFINITIONS[name];
      expect(def).toBeDefined();
      expect(def.name).toBe(name);
      expect(def.totalDuration).toBeGreaterThan(0);
      expect(def.events.length).toBeGreaterThanOrEqual(1);

      for (const ev of def.events) {
        expect(ev.duration).toBeGreaterThan(0);
        expect(ev.startTime).toBeGreaterThanOrEqual(0);
        expect(ev.gain).toBeGreaterThan(0);
        expect(['square', 'triangle', 'noise']).toContain(ev.waveform);
      }
    }
  });

  it('handles mute state and no-DOM playback calls safely without throwing', () => {
    setMuted(true);
    expect(isMuted()).toBe(true);

    setMuted(false);
    expect(isMuted()).toBe(false);

    // Should be no-ops in headless Node environment without AudioContext
    expect(() => {
      startBgm();
      playSfx('click');
      playSfx('splash');
      playSfx('catch');
      playSfx('fail');
      playSfx('collision');
      playSfx('jingle');
      stopBgm();
    }).not.toThrow();
  });
});
