import { SurferStatus } from '../contracts/types.ts';
import type { WorldState } from '../contracts/types.ts';

export interface RenderOptions {
  touchTarget: { x: number; y: number } | null;
  paddleHeld: boolean;
  surferLabels: string[];
  showDensityStrip?: boolean;
}

const SURFER_PALETTES = [
  { board: '#ffd93d', suit: '#ff4d6d', accent: '#ffffff' }, // Player / Surfer 0 (Gold board, coral suit)
  { board: '#00f5d4', suit: '#7b2cbf', accent: '#e0aaff' }, // Surfer 1 (Neon cyan board, purple suit — Trained AI or Bot 1)
  { board: '#ff9f1c', suit: '#2ec4b6', accent: '#cbf3f0' }, // Surfer 2
  { board: '#aacc00', suit: '#d90429', accent: '#edf2f4' }, // Surfer 3
  { board: '#f15bb5', suit: '#00bbf9', accent: '#fee440' }, // Surfer 4
  { board: '#9b5de5', suit: '#00f5d4', accent: '#ffffff' }, // Surfer 5
];

function getSurferPalette(index: number): { board: string; suit: string; accent: string } {
  if (index < SURFER_PALETTES.length) {
    return SURFER_PALETTES[index];
  }
  // Procedural deterministic pixel-art palette for 40+ surfer lineups
  const hue = (index * 137.508) % 360;
  const suitHue = (hue + 150) % 360;
  return {
    board: `hsl(${Math.round(hue)}, 88%, 62%)`,
    suit: `hsl(${Math.round(suitHue)}, 78%, 48%)`,
    accent: '#e8f7fc',
  };
}

/**
 * Pure 2D pixel-art renderer for `WorldState`.
 * Completely decoupled from simulation logic.
 * Renders onto a low-resolution logical canvas (160x240) scaled with CSS `image-rendering: pixelated`.
 */
export class OceanCanvasRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not initialize 2D canvas context');
    }
    this.ctx = ctx;
    this.ctx.imageSmoothingEnabled = false;
  }

  public render(world: WorldState, options: RenderOptions): void {
    const { ctx } = this;
    const { width, height } = world.config;

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      ctx.imageSmoothingEnabled = false;
    }

    // 1. Top-down ocean depth bands (offshore deep water at top -> shallow sandbar at bottom)
    this.drawOceanBackground(ctx, width, height, world.stepCount);

    // 2. Irregular fading whitewater trails (remnants of broken waves)
    this.drawWhitewaterTrails(ctx, world);

    // 3. Incoming swells, green peeling shoulders, and breaking white crests
    this.drawWaves(ctx, world);

    // 4. Touch target reticle (if player has tapped a destination)
    if (options.touchTarget) {
      this.drawTouchTarget(
        ctx,
        options.touchTarget.x,
        options.touchTarget.y,
        world.surferX[0],
        world.surferY[0],
        options.paddleHeld,
        world.stepCount
      );
    }

    // 5. Surfer pixel sprites + status indicators
    this.drawSurfers(ctx, world, options.surferLabels);

    // 6. Optional lineup density strip along top edge
    if (options.showDensityStrip !== false && world.config.numSurfers >= 8) {
      this.drawCrowdDensityStrip(ctx, world);
    }
  }

  private drawCrowdDensityStrip(
    ctx: CanvasRenderingContext2D,
    world: WorldState
  ): void {
    const bins = 16;
    const binW = world.config.width / bins;
    const counts = new Array(bins).fill(0);
    for (let i = 0; i < world.config.numSurfers; i++) {
      if (world.surferState[i] === SurferStatus.WIPED_OUT) continue;
      const b = Math.min(bins - 1, Math.max(0, Math.floor(world.surferX[i] / binW)));
      counts[b] += 1;
    }
    for (let b = 0; b < bins; b++) {
      const c = counts[b];
      if (c === 0) continue;
      ctx.fillStyle =
        c >= 4
          ? 'rgba(255, 77, 109, 0.8)'
          : c >= 2
            ? 'rgba(255, 217, 61, 0.65)'
            : 'rgba(0, 245, 212, 0.45)';
      ctx.fillRect(Math.round(b * binW) + 1, 1, Math.max(2, Math.round(binW) - 2), 2);
    }
  }

  private drawOceanBackground(
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    stepCount: number
  ): void {
    // Deep offshore (y: 0..65)
    ctx.fillStyle = '#072a40';
    ctx.fillRect(0, 0, width, 65);

    // Outside lineup (y: 65..125)
    ctx.fillStyle = '#0a3c59';
    ctx.fillRect(0, 65, width, 60);

    // Inside reform zone (y: 125..185)
    ctx.fillStyle = '#0f5273';
    ctx.fillRect(0, 125, width, 60);

    // Shallow turquoise shorebreak (y: 185..218)
    ctx.fillStyle = '#19728c';
    ctx.fillRect(0, 185, width, 33);

    // Wet sand & shoreline beach at bottom (y: 218..240)
    ctx.fillStyle = '#a48656';
    ctx.fillRect(0, 218, width, 7);
    ctx.fillStyle = '#d9be8b';
    ctx.fillRect(0, 225, width, height - 225);

    // Subtle pixel shoreline wash line
    const washOffset = Math.round(Math.sin(stepCount * 0.15) * 2);
    ctx.fillStyle = '#98e2ee';
    for (let x = 0; x < width; x += 4) {
      const waveJit = ((x * 7 + stepCount) % 3) - 1;
      ctx.fillRect(x, 216 + washOffset + waveJit, 3, 2);
    }

    // Subtle horizontal water texture ripples moving toward shore
    ctx.fillStyle = '#114b6b';
    const drift = (stepCount * 2) % 24;
    for (let y = 12; y < 205; y += 24) {
      const ry = y + drift;
      if (ry >= 212) continue;
      for (let x = (y * 13) % 19; x < width - 8; x += 28) {
        ctx.fillRect(x, ry, 5, 1);
      }
    }

    // Take-off zone subtle guide ticks on left/right edges
    ctx.fillStyle = 'rgba(0, 245, 212, 0.28)';
    for (let y = 68; y <= 124; y += 8) {
      ctx.fillRect(1, y, 2, 4);
      ctx.fillRect(width - 3, y, 2, 4);
    }
  }

  private drawWhitewaterTrails(
    ctx: CanvasRenderingContext2D,
    world: WorldState
  ): void {
    const { maxWhitewaters } = world.config;

    for (let i = 0; i < maxWhitewaters; i++) {
      if (world.wwActive[i] === 0) continue;
      const cx = Math.round(world.wwX[i]);
      const cy = Math.round(world.wwY[i]);
      const halfW = Math.round(world.wwWidth[i] * 0.5);
      const intensity = world.wwIntensity[i];

      if (intensity <= 0.05) continue;

      const startX = Math.max(4, cx - halfW);
      const endX = Math.min(world.config.width - 4, cx + halfW);

      // Draw irregular top-down bubbly foam patches (not triangles/mountains!)
      for (let x = startX; x < endX; x += 3) {
        // Deterministic pseudo-random pattern per pixel column
        const hash = ((x * 37 + i * 17 + Math.floor(cy * 0.5)) & 7) - 3;
        const distNorm = Math.abs(x - cx) / Math.max(1, halfW);
        if (distNorm > 0.92 && (hash & 1) === 0) continue;

        const py = cy + Math.round(hash * 0.6);
        const thickness = Math.max(1, Math.round((1 - distNorm * 0.65) * 4 * intensity));

        ctx.fillStyle =
          intensity > 0.6
            ? 'rgba(225, 248, 255, 0.72)'
            : intensity > 0.3
              ? 'rgba(176, 226, 242, 0.48)'
              : 'rgba(130, 198, 222, 0.28)';
        ctx.fillRect(x, py - thickness, 2, thickness + 1);

        // Extra bright foam specs
        if (intensity > 0.45 && ((x + i) & 3) === 0) {
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(x, py, 1, 1);
        }
      }
    }
  }

  private drawWaves(ctx: CanvasRenderingContext2D, world: WorldState): void {
    const { maxWaves, width } = world.config;

    for (let w = 0; w < maxWaves; w++) {
      if (world.waveActive[w] === 0) continue;

      const wy = Math.round(world.waveY[w]);
      const peakX = Math.round(world.wavePeakX[w]);
      const breakStartY = world.waveBreakStartY[w];
      const breakW = Math.round(world.waveBreakWidth[w]);
      const quality = world.waveQuality[w];
      const isBreaking = wy >= breakStartY;

      // Full swell span across the ocean
      const swellHalfSpan = Math.min(
        width - 8,
        Math.round(46 + quality * 28)
      );
      const leftX = Math.max(4, peakX - swellHalfSpan);
      const rightX = Math.min(width - 4, peakX + swellHalfSpan);

      // 1. Draw unbroken swell band (smooth dark trough + cyan face)
      for (let x = leftX; x <= rightX; x += 2) {
        const distFromPeak = Math.abs(x - peakX);
        // Slight bow curve so the peak leads slightly shoreward
        const curveOffset = -Math.round((distFromPeak * distFromPeak) / 1400);
        const yAtX = wy + curveOffset;

        // Back slope of swell (offshore side)
        ctx.fillStyle = '#062336';
        ctx.fillRect(x, yAtX - 5, 2, 2);

        // Main unbroken swell body
        ctx.fillStyle = isBreaking ? '#0c6980' : '#0e5a73';
        ctx.fillRect(x, yAtX - 3, 2, 4);

        // Clean green/turquoise catchable shoulder near the peak or breaking edge
        const inCatchableShoulder =
          distFromPeak <= breakW + 18 && distFromPeak >= Math.max(0, breakW - 4);
        if (wy >= breakStartY - 8 && inCatchableShoulder) {
          ctx.fillStyle = '#00d2b8';
          ctx.fillRect(x, yAtX - 1, 2, 3);
        } else {
          // Swell crest highlight line
          ctx.fillStyle = '#38a3b8';
          ctx.fillRect(x, yAtX, 2, 1);
        }
      }

      // 2. If wave hasn't broken yet, draw subtle peak focal marker so players can read the peak
      if (!isBreaking) {
        ctx.fillStyle = '#5ce1e6';
        ctx.fillRect(peakX - 3, wy - 1, 7, 2);
        ctx.fillRect(peakX - 1, wy - 2, 3, 4);
      } else {
        // 3. Draw active breaking crest (bright white tumbling lip expanding laterally from peak)
        const breakLeft = Math.max(4, peakX - breakW);
        const breakRight = Math.min(width - 4, peakX + breakW);

        for (let x = breakLeft; x <= breakRight; x += 2) {
          const distFromPeak = Math.abs(x - peakX);
          const curveOffset = -Math.round((distFromPeak * distFromPeak) / 1400);
          const jitter = ((x * 13 + world.stepCount * 3) & 3) - 1;
          const yAtX = wy + curveOffset;

          // Trailing turbulence behind lip
          ctx.fillStyle = '#9be8f5';
          ctx.fillRect(x, yAtX - 3, 2, 3);

          // Bright white rolling lip at the leading edge
          ctx.fillStyle = '#ffffff';
          ctx.fillRect(x, yAtX - 1, 2, 3 + (jitter > 0 ? 1 : 0));
        }
      }
    }
  }

  private drawTouchTarget(
    ctx: CanvasRenderingContext2D,
    tx: number,
    ty: number,
    sx: number,
    sy: number,
    paddleHeld: boolean,
    stepCount: number
  ): void {
    const rx = Math.round(tx);
    const ry = Math.round(ty);

    // Dotted pixel line from player surfer to target when holding paddle
    if (paddleHeld) {
      ctx.fillStyle = 'rgba(255, 217, 61, 0.55)';
      const steps = 8;
      for (let i = 1; i < steps; i++) {
        if ((i + stepCount) % 2 === 0) continue;
        const px = Math.round(sx + ((rx - sx) * i) / steps);
        const py = Math.round(sy + ((ry - sy) * i) / steps);
        ctx.fillRect(px, py, 2, 2);
      }
    }

    // Pixel crosshair reticle
    const color = paddleHeld ? '#ffd93d' : 'rgba(255, 217, 61, 0.65)';
    ctx.fillStyle = color;
    ctx.fillRect(rx - 4, ry, 3, 1);
    ctx.fillRect(rx + 2, ry, 3, 1);
    ctx.fillRect(rx, ry - 4, 1, 3);
    ctx.fillRect(rx, ry + 2, 1, 3);
  }

  private drawSurfers(
    ctx: CanvasRenderingContext2D,
    world: WorldState,
    labels: string[]
  ): void {
    const n = world.config.numSurfers;

    for (let i = n - 1; i >= 0; i--) {
      const x = Math.round(world.surferX[i]);
      const y = Math.round(world.surferY[i]);
      const status = world.surferState[i];
      const palette = getSurferPalette(i);

      if (status === SurferStatus.WIPED_OUT) {
        // Wiped out splash ring + floating board
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x - 3, y - 2, 2, 2);
        ctx.fillRect(x + 2, y + 1, 2, 2);
        ctx.fillStyle = palette.board;
        ctx.fillRect(x - 2, y, 5, 2);
        continue;
      }

      const isRiding = status === SurferStatus.RIDING;
      const speedSq =
        world.surferVx[i] * world.surferVx[i] +
        world.surferVy[i] * world.surferVy[i];

      // Wake spray when riding or paddling
      if (isRiding) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(x - 3, y - 4, 7, 2);
        ctx.fillStyle = '#80ffdb';
        ctx.fillRect(x - 2, y - 6, 5, 2);
      } else if (speedSq > 16) {
        ctx.fillStyle = 'rgba(200, 245, 255, 0.7)';
        ctx.fillRect(x - 2, y + 3, 1, 2);
        ctx.fillRect(x + 2, y + 3, 1, 2);
      }

      // Surfboard (vertical 3x7 pixel silhouette, angled slightly if carving)
      ctx.fillStyle = palette.board;
      ctx.fillRect(x - 1, y - 4, 3, 8);
      // Nose & tail trim
      ctx.fillRect(x, y - 5, 1, 1);
      ctx.fillRect(x, y + 4, 1, 1);

      // Surfer body
      ctx.fillStyle = palette.suit;
      if (isRiding) {
        // Standing stance on board
        ctx.fillRect(x - 1, y - 2, 3, 3);
        ctx.fillStyle = palette.accent;
        ctx.fillRect(x, y - 3, 2, 2); // head/shoulders
      } else {
        // Prone paddling silhouette
        ctx.fillRect(x - 1, y - 2, 3, 4);
        ctx.fillStyle = '#222222';
        ctx.fillRect(x, y - 3, 1, 2); // head
      }

      // Mini stamina bar above Player 0 (and tiny tag)
      const label = labels[i] ?? (i === 0 ? 'YOU' : `B${i}`);
      if (i === 0) {
        // Player indicator arrow + mini stamina bar
        const stamW = Math.round(world.surferStamina[0] * 10);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
        ctx.fillRect(x - 5, y - 9, 10, 2);
        ctx.fillStyle =
          world.surferStamina[0] > 0.3 ? '#00f5d4' : '#ff4d6d';
        ctx.fillRect(x - 5, y - 9, stamW, 2);
      } else {
        // Small pixel badge dot for AI/Bot
        ctx.fillStyle = palette.board;
        ctx.fillRect(x - 1, y - 8, 3, 2);
      }

      // Crisp micro label for riding state
      if (isRiding) {
        ctx.fillStyle = '#ffd93d';
        ctx.fillRect(x - 2, y - 11, 5, 1);
      }
      void label;
    }
  }
}
