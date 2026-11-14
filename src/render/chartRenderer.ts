import type { TrainingProgressPoint } from '../training/trainer.ts';

/**
 * Lightweight Canvas 2D chart renderer for the Training tab.
 * Plots rolling catch rate (%) and rolling average reward across training episodes.
 */
export class TrainingChartRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Could not get 2D context for training chart');
    }
    this.ctx = ctx;
  }

  public render(points: TrainingProgressPoint[], totalEpisodes: number): void {
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#061a29';
    ctx.fillRect(0, 0, w, h);

    const padL = 36;
    const padR = 14;
    const padT = 16;
    const padB = 22;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Grid lines
    ctx.strokeStyle = 'rgba(120, 200, 230, 0.16)';
    ctx.lineWidth = 1;
    ctx.font = '10px monospace';
    ctx.fillStyle = '#7fb3c8';

    for (let i = 0; i <= 4; i++) {
      const frac = i / 4;
      const y = Math.round(padT + plotH * (1 - frac)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();

      const pctLabel = `${Math.round(frac * 60)}%`;
      ctx.fillText(pctLabel, 4, y + 3);
    }

    // X-axis label
    ctx.fillStyle = '#7fb3c8';
    ctx.fillText('0', padL, h - 6);
    const endLabel = `${totalEpisodes} eps`;
    ctx.fillText(endLabel, w - padR - ctx.measureText(endLabel).width, h - 6);

    if (points.length === 0) {
      ctx.fillStyle = '#689db3';
      ctx.font = '11px monospace';
      ctx.fillText('Press TRAIN to start Q-learning in Web Worker', padL + 12, padT + plotH / 2);
      return;
    }

    // Plot Catch Rate curve (scaled 0% -> 60% catch rate)
    ctx.strokeStyle = '#00f5d4';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      const xFrac = Math.min(1, pt.episode / Math.max(1, totalEpisodes));
      const yFrac = Math.min(1, Math.max(0, pt.rollingCatchRate / 0.6));
      const px = padL + xFrac * plotW;
      const py = padT + (1 - yFrac) * plotH;

      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Fill area under Catch Rate curve
    const lastPt = points[points.length - 1];
    const lastX = padL + Math.min(1, lastPt.episode / Math.max(1, totalEpisodes)) * plotW;
    ctx.lineTo(lastX, padT + plotH);
    ctx.lineTo(padL, padT + plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 245, 212, 0.14)';
    ctx.fill();
  }
}
