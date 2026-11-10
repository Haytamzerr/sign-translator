/**
 * Light measurement and low-light boost.
 *
 * Hand tracking relies on contrast between the hand and the background. In a
 * dark room the camera image is dim and noisy, MediaPipe becomes unsure, and
 * hands are lost. The app measures the average brightness a few times per
 * second (on a tiny 32 x 24 copy of the frame, which is almost free) and, when
 * the image is dark, gives MediaPipe a brightened copy of the frame instead.
 * The picture on screen is not changed.
 */

/** Average brightness 0..255 below which the image counts as dark. */
export const DARK_LEVEL = 75;
/** Brightness above which a boosted image goes back to normal (hysteresis). */
export const BRIGHT_LEVEL = 95;
/** Brightness the boost aims for. */
const TARGET_LEVEL = 125;

/** Perceived brightness (luma, ITU-R BT.601) of RGBA pixels. */
export function meanLuma(rgba: Uint8ClampedArray): number {
  let sum = 0;
  const n = rgba.length / 4;
  for (let i = 0; i < rgba.length; i += 4) sum += 0.299 * rgba[i]! + 0.587 * rgba[i + 1]! + 0.114 * rgba[i + 2]!;
  return n ? sum / n : 0;
}

/** CSS filter that brings an image of the given brightness close to the target. */
export function boostFilter(brightness: number): string {
  const gain = Math.min(2.4, Math.max(1, TARGET_LEVEL / Math.max(brightness, 1)));
  return `brightness(${gain.toFixed(2)}) contrast(1.15)`;
}

/** Decides whether the boost should be on, with hysteresis so it does not flicker. */
export function nextBoostState(active: boolean, brightness: number): boolean {
  return active ? brightness < BRIGHT_LEVEL : brightness < DARK_LEVEL;
}

export class LightMeter {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  /** Smoothed brightness, 0..255 (-1 until measured). */
  brightness = -1;

  measure(video: HTMLVideoElement): number {
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.canvas.width = 32;
      this.canvas.height = 24;
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    }
    if (!this.ctx) return this.brightness;
    this.ctx.drawImage(video, 0, 0, 32, 24);
    const value = meanLuma(this.ctx.getImageData(0, 0, 32, 24).data);
    this.brightness = this.brightness < 0 ? value : this.brightness * 0.6 + value * 0.4;
    return this.brightness;
  }
}

/** Produces the brightened copy of the frame given to MediaPipe. */
export class FrameBooster {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  apply(video: HTMLVideoElement, filter: string): HTMLCanvasElement | null {
    if (!this.canvas) {
      this.canvas = document.createElement('canvas');
      this.ctx = this.canvas.getContext('2d');
    }
    if (!this.ctx) return null;
    if (this.canvas.width !== video.videoWidth || this.canvas.height !== video.videoHeight) {
      this.canvas.width = video.videoWidth;
      this.canvas.height = video.videoHeight;
    }
    this.ctx.filter = filter;
    this.ctx.drawImage(video, 0, 0);
    return this.canvas;
  }
}
