/** Small music-template matcher. No speech model; pitch classes and note spectrum. */
export type MusicFrame = { pitch: number[]; chroma: number[]; rms: number };
export type MusicTemplate = { version: 1; frames: MusicFrame[] };
export const HOP_SECONDS = .032;
const normalize = (values: number[]) => { const norm = Math.hypot(...values) || 1; return values.map(v => v / norm); };
export function musicFrame(samples: Float32Array, rate: number): MusicFrame {
  let energy = 0;
  const windowed = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) { energy += samples[i] ** 2; windowed[i] = samples[i] * (.5 - .5 * Math.cos(2 * Math.PI * i / (samples.length - 1))); }
  const pitch = Array.from({length: 48}, (_, index) => {
    const frequency = 440 * 2 ** ((index + 40 - 69) / 12);
    const coefficient = 2 * Math.cos(2 * Math.PI * frequency / rate);
    let previous = 0, before = 0;
    for (const sample of windowed) { const next = sample + coefficient * previous - before; before = previous; previous = next; }
    return Math.sqrt(Math.max(0, previous * previous + before * before - coefficient * previous * before));
  });
  const chroma = new Array<number>(12).fill(0);
  pitch.forEach((value, index) => { chroma[(index + 40) % 12] += value; });
  return { pitch: normalize(pitch), chroma: normalize(chroma), rms: Math.sqrt(energy / samples.length) };
}
export function trimMusic(frames: MusicFrame[]): MusicFrame[] {
  const peak = Math.max(0, ...frames.map(f => f.rms));
  const floor = Math.max(.004, peak * .08);
  const start = frames.findIndex(f => f.rms >= floor);
  let end = frames.length;
  while (end > 0 && frames[end - 1].rms < floor) end--;
  return start < 0 ? [] : frames.slice(start, end);
}
export function validTemplate(value: unknown): value is MusicTemplate {
  const t = value as MusicTemplate;
  return t?.version === 1 && Array.isArray(t.frames) && t.frames.length >= 6 && t.frames.length <= 140 && t.frames.every(f =>
    Number.isFinite(f.rms) && f.rms >= 0 && f.rms <= 1 && [f.pitch, f.chroma].every((v, i) => Array.isArray(v) && v.length === (i ? 12 : 48) && v.every(x => Number.isFinite(x) && x >= 0 && x <= 1)));
}
const cosine = (a: number[], b: number[]) => Math.max(0, 1 - a.reduce((s, v, i) => s + v * b[i], 0));
/** Length-bounded DTW. Relative energy retains attacks while allowing level changes. */
export function musicDistance(a: MusicFrame[], b: MusicFrame[]): number {
  if (!a.length || !b.length || a.length / b.length < .65 || a.length / b.length > 1.5) return Infinity;
  const peakA = Math.max(...a.map(f => f.rms)), peakB = Math.max(...b.map(f => f.rms));
  let previous = new Float64Array(b.length + 1).fill(Infinity); previous[0] = 0;
  for (let i = 1; i <= a.length; i++) {
    const row = new Float64Array(b.length + 1).fill(Infinity);
    for (let j = 1; j <= b.length; j++) {
      if (Math.abs(i / a.length - j / b.length) > .22) continue;
      const x = a[i-1], y = b[j-1];
      const cost = .55 * cosine(x.chroma, y.chroma) + .35 * cosine(x.pitch, y.pitch) + .1 * Math.abs(x.rms / peakA - y.rms / peakB);
      row[j] = cost + Math.min(previous[j-1], previous[j] + .025, row[j-1] + .025);
    }
    previous = row;
  }
  return previous[b.length] / Math.max(a.length, b.length);
}
/** Bounded frame buffer; handles arbitrary PCM block boundaries at the native rate. */
export class MusicFeatures {
  private pending = new Float32Array(0);
  push(chunk: Float32Array, rate: number): MusicFrame[] {
    const input = new Float32Array(this.pending.length + chunk.length); input.set(this.pending); input.set(chunk, this.pending.length);
    const size = Math.round(rate * .064), hop = Math.round(rate * HOP_SECONDS);
    const frames: MusicFrame[] = []; let offset = 0;
    for (; offset + size <= input.length; offset += hop) frames.push(musicFrame(input.subarray(offset, offset + size), rate));
    this.pending = input.slice(offset); return frames;
  }
}
