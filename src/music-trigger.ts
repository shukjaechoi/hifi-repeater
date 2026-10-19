/** Small music-template matcher. No speech model; pitch classes and note spectrum. */
export type MusicFrame = { pitch: number[]; chroma: number[]; rms: number };
export type LegacyMusicTemplate = { version: 1; frames: MusicFrame[] };
export type MusicTemplate = LegacyMusicTemplate | { version: 2; examples: MusicFrame[][]; threshold: number };
export type MusicSegment = { frames: MusicFrame[]; start: number; end: number };
export type Calibration = { template?: MusicTemplate; scores: number[]; error?: string; warning?: string };
export const MAX_EXAMPLES = 8;
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
export function validFrames(value: unknown, maximum = 125): value is MusicFrame[] {
  return Array.isArray(value) && value.length >= 6 && value.length <= maximum && value.every(f =>
    Number.isFinite(f.rms) && f.rms >= 0 && f.rms <= 1 && [f.pitch, f.chroma].every((v, i) => Array.isArray(v) && v.length === (i ? 12 : 48) && v.every((x: number) => Number.isFinite(x) && x >= 0 && x <= 1)));
}
export function validTemplate(value: unknown): value is MusicTemplate {
  const t = value as MusicTemplate;
  if (t?.version === 1) return validFrames(t.frames, 140);
  return t?.version === 2 && Array.isArray(t.examples) && t.examples.length >= 3 && t.examples.length <= MAX_EXAMPLES && t.examples.every(example => validFrames(example))
    && Number.isFinite(t.threshold) && t.threshold >= .10 && t.threshold <= .24;
}
// Require agreement with two examples, so one unusual example cannot trigger alone.
export function templateDistance(template: MusicTemplate, frames: MusicFrame[]): number {
  if (template.version === 1) return musicDistance(template.frames, frames);
  return template.examples.map(example => musicDistance(example, frames)).sort((a,b) => a-b)[1];
}
export function calibrate(examples: MusicFrame[][]): Calibration {
  if (examples.length < 3 || examples.length > MAX_EXAMPLES || !examples.every(example => validFrames(example)))
    return {scores: [], error: '유효한 패턴을 최소 3회, 최대 8회 입력해 주세요. 5회를 권장합니다.'};
  // Leave one recording out; use exactly the same second-nearest rule as detection.
  const scores = examples.map((example, i) => examples.filter((_,j) => i !== j)
    .map(other => musicDistance(other, example)).sort((a,b) => a-b)[1]);
  const worst = Math.max(...scores);
  if (!Number.isFinite(worst) || worst > .20) {
    const outliers = scores.flatMap((score,i) => score > .20 ? [i+1] : []);
    return {scores, error: `${outliers.join(', ')}번 구간이 다른 샘플과 잘 맞지 않습니다. 같은 음과 비슷한 길이로 다시 등록해 주세요.`};
  }
  // A bounded margin is a heuristic, not an estimated false-positive probability.
  return {scores, template: {version: 2, examples, threshold: Math.max(.10, Math.min(.24, worst * 1.15 + .03))}};
}
/** Identical pause segmentation for enrollment and live detection. Times are PCM-relative. */
export class MusicSegmenter {
  private frames: MusicFrame[] = [];
  private quietFrames = 0;
  private peak = 0;
  private index = 0;
  private start = 0;
  private noise = .001;
  private ambient: number[] = [];
  private tail = 0;
  rejected = 0;
  push(frame: MusicFrame): MusicSegment | undefined {
    const index = this.index++;
    // Learn the room level instead of requiring a fixed near-silence threshold.
    // Initial 8 frames (~0.3 s including the analysis window) are room calibration.
    if (this.ambient.length < 8) {
      this.ambient.push(frame.rms);
      if (this.ambient.length === 8) this.noise = Math.max(.001, [...this.ambient].sort((a,b)=>a-b)[2]);
      return;
    }
    if (!this.frames.length) {
      const onset = Math.max(.005, this.noise * 2.2, this.tail * 1.6);
      if (frame.rms < onset) {
        // Track changes slowly; a decaying string tail must not become a new event.
        this.noise += (frame.rms - this.noise) * (frame.rms < this.noise ? .1 : .005);
        this.tail *= .96;
        return;
      }
      this.start = index;
    }
    this.frames.push(frame); this.peak = Math.max(this.peak, frame.rms);
    // A substantial drop, not absolute silence: 20% of the event peak or room floor.
    const release = Math.max(.004, this.noise * 1.6, this.peak * .20);
    this.quietFrames = frame.rms < release ? this.quietFrames + 1 : 0;
    if (this.quietFrames >= 7) return this.close(this.quietFrames);
    if (this.frames.length > 140) {
      this.rejected++; this.tail = this.peak; this.frames = []; this.peak = 0; this.quietFrames = 0;
    }
  }
  private close(trimTail: number): MusicSegment | undefined {
    const original = this.frames;
    const content = original.slice(0, original.length - trimTail);
    const trimmed = trimMusic(content);
    const offset = trimmed.length ? original.indexOf(trimmed[0]) : 0;
    const start = (this.start + offset) * HOP_SECONDS;
    this.tail = Math.max(this.noise, ...original.slice(-7).map(frame=>frame.rms));
    this.frames = []; this.quietFrames = 0; this.peak = 0;
    if (!validFrames(trimmed)) { this.rejected++; return; }
    return {frames: trimmed, start, end: start + (trimmed.length - 1) * HOP_SECONDS + .064};
  }
  // The user's Finish action is an explicit boundary. Keep a valid final sample
  // for review instead of rejecting the entire enrollment for a missing pause.
  finish(): MusicSegment | undefined {
    return this.frames.length ? this.close(this.quietFrames) : undefined;
  }
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
