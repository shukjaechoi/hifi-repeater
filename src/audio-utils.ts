export function concatenate(chunks: Float32Array[]): Float32Array<ArrayBuffer> {
  const result = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
  return result;
}
export function validRange(start: number, end: number, duration: number): [number, number] {
  const a = Math.max(0, Math.min(start, Math.max(0, duration - .02)));
  return [a, Math.min(duration, Math.max(a + .02, end))];
}

/**
 * Finds the first sustained sound above the opening room-noise floor. The
 * returned offset keeps a small lead-in, and never alters the PCM original.
 */
export function leadingSoundOffset(samples: Float32Array, rate: number): number | null {
  const frame = Math.max(1, Math.round(rate * 0.01));
  const openingFrames = Math.min(12, Math.floor(samples.length / frame));
  if (openingFrames < 3) return null;
  let noise = 0;
  for (let i = 0; i < openingFrames; i++) {
    let sum = 0;
    for (let j = i * frame; j < (i + 1) * frame; j++) sum += (samples[j] ?? 0) ** 2;
    noise += Math.sqrt(sum / frame);
  }
  const threshold = Math.max(0.008, noise / openingFrames * 4);
  let sustained = 0;
  for (let i = 0; i < Math.floor(samples.length / frame); i++) {
    let sum = 0;
    for (let j = i * frame; j < (i + 1) * frame; j++) sum += (samples[j] ?? 0) ** 2;
    if (Math.sqrt(sum / frame) >= threshold) sustained++; else sustained = 0;
    if (sustained >= 3) return Math.max(0, (i - sustained + 1) * frame / rate - 0.02);
  }
  return null;
}
export function wav(samples: Float32Array, rate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
  const text = (offset: number, s: string) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((v, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * (v < 0 ? 32768 : 32767), true));
  return buffer;
}
