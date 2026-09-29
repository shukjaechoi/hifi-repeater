/** Conservative split-band de-esser. Operates on a copy; timing and PCM remain intact. */
export function reduceStringSqueak(samples: Float32Array, rate: number): Float32Array<ArrayBuffer> {
  if (!Number.isFinite(rate) || rate <= 0) throw new RangeError('Invalid sample rate');
  const output = new Float32Array(samples.length);
  // Unity-peak bandpass centred on the typical squeak region. Subtracting a
  // controlled portion makes a dynamic notch, with exact unity when inactive.
  const frequency = Math.min(4500, rate * 0.2);
  const omega = 2 * Math.PI * frequency / rate;
  const alpha = Math.sin(omega) / (2 * 0.8);
  const b0 = alpha / (1 + alpha), b2 = -b0;
  const a1 = -2 * Math.cos(omega) / (1 + alpha), a2 = (1 - alpha) / (1 + alpha);
  const envelope = Math.exp(-1 / (rate * 0.008));
  const attack = Math.exp(-1 / (rate * 0.005));
  const release = Math.exp(-1 / (rate * 0.100));
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  let bandPower = 0, totalPower = 0, reduction = 0, peak = 0;
  for (let i = 0; i < samples.length; i++) {
    const x = samples[i];
    const band = b0 * x + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x; y2 = y1; y1 = band;
    bandPower = envelope * bandPower + (1 - envelope) * band * band;
    totalPower = envelope * totalPower + (1 - envelope) * x * x;
    // Relative detection adapts to recording level; the -45 dBFS floor keeps
    // quiet room noise from driving it. Limit band attenuation to about 6 dB.
    const prominence = Math.max(0, Math.min(1, (bandPower / Math.max(totalPower, 1e-12) - 0.35) / 0.4));
    const audibility = Math.max(0, Math.min(1, (bandPower - 0.0000316) / 0.0001));
    const target = 0.5 * prominence * audibility;
    const smoothing = target > reduction ? attack : release;
    reduction = smoothing * reduction + (1 - smoothing) * target;
    output[i] = x - reduction * band;
    peak = Math.max(peak, Math.abs(output[i]));
  }
  // No makeup gain: it would undo reduction. Only guard against overload.
  if (peak > 1) for (let i = 0; i < output.length; i++) output[i] /= peak;
  return output;
}
