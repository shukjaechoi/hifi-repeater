export type PlaybackFilter = 'none' | 'iphone-km184-hybrid-fir' | 'iphone-km184-iir';

export const iphoneKm184IirBands = [
  { frequency: 100, q: 0.75, gainDb: 3.3 },
  { frequency: 180, q: 0.9, gainDb: 4.7 },
  { frequency: 320, q: 0.95, gainDb: 3.0 },
  { frequency: 650, q: 0.9, gainDb: -0.7 },
  { frequency: 1300, q: 0.85, gainDb: -0.1 },
  { frequency: 2800, q: 1.0, gainDb: 0.6 },
] as const;

const firUrl = new URL('filters/iphone-to-km184-4097.f32le', document.baseURI);
let hybridFirPromise: Promise<Float32Array<ArrayBuffer>> | undefined;

export function loadHybridFir(): Promise<Float32Array<ArrayBuffer>> {
  hybridFirPromise ??= fetch(firUrl).then(async response => {
    if (!response.ok) throw new Error('FIR 필터 파일을 불러오지 못했습니다.');
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength !== 4097 * Float32Array.BYTES_PER_ELEMENT) throw new Error('FIR 필터 파일 크기가 올바르지 않습니다.');
    const view = new DataView(bytes);
    const taps = new Float32Array(4097);
    for (let i = 0; i < taps.length; i++) taps[i] = view.getFloat32(i * 4, true);
    return taps;
  }).catch(error => {
    hybridFirPromise = undefined;
    throw error;
  });
  return hybridFirPromise;
}

export const hybridFirCalibration = {
  inputPolarity: -1,
  inputDc: -1.556084416504529e-5,
  inputGain: 0.32236124172299196,
  sharedScale: 2.6776594562397706,
  targetDc: -1.3585624694824218e-5,
};
