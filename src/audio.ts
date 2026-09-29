import { concatenate } from './audio-utils';
import { hybridFirCalibration, iphoneKm184IirBands, loadHybridFir, type PlaybackFilter } from './filters';
export class AudioEngine {
  context?: AudioContext;
  stream?: MediaStream;
  node?: AudioWorkletNode;
  input?: MediaStreamAudioSourceNode;
  source?: AudioBufferSourceNode;
  chunks: Float32Array[] = [];
  started = 0;
  from = 0;
  to = 0;
  looping = false;
  stopping?: () => void;
  loaded = false;
  async ready() {
    this.context ??= new AudioContext();
    await this.context.resume();
    return this.context;
  }
  async record(onChunk: (chunk: Float32Array) => void, onInterrupted: () => void) {
    this.stopPlayback();
    const ctx = await this.ready();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
      if (!this.loaded) { await ctx.audioWorklet.addModule(new URL('recorder-worklet.js', document.baseURI)); this.loaded = true; }
      this.chunks = [];
      this.node = new AudioWorkletNode(ctx, 'pcm-recorder');
      this.node.port.onmessage = ({data}) => {
        if (data.samples) { this.chunks.push(data.samples); onChunk(data.samples); }
        if (data.done) this.stopping?.();
      };
      this.stream.getAudioTracks()[0].onended = onInterrupted;
      this.input = ctx.createMediaStreamSource(this.stream);
      this.input.connect(this.node); this.node.connect(ctx.destination);
      return this.stream.getAudioTracks()[0].getSettings();
    } catch (error) { this.release(); throw error; }
  }
  release() {
    this.stream?.getTracks().forEach(t => { t.onended = null; t.stop(); });
    this.input?.disconnect(); this.node?.disconnect(); this.stream = undefined; this.node = undefined;
  }
  async finish() {
    if (!this.node || !this.context) throw new Error('녹음 장치가 없습니다.');
    try {
      await new Promise<void>((resolve, reject) => {
        const timeout = window.setTimeout(() => reject(new Error('녹음 종료가 지연되었습니다. 다시 시도해 주세요.')), 3000);
        this.stopping = () => { clearTimeout(timeout); resolve(); };
        this.node!.port.postMessage('stop');
      });
      const samples = concatenate(this.chunks);
      this.chunks = [];
      return {samples, rate: this.context.sampleRate};
    } finally { this.release(); this.stopping = undefined; }
  }
  async play(samples: Float32Array<ArrayBuffer>, rate: number, start: number, end: number, loop: boolean, ended: () => void, filter: PlaybackFilter = 'none') {
    this.stopPlayback();
    const ctx = await this.ready();
    let playbackSamples = samples;
    if (filter === 'iphone-km184-hybrid-fir') playbackSamples = await this.applyHybridFir(samples, rate, start, end);
    else if (filter === 'iphone-km184-iir') playbackSamples = await this.applyIir(samples, rate, start, end);
    const buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(playbackSamples, 0);
    const source = ctx.createBufferSource(); source.buffer = buffer; source.connect(ctx.destination);
    source.loop = loop; source.loopStart = start; source.loopEnd = end;
    source.onended = () => { if (this.source === source) { this.source = undefined; source.disconnect(); ended(); } };
    this.source = source; this.started = ctx.currentTime; this.from = start; this.to = end; this.looping = loop;
    if (loop) source.start(0, start); else source.start(0, start, end - start);
  }
  private async applyHybridFir(samples: Float32Array<ArrayBuffer>, rate: number, start: number, end: number): Promise<Float32Array<ArrayBuffer>> {
    const taps = await loadHybridFir();
    let tapsAtRate = taps;
    if (rate !== 48000) {
      const halfLength = Math.round((taps.length - 1) * rate / 48000 / 2);
      const resampler = new OfflineAudioContext(1, halfLength * 2 + 1, rate);
      const originalImpulse = resampler.createBuffer(1, taps.length, 48000);
      originalImpulse.copyToChannel(taps, 0);
      const impulseSource = resampler.createBufferSource(); impulseSource.buffer = originalImpulse;
      impulseSource.connect(resampler.destination); impulseSource.start();
      tapsAtRate = (await resampler.startRendering()).getChannelData(0).slice();
    }
    const outputLength = samples.length + tapsAtRate.length - 1;
    const offline = new OfflineAudioContext(1, outputLength, rate);
    const input = offline.createBuffer(1, samples.length, rate);
    const calibrated = new Float32Array(samples.length);
    const calibration = hybridFirCalibration;
    const inputScale = calibration.inputGain * calibration.sharedScale;
    for (let i = 0; i < samples.length; i++) calibrated[i] = (samples[i] * calibration.inputPolarity - calibration.inputDc) * inputScale;
    input.copyToChannel(calibrated, 0);

    const impulse = offline.createBuffer(1, tapsAtRate.length, rate);
    impulse.copyToChannel(tapsAtRate, 0);
    const source = offline.createBufferSource(); source.buffer = input;
    const convolver = offline.createConvolver(); convolver.normalize = false;
    convolver.buffer = impulse;
    source.connect(convolver); convolver.connect(offline.destination); source.start();
    const rendered = await offline.startRendering();
    const centered = rendered.getChannelData(0).subarray((tapsAtRate.length - 1) / 2, (tapsAtRate.length - 1) / 2 + samples.length);
    const output = new Float32Array(samples.length);
    for (let i = 0; i < output.length; i++) output[i] = centered[i] / calibration.sharedScale + calibration.targetDc;

    return this.matchPlaybackLevel(samples, output, rate, start, end);
  }
  private async applyIir(samples: Float32Array<ArrayBuffer>, rate: number, start: number, end: number): Promise<Float32Array<ArrayBuffer>> {
    const offline = new OfflineAudioContext(1, samples.length, rate);
    const input = offline.createBuffer(1, samples.length, rate);
    input.copyToChannel(samples, 0);
    const source = offline.createBufferSource(); source.buffer = input;
    let previous: AudioNode = source;
    for (const band of iphoneKm184IirBands) {
      const filter = offline.createBiquadFilter();
      filter.type = 'peaking'; filter.frequency.value = band.frequency; filter.Q.value = band.q; filter.gain.value = band.gainDb;
      previous.connect(filter); previous = filter;
    }
    previous.connect(offline.destination); source.start();
    const output = (await offline.startRendering()).getChannelData(0).slice();
    return this.matchPlaybackLevel(samples, output, rate, start, end);
  }
  private matchPlaybackLevel(samples: Float32Array<ArrayBuffer>, output: Float32Array<ArrayBuffer>, rate: number, start: number, end: number) {
    const first = Math.max(0, Math.min(samples.length - 1, Math.floor(start * rate)));
    const last = Math.max(first + 1, Math.min(samples.length, Math.ceil(end * rate)));
    let sourcePower = 0, filteredPower = 0, filteredPeak = 0;
    for (let i = first; i < last; i++) {
      const sourceSample = samples[i] ?? 0, filteredSample = output[i] ?? 0;
      sourcePower += sourceSample * sourceSample;
      filteredPower += filteredSample * filteredSample;
      filteredPeak = Math.max(filteredPeak, Math.abs(filteredSample));
    }
    const sourceRms = Math.sqrt(sourcePower / (last - first));
    const filteredRms = Math.sqrt(filteredPower / (last - first));
    if (sourceRms > 1e-6 && filteredRms > 1e-6) {
      let gain = Math.max(0.2512, Math.min(3.9811, sourceRms / filteredRms));
      if (filteredPeak > 0) gain = Math.min(gain, 0.98 / filteredPeak);
      for (let i = 0; i < output.length; i++) output[i] *= gain;
    }
    return output;
  }
  position() {
    const elapsed = (this.context?.currentTime ?? 0) - this.started;
    return this.looping ? this.from + elapsed % (this.to - this.from) : Math.min(this.to, this.from + elapsed);
  }
  stopPlayback() { if (this.source) { this.source.onended = null; this.source.stop(); this.source.disconnect(); this.source = undefined; } }
}
