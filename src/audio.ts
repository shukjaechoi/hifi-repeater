import { concatenate } from './audio-utils';
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
  async play(samples: Float32Array<ArrayBuffer>, rate: number, start: number, end: number, loop: boolean, ended: () => void) {
    this.stopPlayback();
    const ctx = await this.ready(), buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.copyToChannel(samples, 0);
    const source = ctx.createBufferSource(); source.buffer = buffer; source.connect(ctx.destination);
    source.loop = loop; source.loopStart = start; source.loopEnd = end;
    source.onended = () => { if (this.source === source) { this.source = undefined; source.disconnect(); ended(); } };
    this.source = source; this.started = ctx.currentTime; this.from = start; this.to = end; this.looping = loop;
    if (loop) source.start(0, start); else source.start(0, start, end - start);
  }
  position() {
    const elapsed = (this.context?.currentTime ?? 0) - this.started;
    return this.looping ? this.from + elapsed % (this.to - this.from) : Math.min(this.to, this.from + elapsed);
  }
  stopPlayback() { if (this.source) { this.source.onended = null; this.source.stop(); this.source.disconnect(); this.source = undefined; } }
}
