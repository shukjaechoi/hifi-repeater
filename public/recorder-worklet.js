class Recorder extends AudioWorkletProcessor {
  constructor() {
    super(); this.buffer = new Float32Array(4096); this.offset = 0; this.active = false;
    this.port.onmessage = ({data}) => {
      if (data === 'start' || data === 'stop') {
        if (this.offset) this.port.postMessage({samples: this.buffer.slice(0, this.offset), recording: this.active});
        this.offset = 0; this.active = data === 'start';
        if (data === 'stop') this.port.postMessage({done: true});
      }
    };
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input) {
      for (const sample of input) {
        this.buffer[this.offset++] = sample;
        if (this.offset === this.buffer.length) {
          this.port.postMessage({samples: this.buffer, recording: this.active}, [this.buffer.buffer]);
          this.buffer = new Float32Array(4096); this.offset = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('pcm-recorder', Recorder);
