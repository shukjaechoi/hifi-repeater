class Recorder extends AudioWorkletProcessor {
  constructor() {
    super(); this.buffer = new Float32Array(4096); this.offset = 0; this.active = true;
    this.port.onmessage = ({data}) => {
      if (data === 'stop') {
        this.active = false;
        if (this.offset) this.port.postMessage({samples: this.buffer.slice(0, this.offset)});
        this.port.postMessage({done: true});
      }
    };
  }
  process(inputs) {
    const input = inputs[0]?.[0];
    if (input && this.active) {
      for (const sample of input) {
        this.buffer[this.offset++] = sample;
        if (this.offset === this.buffer.length) {
          this.port.postMessage({samples: this.buffer}, [this.buffer.buffer]);
          this.buffer = new Float32Array(4096); this.offset = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('pcm-recorder', Recorder);
