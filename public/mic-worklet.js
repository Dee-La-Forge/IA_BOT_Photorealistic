// Converts microphone audio to 16 kHz mono 16-bit PCM and posts it in small blocks.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;
    this.position = 0;
    this.block = new Int16Array(1024);
    this.filled = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    // Linear resampling; `position` carries the fractional read point across calls.
    while (this.position < input.length) {
      const i = Math.floor(this.position);
      const next = i + 1 < input.length ? input[i + 1] : input[i];
      const sample = input[i] + (next - input[i]) * (this.position - i);
      this.block[this.filled++] = Math.max(-1, Math.min(1, sample)) * 32767;
      if (this.filled === this.block.length) {
        this.port.postMessage(this.block.slice(0).buffer);
        this.filled = 0;
      }
      this.position += this.step;
    }
    this.position -= input.length;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
