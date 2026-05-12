// Runs inside the AudioWorkletGlobalScope. The main-thread AudioContext feeds
// mic samples at the device rate (typically 44.1 or 48 kHz); we resample to
// 16 kHz mono with linear interpolation and post Float32Array chunks back.
//
// Linear interp is not a great anti-aliasing filter, but for speech captured
// on consumer mics at 48 kHz → 16 kHz (ratio 3) the audible aliasing is
// negligible, and voice content above 8 kHz carries almost no energy.

class DownsamplerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const out = options?.processorOptions?.outRate ?? 16000;
    this.ratio = sampleRate / out; // sampleRate is global in worklet scope
    this.buffer = new Float32Array(0);
    this.pos = 0; // fractional read index into this.buffer
  }

  process(inputs) {
    const input = inputs[0];
    const ch = input?.[0];
    if (!ch || ch.length === 0) return true;

    const merged = new Float32Array(this.buffer.length + ch.length);
    merged.set(this.buffer);
    merged.set(ch, this.buffer.length);
    this.buffer = merged;

    // Need at least this.pos + 1 samples to interpolate. Consume greedily.
    const out = [];
    while (this.pos + 1 < this.buffer.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      out.push(this.buffer[i] * (1 - frac) + this.buffer[i + 1] * frac);
      this.pos += this.ratio;
    }

    if (out.length > 0) {
      const buf = new Float32Array(out);
      this.port.postMessage(buf, [buf.buffer]);
      // Drop consumed samples; keep the tail we'll interpolate from next time.
      const keep = Math.floor(this.pos);
      if (keep > 0) {
        this.buffer = this.buffer.slice(keep);
        this.pos -= keep;
      }
    }

    return true;
  }
}

registerProcessor("downsampler", DownsamplerProcessor);
