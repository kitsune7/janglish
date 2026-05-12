// Encodes a Float32 PCM buffer (mono, 16 kHz) into a 16-bit little-endian WAV
// Blob with an audio/wav content-type. Format matches what the Worker expects
// and what `ffmpeg -ar 16000 -ac 1 -sample_fmt s16 -c:a flac` ingests cleanly.

export function encodeWav(samples, sampleRate) {
  const pcm = floatToInt16(samples);
  const header = wavHeader(pcm.length, sampleRate);
  return new Blob([header, pcm.buffer], { type: "audio/wav" });
}

function floatToInt16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function wavHeader(sampleCount, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataBytes = sampleCount * 2;

  const buf = new ArrayBuffer(44);
  const v = new DataView(buf);
  writeString(v, 0, "RIFF");
  v.setUint32(4, 36 + dataBytes, true);
  writeString(v, 8, "WAVE");
  writeString(v, 12, "fmt ");
  v.setUint32(16, 16, true);           // fmt chunk size
  v.setUint16(20, 1, true);            // PCM
  v.setUint16(22, numChannels, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, byteRate, true);
  v.setUint16(32, blockAlign, true);
  v.setUint16(34, bitsPerSample, true);
  writeString(v, 36, "data");
  v.setUint32(40, dataBytes, true);
  return buf;
}

function writeString(view, offset, s) {
  for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
}
