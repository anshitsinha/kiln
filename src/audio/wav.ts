/**
 * WAV decoding to mono f32 in [-1, 1]. Handles 8/16/24/32-bit PCM and 32-bit
 * float, any channel count (averaged to mono). Anything not at 16 kHz is
 * resampled, since that's what Whisper expects.
 */

export interface Audio {
  samples: Float32Array;
  sampleRate: number;
}

export class WavError extends Error {}

export function decodeWav(buffer: ArrayBuffer): Audio {
  const view = new DataView(buffer);
  const tag = (o: number) => String.fromCharCode(...new Uint8Array(buffer, o, 4));
  if (buffer.byteLength < 12 || tag(0) !== "RIFF" || tag(8) !== "WAVE") throw new WavError("not a RIFF/WAVE file");

  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let data: DataView | null = null;

  // walk the chunks; "fmt " and "data" are the only ones we need
  for (let o = 12; o + 8 <= buffer.byteLength; ) {
    const id = tag(o);
    const size = view.getUint32(o + 4, true);
    const body = o + 8;
    if (id === "fmt ") {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
      // WAVE_FORMAT_EXTENSIBLE: the real format is the first 2 bytes of the GUID
      if (format === 0xfffe) format = view.getUint16(body + 24, true);
    } else if (id === "data") {
      data = new DataView(buffer, body, Math.min(size, buffer.byteLength - body));
    }
    o = body + size + (size & 1); // chunks are padded to even sizes
  }
  if (!data || !channels) throw new WavError("missing fmt or data chunk");
  if (format !== 1 && format !== 3) throw new WavError(`unsupported WAV format ${format} (want PCM or float)`);

  const bytes = bits / 8;
  const frames = Math.floor(data.byteLength / (bytes * channels));
  const read = (o: number): number => {
    if (format === 3) return bits === 64 ? data!.getFloat64(o, true) : data!.getFloat32(o, true);
    switch (bits) {
      case 8:
        return (data!.getUint8(o) - 128) / 128;
      case 16:
        return data!.getInt16(o, true) / 32768;
      case 24: {
        const v = data!.getUint8(o) | (data!.getUint8(o + 1) << 8) | (data!.getInt8(o + 2) << 16);
        return v / 8388608;
      }
      case 32:
        return data!.getInt32(o, true) / 2147483648;
      default:
        throw new WavError(`unsupported bit depth ${bits}`);
    }
  };

  const mono = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += read((f * channels + c) * bytes);
    mono[f] = sum / channels;
  }
  return { samples: mono, sampleRate };
}

/**
 * Resample with linear interpolation. Not studio quality, but speech
 * recognition doesn't care about the top octave, and it keeps this file free
 * of dependencies. 16 kHz input (the common case) passes straight through.
 */
export function resample(audio: Audio, rate = 16000): Float32Array {
  if (audio.sampleRate === rate) return audio.samples;
  const ratio = audio.sampleRate / rate;
  const out = new Float32Array(Math.floor(audio.samples.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, audio.samples.length - 1);
    out[i] = audio.samples[i0] + (audio.samples[i1] - audio.samples[i0]) * (x - i0);
  }
  return out;
}

/** Build a 16-bit PCM WAV (tests and fixtures). */
export function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => v.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(s * 32767))), true));
  return buf;
}
