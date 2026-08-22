import { decodeWav, resample } from "./wav";

/**
 * Any audio the browser can decode (WAV, MP3, M4A/AAC, OGG/Opus, FLAC, a
 * MediaRecorder blob...) -> 16 kHz mono f32.
 *
 * Uses Web Audio's decoder, then an OfflineAudioContext at 16 kHz to mix
 * down and resample properly (band-limited, unlike the linear resampler in
 * wav.ts). Falls back to the built-in WAV decoder where Web Audio isn't
 * available (workers, tests in Node).
 */
export async function decodeAudio(data: ArrayBuffer, sampleRate = 16000): Promise<Float32Array> {
  if (typeof OfflineAudioContext === "undefined") {
    return resample(decodeWav(data), sampleRate);
  }
  // decodeAudioData needs some context; its own rate doesn't matter here
  const ctx = new OfflineAudioContext(1, 1, sampleRate);
  const decoded = await ctx.decodeAudioData(data.slice(0)); // it detaches the buffer
  const frames = Math.ceil(decoded.duration * sampleRate);
  const offline = new OfflineAudioContext(1, Math.max(1, frames), sampleRate);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination); // 1 output channel: Web Audio mixes down to mono
  src.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}
