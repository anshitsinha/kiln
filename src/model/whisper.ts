import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { argmax, type ArgmaxOptions } from "../kernels/argmax";
import { attention } from "../kernels/attention";
import { gemvInt4, uploadInt4, type Int4Tensor } from "../kernels/gemv-int4";
import { quantizeInt4 } from "../quant/int4";
import { activation, add } from "../kernels/elementwise";
import { layerNorm } from "../kernels/layernorm";
import { embed, im2col, mergeHeads, splitHeads } from "../kernels/layout";
import { matmul } from "../kernels/matmul";
import { parseSafetensors, toF32 } from "./safetensors";

/**
 * Whisper (encoder-decoder transformer for speech recognition), built from
 * kiln's kernels. Activations are [tokens, channels] throughout.
 *
 *   mel [80, 3000]
 *    -> conv1 (k3, s1) + GELU -> conv2 (k3, s2) + GELU -> + positions   [1500, d]
 *    -> N x (LN -> self-attn -> +res, LN -> MLP -> +res) -> LN          encoder out
 *
 *   tokens -> embed + positions
 *    -> N x (LN -> causal self-attn (KV cache) -> +res,
 *            LN -> cross-attn over encoder out -> +res,
 *            LN -> MLP -> +res) -> LN -> logits via the tied token embedding
 */

export interface WhisperConfig {
  d_model: number;
  encoder_layers: number;
  decoder_layers: number;
  encoder_attention_heads: number;
  decoder_attention_heads: number;
  num_mel_bins: number;
  max_source_positions: number;
  max_target_positions: number;
  vocab_size: number;
}

/**
 * A Linear layer: f32 weight, or an int4 one (decoder only - its inputs are
 * a few rows at a time, which is exactly what the int4 GEMV handles).
 */
interface Linear {
  w?: Tensor; // [out, in], f32
  q4?: Int4Tensor; // [out, in], int4
  b?: Tensor; // [out]
}

export interface LoadOptions {
  /**
   * "int4": quantize the decoder's linear layers and the output projection
   * to group-wise int4 at load time. The encoder stays f32 - it processes
   * 1500 rows at once, so it's compute-bound and int4 GEMV doesn't apply.
   */
  quantize?: "int4";
}

interface Norm {
  w: Tensor;
  b: Tensor;
}

interface Attn {
  q: Linear;
  k: Linear; // Whisper's key projections have no bias
  v: Linear;
  o: Linear;
}

interface EncoderLayer {
  attnNorm: Norm;
  attn: Attn;
  mlpNorm: Norm;
  fc1: Linear;
  fc2: Linear;
}

interface DecoderLayer extends EncoderLayer {
  crossNorm: Norm;
  cross: Attn;
}

export interface WhisperWeights {
  conv1: Linear; // weight viewed as [d, 80*3]
  conv2: Linear; // [d, d*3]
  encPos: Tensor;
  encLayers: EncoderLayer[];
  encNorm: Norm;
  tokEmb: Tensor; // [vocab, d], also the output projection
  tokEmbQ4?: Int4Tensor; // int4 copy used for the logits when quantized
  decPos: Tensor;
  decLayers: DecoderLayer[];
  decNorm: Norm;
}

/** Upload every tensor in a safetensors file and wire them into the model. */
export function loadWhisper(kiln: Kiln, file: ArrayBuffer, config: WhisperConfig, opts: LoadOptions = {}): Whisper {
  const tensors = parseSafetensors(file);
  const raw = (name: string) => {
    const t = tensors.get(name);
    if (!t) throw new Error(`whisper: missing weight ${name}`);
    return t;
  };
  const get = (name: string, shape?: number[]): Tensor => {
    const t = raw(name);
    return kiln.tensor(toF32(t), shape ?? t.shape, "f32", name);
  };
  const opt = (name: string) => (tensors.has(name) ? get(name) : undefined);
  const int4 = (name: string): Int4Tensor => {
    const t = raw(name);
    const [N, K] = t.shape;
    return uploadInt4(kiln, quantizeInt4(toF32(t), N, K), name);
  };
  const q4 = opts.quantize === "int4";
  const linear = (p: string, shape?: number[]): Linear => ({ w: get(`${p}.weight`, shape), b: opt(`${p}.bias`) });
  /** decoder linears: int4 when quantizing */
  const decLinear = (p: string): Linear => (q4 ? { q4: int4(`${p}.weight`), b: opt(`${p}.bias`) } : linear(p));
  const norm = (p: string): Norm => ({ w: get(`${p}.weight`), b: get(`${p}.bias`) });
  const attn = (p: string): Attn => ({
    q: linear(`${p}.q_proj`),
    k: linear(`${p}.k_proj`),
    v: linear(`${p}.v_proj`),
    o: linear(`${p}.out_proj`),
  });
  const d = config.d_model;
  const encLayers = Array.from({ length: config.encoder_layers }, (_, i) => {
    const p = `model.encoder.layers.${i}`;
    return {
      attnNorm: norm(`${p}.self_attn_layer_norm`),
      attn: attn(`${p}.self_attn`),
      mlpNorm: norm(`${p}.final_layer_norm`),
      fc1: linear(`${p}.fc1`),
      fc2: linear(`${p}.fc2`),
    };
  });
  const decLayers = Array.from({ length: config.decoder_layers }, (_, i) => {
    const p = `model.decoder.layers.${i}`;
    return {
      attnNorm: norm(`${p}.self_attn_layer_norm`),
      attn: {
        q: decLinear(`${p}.self_attn.q_proj`),
        k: decLinear(`${p}.self_attn.k_proj`),
        v: decLinear(`${p}.self_attn.v_proj`),
        o: decLinear(`${p}.self_attn.out_proj`),
      },
      crossNorm: norm(`${p}.encoder_attn_layer_norm`),
      cross: {
        q: decLinear(`${p}.encoder_attn.q_proj`),
        // cross K/V are computed once from the 1500-row encoder output: a
        // big matmul, so they stay f32
        k: linear(`${p}.encoder_attn.k_proj`),
        v: linear(`${p}.encoder_attn.v_proj`),
        o: decLinear(`${p}.encoder_attn.out_proj`),
      },
      mlpNorm: norm(`${p}.final_layer_norm`),
      fc1: decLinear(`${p}.fc1`),
      fc2: decLinear(`${p}.fc2`),
    };
  });
  return new Whisper(kiln, config, {
    conv1: linear("model.encoder.conv1", [d, config.num_mel_bins * 3]),
    conv2: linear("model.encoder.conv2", [d, d * 3]),
    encPos: get("model.encoder.embed_positions.weight"),
    encLayers,
    encNorm: norm("model.encoder.layer_norm"),
    tokEmb: get("model.decoder.embed_tokens.weight"),
    // the embedding lookup needs the f32 table; the logits projection (the
    // single biggest read per token) gets an int4 copy
    tokEmbQ4: q4 ? int4("model.decoder.embed_tokens.weight") : undefined,
    decPos: get("model.decoder.embed_positions.weight"),
    decLayers,
    decNorm: norm("model.decoder.layer_norm"),
  });
}

/** Decoder state for one transcription: self-attn KV cache + cross-attn K/V. */
export interface DecodeState {
  selfK: Tensor[]; // per layer [H, maxTarget, D]
  selfV: Tensor[];
  crossK: Tensor[]; // per layer [H, 1500, D]
  crossV: Tensor[];
  length: number; // tokens already in the cache
}

export class Whisper {
  constructor(
    readonly kiln: Kiln,
    readonly config: WhisperConfig,
    readonly w: WhisperWeights,
  ) {}

  private linear(x: Tensor, l: Linear, withBias = true): Tensor {
    const bias = withBias ? l.b : undefined;
    if (l.q4) return gemvInt4(this.kiln, x, l.q4, { bias });
    return matmul(this.kiln, x, l.w!, { transposeB: true, bias });
  }

  private norm(x: Tensor, n: Norm): Tensor {
    return layerNorm(this.kiln, x, n.w, n.b);
  }

  private mlp(x: Tensor, l: EncoderLayer): Tensor {
    // fc1 without bias, then bias + GELU fused in one pass
    const h = this.linear(x, l.fc1, false);
    return this.linear(activation(this.kiln, h, "gelu", { bias: l.fc1.b }), l.fc2);
  }

  /**
   * Log-mel -> encoder output [1500, d]. Takes either the CPU spectrogram
   * ([80, 3000] array) or the GPU one ([3000, 80] tensor - already on the
   * device, no upload). Intermediates are freed.
   */
  encode(mel: Float32Array | Tensor): Promise<Tensor> {
    return this.kiln.scope(() => this.encodeInner(mel));
  }

  private encodeInner(mel: Float32Array | Tensor): Tensor {
    const { kiln, w, config } = this;
    const d = config.d_model;
    const nMels = config.num_mel_bins;
    const H = config.encoder_attention_heads;
    const onGpu = !(mel instanceof Float32Array);
    const T = (onGpu ? mel.size : mel.length) / nMels;
    const input = onGpu ? mel : kiln.tensor(mel, [nMels, T]);

    let x = im2col(kiln, input, { C: nMels, T, kernel: 3, stride: 1, pad: 1, layout: onGpu ? "TC" : "CT" });
    x = activation(kiln, this.linear(x, w.conv1, false), "gelu", { bias: w.conv1.b });
    x = im2col(kiln, x, { C: d, T, kernel: 3, stride: 2, pad: 1, layout: "TC" });
    x = activation(kiln, this.linear(x, w.conv2, false), "gelu", { bias: w.conv2.b });
    x = add(kiln, x, w.encPos);

    for (const l of w.encLayers) {
      const h = this.norm(x, l.attnNorm);
      const q = splitHeads(kiln, this.linear(h, l.attn.q), H);
      const k = splitHeads(kiln, this.linear(h, l.attn.k), H);
      const v = splitHeads(kiln, this.linear(h, l.attn.v), H);
      x = add(kiln, x, this.linear(mergeHeads(kiln, attention(kiln, q, k, v)), l.attn.o));
      x = add(kiln, x, this.mlp(this.norm(x, l.mlpNorm), l));
    }
    return this.norm(x, w.encNorm);
  }

  /**
   * Allocate the KV cache and precompute cross-attention K/V. The state's
   * tensors live until the enclosing scope ends (or freeDecode()).
   */
  startDecode(encoderOut: Tensor): Promise<DecodeState> {
    const { kiln, w, config } = this;
    const H = config.decoder_attention_heads;
    const D = config.d_model / H;
    // zeros, not empty: attention never reads past kvLength, but a cache full
    // of a pooled buffer's leftovers would make any such bug silent
    const cache = () => kiln.zeros([H, config.max_target_positions, D]);
    return kiln.scope(() => ({
      selfK: w.decLayers.map(cache),
      selfV: w.decLayers.map(cache),
      crossK: w.decLayers.map((l) => splitHeads(kiln, this.linear(encoderOut, l.cross.k), H)),
      crossV: w.decLayers.map((l) => splitHeads(kiln, this.linear(encoderOut, l.cross.v), H)),
      length: 0,
    }));
  }

  freeDecode(state: DecodeState): void {
    for (const t of [...state.selfK, ...state.selfV, ...state.crossK, ...state.crossV]) t.destroy();
  }

  /**
   * Run new tokens through the decoder, appending to the cache. Returns the
   * logits for the last token. Per-step intermediates are freed.
   */
  decode(state: DecodeState, tokens: number[]): Promise<Float32Array> {
    return this.kiln.scope(async () => {
      const n = tokens.length;
      const all = await this.logits(state, tokens).readF32();
      return all.slice((n - 1) * this.config.vocab_size, n * this.config.vocab_size);
    });
  }

  /**
   * Like decode(), but picks the greedy next token on the GPU and reads back
   * just its id (4 bytes instead of ~200 KB of logits). `limit`/`extra`
   * restrict the choice, e.g. to text tokens plus end-of-text.
   */
  decodeNext(state: DecodeState, tokens: number[], choice: ArgmaxOptions = {}): Promise<number> {
    return this.kiln.scope(async () => {
      const ids = await argmax(this.kiln, this.logits(state, tokens), choice).read();
      return ids[tokens.length - 1];
    });
  }

  /** Decoder forward pass; returns logits [n, vocab] (still on the GPU). */
  private logits(state: DecodeState, tokens: number[]): Tensor {
    const { kiln, w, config } = this;
    const H = config.decoder_attention_heads;
    const n = tokens.length;
    if (state.length + n > config.max_target_positions) throw new Error("whisper: decoder context is full");

    let x = embed(kiln, w.tokEmb, w.decPos, kiln.tensor(tokens, [n], "u32"), state.length);
    w.decLayers.forEach((l, i) => {
      // causal self-attention: append this step's K/V to the cache first
      const h = this.norm(x, l.attnNorm);
      const q = splitHeads(kiln, this.linear(h, l.attn.q), H);
      splitHeads(kiln, this.linear(h, l.attn.k), H, { out: state.selfK[i], rowOffset: state.length });
      splitHeads(kiln, this.linear(h, l.attn.v), H, { out: state.selfV[i], rowOffset: state.length });
      const a = attention(kiln, q, state.selfK[i], state.selfV[i], { causal: true, kvLength: state.length + n });
      x = add(kiln, x, this.linear(mergeHeads(kiln, a), l.attn.o));

      // cross-attention over the encoder output
      const hc = this.norm(x, l.crossNorm);
      const qc = splitHeads(kiln, this.linear(hc, l.cross.q), H);
      const c = attention(kiln, qc, state.crossK[i], state.crossV[i]);
      x = add(kiln, x, this.linear(mergeHeads(kiln, c), l.cross.o));

      x = add(kiln, x, this.mlp(this.norm(x, l.mlpNorm), l));
    });
    state.length += n;

    // logits through the tied token embedding: [n, d] @ [vocab, d]^T
    const h = this.norm(x, w.decNorm);
    if (w.tokEmbQ4) return gemvInt4(kiln, h, w.tokEmbQ4);
    return matmul(kiln, h, w.tokEmb, { transposeB: true });
  }
}
