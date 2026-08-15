import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { argmax, type ArgmaxOptions } from "../kernels/argmax";
import { attention } from "../kernels/attention";
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

interface Linear {
  w: Tensor; // [out, in]
  b?: Tensor; // [out]
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
  decPos: Tensor;
  decLayers: DecoderLayer[];
  decNorm: Norm;
}

/** Upload every tensor in a safetensors file and wire them into the model. */
export function loadWhisper(kiln: Kiln, file: ArrayBuffer, config: WhisperConfig): Whisper {
  const tensors = parseSafetensors(file);
  const get = (name: string, shape?: number[]): Tensor => {
    const t = tensors.get(name);
    if (!t) throw new Error(`whisper: missing weight ${name}`);
    return kiln.tensor(toF32(t), shape ?? t.shape, "f32", name);
  };
  const opt = (name: string) => (tensors.has(name) ? get(name) : undefined);
  const linear = (p: string, shape?: number[]): Linear => ({ w: get(`${p}.weight`, shape), b: opt(`${p}.bias`) });
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
      attn: attn(`${p}.self_attn`),
      crossNorm: norm(`${p}.encoder_attn_layer_norm`),
      cross: attn(`${p}.encoder_attn`),
      mlpNorm: norm(`${p}.final_layer_norm`),
      fc1: linear(`${p}.fc1`),
      fc2: linear(`${p}.fc2`),
    };
  });
  return new Whisper(kiln, config, {
    conv1: linear("model.encoder.conv1", [d, config.num_mel_bins * 3]),
    conv2: linear("model.encoder.conv2", [d, d * 3]),
    encPos: get("model.encoder.embed_positions.weight"),
    encLayers,
    encNorm: norm("model.encoder.layer_norm"),
    tokEmb: get("model.decoder.embed_tokens.weight"),
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

  private linear(x: Tensor, l: Linear): Tensor {
    return matmul(this.kiln, x, l.w, { transposeB: true, bias: l.b });
  }

  private norm(x: Tensor, n: Norm): Tensor {
    return layerNorm(this.kiln, x, n.w, n.b);
  }

  private mlp(x: Tensor, l: EncoderLayer): Tensor {
    // fc1 without bias, then bias + GELU fused in one pass
    const h = matmul(this.kiln, x, l.fc1.w, { transposeB: true });
    return this.linear(activation(this.kiln, h, "gelu", { bias: l.fc1.b }), l.fc2);
  }

  /** mel: [80, 3000] -> encoder output [1500, d]. Intermediates are freed. */
  encode(mel: Float32Array): Promise<Tensor> {
    return this.kiln.scope(() => this.encodeInner(mel));
  }

  private encodeInner(mel: Float32Array): Tensor {
    const { kiln, w, config } = this;
    const d = config.d_model;
    const T = mel.length / config.num_mel_bins;
    const H = config.encoder_attention_heads;

    let x = im2col(kiln, kiln.tensor(mel, [config.num_mel_bins, T]), {
      C: config.num_mel_bins,
      T,
      kernel: 3,
      stride: 1,
      pad: 1,
      layout: "CT",
    });
    x = activation(kiln, matmul(kiln, x, w.conv1.w, { transposeB: true }), "gelu", { bias: w.conv1.b });
    x = im2col(kiln, x, { C: d, T, kernel: 3, stride: 2, pad: 1, layout: "TC" });
    x = activation(kiln, matmul(kiln, x, w.conv2.w, { transposeB: true }), "gelu", { bias: w.conv2.b });
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
    return matmul(kiln, this.norm(x, w.decNorm), w.tokEmb, { transposeB: true });
  }
}
