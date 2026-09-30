import type { Kiln } from "../gpu/kiln";
import type { Tensor } from "../gpu/tensor";
import { argmax, type ArgmaxOptions } from "../kernels/argmax";
import { attention } from "../kernels/attention";
import { gemvInt4, uploadInt4, type Int4Tensor } from "../kernels/gemv-int4";
import { quantizeInt4 } from "../quant/int4";
import type { Epilogue } from "../kernels/epilogue";
import { layerNorm } from "../kernels/layernorm";
import { embed, im2col, splitHeads, splitQkv } from "../kernels/layout";
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

/**
 * Self-attention with Q, K and V fused into one [3d, d] projection: one
 * matmul/GEMV instead of three, then one splitQkv to unpack it.
 */
interface SelfAttn {
  qkv: Linear;
  o: Linear;
}

/** Cross-attention: K/V come from the encoder output, once per clip. */
interface CrossAttn {
  q: Linear;
  k: Linear; // Whisper's key projections have no bias
  v: Linear;
  o: Linear;
}

interface EncoderLayer {
  attnNorm: Norm;
  attn: SelfAttn;
  mlpNorm: Norm;
  fc1: Linear;
  fc2: Linear;
}

interface DecoderLayer extends EncoderLayer {
  crossNorm: Norm;
  cross: CrossAttn;
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
  // every GPU tensor the model owns, so destroy() can free them all
  const owned: Tensor[] = [];
  const own = <T extends Tensor>(t: T): T => {
    owned.push(t);
    return t;
  };
  const raw = (name: string) => {
    const t = tensors.get(name);
    if (!t) throw new Error(`whisper: missing weight ${name}`);
    return t;
  };
  const get = (name: string, shape?: number[]): Tensor => {
    const t = raw(name);
    return own(kiln.tensor(toF32(t), shape ?? t.shape, "f32", name));
  };
  const opt = (name: string) => (tensors.has(name) ? get(name) : undefined);
  const quantize = (data: Float32Array, N: number, K: number, label: string): Int4Tensor => {
    const q = uploadInt4(kiln, quantizeInt4(data, N, K), label);
    own(q.packed);
    own(q.scales);
    return q;
  };
  const int4 = (name: string): Int4Tensor => {
    const t = raw(name);
    return quantize(toF32(t), t.shape[0], t.shape[1], name);
  };
  const q4 = opts.quantize === "int4";
  const linear = (p: string, shape?: number[]): Linear => ({ w: get(`${p}.weight`, shape), b: opt(`${p}.bias`) });
  /** decoder linears: int4 when quantizing */
  const decLinear = (p: string): Linear => (q4 ? { q4: int4(`${p}.weight`), b: opt(`${p}.bias`) } : linear(p));
  const norm = (p: string): Norm => ({ w: get(`${p}.weight`), b: get(`${p}.bias`) });
  /**
   * q_proj, k_proj, v_proj stacked into one [3d, d] weight (+ [3d] bias, with
   * zeros where k_proj has none). int4 groups run along rows, so quantizing
   * the stack is the same as quantizing each part.
   */
  const fusedQkv = (p: string, int4: boolean): Linear => {
    const parts = ["q", "k", "v"].map((n) => toF32(raw(`${p}.${n}_proj.weight`)));
    const [N, K] = raw(`${p}.q_proj.weight`).shape;
    const w = new Float32Array(3 * N * K);
    parts.forEach((part, i) => w.set(part, i * N * K));
    const b = new Float32Array(3 * N);
    ["q", "k", "v"].forEach((n, i) => {
      const name = `${p}.${n}_proj.bias`;
      if (tensors.has(name)) b.set(toF32(raw(name)), i * N);
    });
    const bias = own(kiln.tensor(b, [3 * N], "f32", `${p}.qkv.bias`));
    if (int4) return { q4: quantize(w, 3 * N, K, `${p}.qkv`), b: bias };
    return { w: own(kiln.tensor(w, [3 * N, K], "f32", `${p}.qkv.weight`)), b: bias };
  };
  const selfAttn = (p: string, int4: boolean): SelfAttn => ({
    qkv: fusedQkv(p, int4),
    o: int4 ? decLinear(`${p}.out_proj`) : linear(`${p}.out_proj`),
  });
  const d = config.d_model;
  const encLayers = Array.from({ length: config.encoder_layers }, (_, i) => {
    const p = `model.encoder.layers.${i}`;
    return {
      attnNorm: norm(`${p}.self_attn_layer_norm`),
      attn: selfAttn(`${p}.self_attn`, false),
      mlpNorm: norm(`${p}.final_layer_norm`),
      fc1: linear(`${p}.fc1`),
      fc2: linear(`${p}.fc2`),
    };
  });
  const decLayers = Array.from({ length: config.decoder_layers }, (_, i) => {
    const p = `model.decoder.layers.${i}`;
    return {
      attnNorm: norm(`${p}.self_attn_layer_norm`),
      attn: selfAttn(`${p}.self_attn`, q4),
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
  }, owned);
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
    private readonly owned: Tensor[] = [],
  ) {}

  /** Free every weight on the GPU (e.g. before loading another precision). */
  destroy(): void {
    for (const t of this.owned) t.destroy();
    this.owned.length = 0;
    this.kiln.trimPool();
  }

  /** Bytes of weights this model holds on the GPU. */
  get weightBytes(): number {
    return this.owned.reduce((a, t) => a + t.byteLength, 0);
  }

  /** x @ W^T + b, optionally with an activation fused into the store. */
  private linear(x: Tensor, l: Linear, activation: Epilogue = "none"): Tensor {
    if (l.q4) return gemvInt4(this.kiln, x, l.q4, { bias: l.b, activation });
    return matmul(this.kiln, x, l.w!, { transposeB: true, bias: l.b, activation });
  }

  /**
   * Residual add + LayerNorm in one kernel: returns the new residual stream
   * x + delta and LN(x + delta). Every residual add in Whisper is followed by
   * a LayerNorm (the next block's, or the final one), so none of them needs
   * a dispatch of its own. Without a delta it's a plain LayerNorm.
   */
  private addNorm(x: Tensor, delta: Tensor | undefined, n: Norm): { x: Tensor; h: Tensor } {
    if (!delta) return { x, h: layerNorm(this.kiln, x, n.w, n.b) };
    const sum = this.kiln.empty(x.shape);
    const h = layerNorm(this.kiln, delta, n.w, n.b, { residual: x, sumOut: sum });
    return { x: sum, h };
  }

  private mlp(x: Tensor, l: EncoderLayer): Tensor {
    // GELU runs in fc1's epilogue: the 4x-wide hidden layer is written once,
    // already activated
    return this.linear(this.linear(x, l.fc1, "gelu"), l.fc2);
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
    x = this.linear(x, w.conv1, "gelu");
    x = im2col(kiln, x, { C: d, T, kernel: 3, stride: 2, pad: 1, layout: "TC" });
    x = this.linear(x, w.conv2, "gelu");

    // `delta` is what still has to be added to the residual stream x; the
    // next addNorm does it. The positional embedding is the first one.
    let delta: Tensor | undefined = w.encPos;
    let h: Tensor;
    for (const l of w.encLayers) {
      ({ x, h } = this.addNorm(x, delta, l.attnNorm));
      // q stays token-major [T, d]; attention reads it and writes its output
      // in that layout too, so out_proj takes it directly
      const { q, k, v } = splitQkv(kiln, this.linear(h, l.attn.qkv), H);
      delta = this.linear(attention(kiln, q, k, v, { layout: "tokens" }), l.attn.o);
      ({ x, h } = this.addNorm(x, delta, l.mlpNorm));
      delta = this.mlp(h, l);
    }
    return this.addNorm(x, delta, w.encNorm).h;
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

  /**
   * `steps` greedy steps with a single readback at the end. `tokens` go in
   * first (the prompt, or the last token picked); after that each step's
   * input is the previous step's argmax, which never leaves the GPU - the
   * embedding reads it straight from the id buffer. A step-at-a-time loop
   * waits a full CPU<->GPU round trip per token; this waits once per call.
   *
   * The catch: the GPU can't stop at end-of-text, so steps after it are
   * wasted work (and their tokens junk the caller drops). Returns the
   * `steps` picks in order.
   */
  decodeGreedy(state: DecodeState, tokens: number[], steps: number, choice: ArgmaxOptions = {}): Promise<number[]> {
    const { kiln } = this;
    const n = tokens.length;
    if (state.length + n + steps - 1 > this.config.max_target_positions) throw new Error("whisper: decoder context is full");
    return kiln.scope(async () => {
      // ids = [fed tokens (n)..., pick 0, pick 1, ...]: pick k at n + k
      const ids = kiln.tensor([...tokens, ...new Array(steps).fill(0)], [n + steps], "u32");
      // step 0 embeds the fed tokens; its argmax yields one pick per row,
      // and only the last row's is a prediction. Writing all n at offset 1
      // puts that one at index n (the rest overwrite fed tokens the embed
      // has already consumed - kernels run in order).
      argmax(kiln, this.logitsFrom(state, ids, { idOffset: 0, n }), { ...choice, out: ids, offset: 1 });
      // step s embeds pick s-1 and writes pick s
      for (let s = 1; s < steps; s++) {
        argmax(kiln, this.logitsFrom(state, ids, { idOffset: n + s - 1, n: 1 }), { ...choice, out: ids, offset: n + s });
      }
      return Array.from(await ids.read()).slice(n, n + steps);
    });
  }

  /** Decoder forward pass; returns logits [n, vocab] (still on the GPU). */
  private logits(state: DecodeState, tokens: number[]): Tensor {
    return this.logitsFrom(state, this.kiln.tensor(tokens, [tokens.length], "u32"), { idOffset: 0, n: tokens.length });
  }

  /** Same, embedding ids[idOffset .. idOffset+n) of a GPU id buffer. */
  private logitsFrom(state: DecodeState, ids: Tensor, { idOffset, n }: { idOffset: number; n: number }): Tensor {
    const { kiln, w, config } = this;
    const H = config.decoder_attention_heads;
    if (state.length + n > config.max_target_positions) throw new Error("whisper: decoder context is full");

    let x = embed(kiln, w.tokEmb, w.decPos, ids, state.length, { idOffset, n });
    // pending residual update, applied by the next addNorm (see encode)
    let delta: Tensor | undefined;
    let h: Tensor;
    w.decLayers.forEach((l, i) => {
      // causal self-attention: append this step's K/V to the cache first
      ({ x, h } = this.addNorm(x, delta, l.attnNorm));
      const { q } = splitQkv(kiln, this.linear(h, l.attn.qkv), H, {
        k: state.selfK[i],
        v: state.selfV[i],
        rowOffset: state.length,
      });
      const a = attention(kiln, q, state.selfK[i], state.selfV[i], {
        causal: true,
        kvLength: state.length + n,
        layout: "tokens",
      });
      delta = this.linear(a, l.attn.o);

      // cross-attention over the encoder output
      ({ x, h } = this.addNorm(x, delta, l.crossNorm));
      const c = attention(kiln, this.linear(h, l.cross.q), state.crossK[i], state.crossV[i], { layout: "tokens" });
      delta = this.linear(c, l.cross.o);

      ({ x, h } = this.addNorm(x, delta, l.mlpNorm));
      delta = this.mlp(h, l);
    });
    state.length += n;

    // logits through the tied token embedding: [n, d] @ [vocab, d]^T
    const out = this.addNorm(x, delta, w.decNorm).h;
    if (w.tokEmbQ4) return gemvInt4(kiln, out, w.tokEmbQ4);
    return matmul(kiln, out, w.tokEmb, { transposeB: true });
  }
}
