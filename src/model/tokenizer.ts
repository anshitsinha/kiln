/**
 * Decode-side tokenizer for Whisper (GPT-2 style byte-level BPE).
 *
 * Decoding doesn't need the BPE merges at all: each token id maps to a
 * string in the vocab, and the strings are bytes disguised as printable
 * characters (GPT-2's bytes_to_unicode, which moves control bytes and
 * whitespace up to U+0100+ so every byte has a visible symbol - "Ġ" is a
 * space). So: id -> string -> bytes -> UTF-8.
 */

export interface TokenizerJson {
  model: { vocab: Record<string, number> };
  added_tokens: { id: number; content: string; special: boolean }[];
}

/** GPT-2's byte <-> unicode-char table. */
function byteDecoder(): Map<string, number> {
  const bytes: number[] = [];
  const range = (a: number, b: number) => {
    for (let i = a; i <= b; i++) bytes.push(i);
  };
  range(0x21, 0x7e); // ! .. ~
  range(0xa1, 0xac); // ¡ .. ¬
  range(0xae, 0xff); // ® .. ÿ
  const chars = bytes.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + n++);
    }
  }
  return new Map(bytes.map((b, i) => [String.fromCharCode(chars[i]), b]));
}

export class Tokenizer {
  private idToToken: string[] = [];
  private special = new Set<number>();
  private bytesOf = byteDecoder();
  private ids = new Map<string, number>();

  constructor(json: TokenizerJson) {
    for (const [tok, id] of Object.entries(json.model.vocab)) {
      this.idToToken[id] = tok;
      this.ids.set(tok, id);
    }
    for (const t of json.added_tokens) {
      this.idToToken[t.id] = t.content;
      this.ids.set(t.content, t.id);
      if (t.special) this.special.add(t.id);
    }
  }

  /** Id of a token by its exact string, e.g. "<|en|>". */
  id(token: string): number {
    const id = this.ids.get(token);
    if (id === undefined) throw new Error(`tokenizer: no token ${JSON.stringify(token)}`);
    return id;
  }

  isSpecial(id: number): boolean {
    return this.special.has(id);
  }

  /** Text for a list of ids; special tokens are dropped unless asked for. */
  decode(ids: number[], { keepSpecial = false } = {}): string {
    const bytes: number[] = [];
    let text = "";
    const flush = () => {
      text += new TextDecoder().decode(new Uint8Array(bytes));
      bytes.length = 0;
    };
    for (const id of ids) {
      const tok = this.idToToken[id];
      if (tok === undefined) throw new Error(`tokenizer: unknown id ${id}`);
      if (this.special.has(id)) {
        if (keepSpecial) {
          flush();
          text += tok;
        }
        continue;
      }
      for (const ch of tok) {
        const b = this.bytesOf.get(ch);
        if (b === undefined) throw new Error(`tokenizer: ${JSON.stringify(ch)} isn't a byte-level character`);
        bytes.push(b);
      }
    }
    // decode bytes together: a multi-byte UTF-8 character can span tokens
    flush();
    return text;
  }
}
