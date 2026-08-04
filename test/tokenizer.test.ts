import { describe, expect, it } from "vitest";
import { Tokenizer, type TokenizerJson } from "../src/model/tokenizer";
import { hasAsset } from "./helpers";

const TOKENIZER = "/weights/whisper-tiny/tokenizer.json";
const hasTokenizer = await hasAsset(TOKENIZER);

// a tiny hand-made vocab in the byte-level alphabet ("Ġ" = space)
const tiny: TokenizerJson = {
  model: { vocab: { Hello: 0, "Ġworld": 1, "!": 2, "Ã©": 3, "Ã": 4, "©": 5 } },
  added_tokens: [{ id: 6, content: "<|endoftext|>", special: true }],
};

describe("tokenizer", () => {
  it("decodes byte-level tokens, spaces included", () => {
    const t = new Tokenizer(tiny);
    expect(t.decode([0, 1, 2])).toBe("Hello world!");
  });

  it("drops special tokens unless asked", () => {
    const t = new Tokenizer(tiny);
    expect(t.decode([0, 6])).toBe("Hello");
    expect(t.decode([0, 6], { keepSpecial: true })).toBe("Hello<|endoftext|>");
  });

  it("reassembles a UTF-8 character split across tokens", () => {
    const t = new Tokenizer(tiny);
    // é is 0xC3 0xA9; "Ã" and "©" carry one byte each
    expect(t.decode([3])).toBe("é");
    expect(t.decode([4, 5])).toBe("é");
  });

  it.runIf(hasTokenizer)("knows Whisper's special tokens", async () => {
    const t = new Tokenizer(await (await fetch(TOKENIZER)).json());
    expect(t.id("<|startoftranscript|>")).toBe(50258);
    expect(t.id("<|en|>")).toBe(50259);
    expect(t.id("<|transcribe|>")).toBe(50359);
    expect(t.id("<|notimestamps|>")).toBe(50363);
    expect(t.id("<|endoftext|>")).toBe(50257);
    expect(t.isSpecial(50258)).toBe(true);
  });
});
