import { describe, expect, it } from "vitest";
import { corpusWer, normalizeText, wordErrors } from "../src/eval/wer";

describe("normalizeText", () => {
  it("lowercases, strips punctuation and expands a few abbreviations", () => {
    expect(normalizeText("Mr. Quilter's apostle, of the middle-classes!")).toEqual([
      "mister",
      "quilters",
      "apostle",
      "of",
      "the",
      "middle",
      "classes",
    ]);
  });

  it("drops apostrophes so possessive spellings don't count as errors", () => {
    expect(normalizeText("'don't' he said ’twas")).toEqual(["dont", "he", "said", "twas"]);
    expect(normalizeText("Quilters'")).toEqual(normalizeText("QUILTER'S"));
  });
});

describe("wordErrors", () => {
  it("is 0 for a perfect transcript, ignoring case and punctuation", () => {
    expect(wordErrors("MISTER QUILTER IS THE APOSTLE", "Mr. Quilter is the apostle.").wer).toBe(0);
  });

  it("counts substitutions, deletions and insertions", () => {
    const r = wordErrors("the cat sat on the mat", "the bat sat the mat today");
    // cat->bat (S), on deleted (D), today inserted (I)
    expect(r).toMatchObject({ substitutions: 1, deletions: 1, insertions: 1, refWords: 6 });
    expect(r.wer).toBeCloseTo(3 / 6, 10);
  });

  it("handles empty sides", () => {
    expect(wordErrors("", "").wer).toBe(0);
    expect(wordErrors("a b", "").wer).toBe(1);
    expect(wordErrors("", "a").wer).toBe(1);
  });

  it("weights corpus WER by words, not utterances", () => {
    const r = corpusWer([
      { reference: "a", hypothesis: "b" }, // 1 error / 1 word
      { reference: "a b c d e f g h i", hypothesis: "a b c d e f g h i" }, // 0 / 9
    ]);
    expect(r.wer).toBeCloseTo(1 / 10, 10);
  });
});
