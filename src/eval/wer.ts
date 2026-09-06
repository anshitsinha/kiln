/**
 * Word error rate: (substitutions + deletions + insertions) / reference words,
 * from a word-level Levenshtein alignment.
 *
 * Both sides are normalised first, because "Mr. Quilter's" vs "MISTER
 * QUILTER'S" isn't a recognition error. This is a deliberately small
 * normaliser - lowercase, strip punctuation (keeping in-word apostrophes),
 * a few spelling conventions - not Whisper's full English text normaliser,
 * so absolute numbers aren't directly comparable with published WERs. Within
 * this repo (f32 vs int4, kiln vs another runtime on the same audio) it's
 * the same yardstick for everyone.
 */

const ABBREVIATIONS: Record<string, string> = {
  mr: "mister",
  mrs: "missus",
  dr: "doctor",
  st: "saint",
};

export function normalizeText(s: string): string[] {
  return s
    .toLowerCase()
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9' ]+/g, " ")
    .replace(/(^|\s)'+|'+(\s|$)/g, " ") // quotes, not apostrophes
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => ABBREVIATIONS[w] ?? w);
}

export interface WerResult {
  wer: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  refWords: number;
}

export function wordErrors(reference: string, hypothesis: string): WerResult {
  const ref = normalizeText(reference);
  const hyp = normalizeText(hypothesis);
  const n = ref.length;
  const m = hyp.length;
  // dp[i][j] = edit distance between ref[0..i) and hyp[0..j), with backpointers
  // tracked as op counts so the breakdown comes out too
  type Cell = { cost: number; s: number; d: number; i: number };
  let prev: Cell[] = Array.from({ length: m + 1 }, (_, j) => ({ cost: j, s: 0, d: 0, i: j }));
  for (let i = 1; i <= n; i++) {
    const cur: Cell[] = [{ cost: i, s: 0, d: i, i: 0 }];
    for (let j = 1; j <= m; j++) {
      const sub = prev[j - 1];
      const same = ref[i - 1] === hyp[j - 1];
      const options: Cell[] = [
        { ...sub, cost: sub.cost + (same ? 0 : 1), s: sub.s + (same ? 0 : 1) },
        { ...prev[j], cost: prev[j].cost + 1, d: prev[j].d + 1 },
        { ...cur[j - 1], cost: cur[j - 1].cost + 1, i: cur[j - 1].i + 1 },
      ];
      cur.push(options.reduce((a, b) => (b.cost < a.cost ? b : a)));
    }
    prev = cur;
  }
  const last = prev[m];
  return {
    wer: n === 0 ? (m === 0 ? 0 : 1) : last.cost / n,
    substitutions: last.s,
    deletions: last.d,
    insertions: last.i,
    refWords: n,
  };
}

/** Corpus WER: total errors over total reference words (not a mean of rates). */
export function corpusWer(pairs: { reference: string; hypothesis: string }[]): WerResult {
  const total = { wer: 0, substitutions: 0, deletions: 0, insertions: 0, refWords: 0 };
  for (const p of pairs) {
    const r = wordErrors(p.reference, p.hypothesis);
    total.substitutions += r.substitutions;
    total.deletions += r.deletions;
    total.insertions += r.insertions;
    total.refWords += r.refWords;
  }
  total.wer = total.refWords ? (total.substitutions + total.deletions + total.insertions) / total.refWords : 0;
  return total;
}
