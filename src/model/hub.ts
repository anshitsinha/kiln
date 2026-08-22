/**
 * Fetching model files from the Hugging Face Hub in the browser, with
 * progress and a persistent cache.
 *
 * The weights are 151 MB for whisper-tiny; nobody should download them
 * twice. Files go into Cache Storage (survives reloads, per origin) keyed by
 * URL. Where Cache Storage isn't available (some private-browsing modes) it
 * just downloads every time.
 */

const CACHE = "kiln-models-v1";

export interface Progress {
  loaded: number;
  total: number; // 0 if the server didn't say
  cached: boolean;
}

export function hubUrl(repo: string, file: string, revision = "main"): string {
  return `https://huggingface.co/${repo}/resolve/${revision}/${file}`;
}

async function openCache(): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(CACHE);
  } catch {
    return null;
  }
}

/** Fetch a URL as an ArrayBuffer, from cache when possible, reporting progress. */
export async function fetchCached(url: string, onProgress: (p: Progress) => void = () => {}): Promise<ArrayBuffer> {
  const cache = await openCache();
  const hit = await cache?.match(url);
  if (hit) {
    const buf = await hit.arrayBuffer();
    onProgress({ loaded: buf.byteLength, total: buf.byteLength, cached: true });
    return buf;
  }

  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);

  // read the stream ourselves to report progress
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress({ loaded, total, cached: false });
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.byteLength;
  }

  try {
    await cache?.put(url, new Response(out.slice(), { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    // quota exceeded etc. - not fatal, it just won't be cached
  }
  return out.buffer;
}

/** Drop everything kiln has cached. */
export async function clearModelCache(): Promise<void> {
  if (typeof caches !== "undefined") await caches.delete(CACHE);
}
