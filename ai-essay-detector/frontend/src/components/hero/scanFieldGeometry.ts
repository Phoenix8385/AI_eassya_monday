/**
 * Point layout for the hero scene: a page of prose, rendered as particles.
 *
 * The shape is the whole idea. A generic particle cloud says nothing; this
 * lays points out as justified lines with word gaps, ragged paragraph endings
 * and blank lines between paragraphs, so from a distance it reads as a page of
 * text and up close it dissolves into measurement points. That is the tool's
 * actual subject — prose looked at as data — rather than decoration bolted on
 * top of it.
 *
 * Deterministic on purpose: a seeded RNG means the "page" is identical on every
 * load, so the hero is a designed composition rather than a different accident
 * each refresh.
 */

export interface ScanFieldGeometry {
  positions: Float32Array
  /** Per-point random 0..1, used to desynchronise drift in the shader. */
  seeds: Float32Array
  count: number
}

/** Mulberry32 — small, fast, and stable across engines. */
function makeRandom(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export interface ScanFieldOptions {
  /** Upper bound on points. The layout stops early rather than exceeding it. */
  maxPoints: number
  seed?: number
}

const HALF_WIDTH = 3.3
const LINE_GAP = 0.2
const ROWS = 30
// Two sub-rows per line of text give each "line" some vertical body, so it
// reads as a strip of glyphs rather than a bare 1px rule.
const SUBROWS = [0.0, 0.055]
const POINT_SPACING = 0.038

export function buildScanField({
  maxPoints,
  seed = 20260816,
}: ScanFieldOptions): ScanFieldGeometry {
  const random = makeRandom(seed)
  const xs: number[] = []
  const ys: number[] = []
  const zs: number[] = []
  const seeds: number[] = []

  const topY = ((ROWS - 1) * LINE_GAP) / 2

  for (let row = 0; row < ROWS; row += 1) {
    // A blank line every few rows: paragraph breaks are most of what makes a
    // block of marks look like prose instead of a texture.
    const isParagraphBreak = row % 7 === 6
    if (isParagraphBreak) continue

    const y = topY - row * LINE_GAP
    // The line before a break is the end of a paragraph, so it runs short.
    const isParagraphEnd = row % 7 === 5
    const lineWidth = isParagraphEnd ? HALF_WIDTH * (0.3 + random() * 0.5) : HALF_WIDTH

    let x = -HALF_WIDTH
    while (x < lineWidth) {
      const wordLength = 0.12 + random() * 0.42
      const wordEnd = Math.min(x + wordLength, lineWidth)

      for (let px = x; px < wordEnd; px += POINT_SPACING) {
        for (const subRow of SUBROWS) {
          if (xs.length >= maxPoints) {
            return toGeometry(xs, ys, zs, seeds)
          }
          xs.push(px + (random() - 0.5) * 0.012)
          ys.push(y + subRow + (random() - 0.5) * 0.014)
          // Slight depth spread so the page has thickness and the scan band
          // catches points at visibly different distances.
          zs.push((random() - 0.5) * 0.16)
          seeds.push(random())
        }
      }

      x = wordEnd + 0.055 + random() * 0.05 // inter-word gap
    }
  }

  return toGeometry(xs, ys, zs, seeds)
}

function toGeometry(
  xs: number[],
  ys: number[],
  zs: number[],
  seeds: number[],
): ScanFieldGeometry {
  const count = xs.length
  const positions = new Float32Array(count * 3)
  for (let i = 0; i < count; i += 1) {
    positions[i * 3] = xs[i]
    positions[i * 3 + 1] = ys[i]
    positions[i * 3 + 2] = zs[i]
  }
  return { positions, seeds: new Float32Array(seeds), count }
}

/** Vertical extent the scan bar has to travel to cross the whole page. */
export const FIELD_TOP = ((ROWS - 1) * LINE_GAP) / 2 + 0.3
export const FIELD_BOTTOM = -FIELD_TOP
