/**
 * The non-WebGL hero visual.
 *
 * Used in three situations, all of which must look deliberate rather than
 * degraded: prefers-reduced-motion is set, WebGL failed to initialise, or the
 * 3D chunk has not finished loading yet.
 *
 * It is the scan field's end state — the page of text with the band resting
 * mid-sweep — drawn in CSS gradients. Under reduced motion that is the point:
 * a still frame of the composition, not the same animation played slower.
 */

interface Props {
  /** Shown to nobody; kept for parity with the Canvas, which is also hidden. */
  variant?: 'reduced-motion' | 'fallback' | 'loading'
}

export function StaticScanField({ variant = 'fallback' }: Props) {
  return (
    <div
      className={`scanfield-static is-${variant}`}
      aria-hidden="true"
      data-variant={variant}
    >
      <div className="scanfield-static__page">
        {/* Text lines, in the same paragraph rhythm as the 3D layout. */}
        {STATIC_LINES.map((line, i) => (
          <span
            key={i}
            className={`scanfield-static__line${line.gap ? ' is-gap' : ''}`}
            style={{ width: `${line.width}%` }}
          />
        ))}
      </div>
      <span className="scanfield-static__band" />
    </div>
  )
}

// Mirrors buildScanField's shape: full-width lines, a short line at each
// paragraph end, and a blank line between paragraphs.
const STATIC_LINES: { width: number; gap?: boolean }[] = [
  { width: 96 },
  { width: 92 },
  { width: 97 },
  { width: 89 },
  { width: 94 },
  { width: 46 },
  { width: 0, gap: true },
  { width: 95 },
  { width: 91 },
  { width: 96 },
  { width: 88 },
  { width: 93 },
  { width: 38 },
  { width: 0, gap: true },
  { width: 94 },
  { width: 90 },
  { width: 97 },
  { width: 61 },
]
