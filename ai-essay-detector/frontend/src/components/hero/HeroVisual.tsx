/**
 * Decides whether the hero gets WebGL at all, and guarantees it always gets
 * *something*.
 *
 * Three gates, in order, and each one falls back to the same static
 * composition rather than to empty space:
 *
 * 1. **prefers-reduced-motion** — the 3D module is never imported. Not paused,
 *    not slowed: `React.lazy` is never invoked, so no three.js, no WebGL
 *    context, no GPU work. Honouring the preference and skipping the download
 *    are the same decision here.
 * 2. **WebGL availability** — probed once against a throwaway canvas before
 *    mounting, because a context that cannot be created throws during render,
 *    which is a worse place to find out.
 * 3. **Runtime failure** — an error boundary catches anything that still goes
 *    wrong (driver crash, context loss on mount) and swaps in the static
 *    visual.
 *
 * The hero's copy and CTA never wait on any of this. They are rendered by the
 * parent and are interactive immediately; this component only ever occupies
 * the decorative layer behind them.
 */

import { Component, lazy, Suspense, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useReducedMotion } from 'framer-motion'
import { StaticScanField } from './StaticScanField'

// The only import of the 3D module. Everything three.js-shaped lives behind
// this boundary and ships as its own chunk.
const ScanFieldScene = lazy(() => import('./ScanFieldScene'))

class CanvasErrorBoundary extends Component<
  { fallback: ReactNode; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: unknown) {
    // Worth surfacing in the console: a hero that silently degrades on a whole
    // class of devices is a bug you never hear about otherwise.
    console.warn('Hero 3D scene failed; using the static visual instead.', error)
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

/** One-shot probe. Cheap, and cheaper than a thrown context error on mount. */
function detectWebGL(): boolean {
  if (typeof document === 'undefined') return false
  try {
    const canvas = document.createElement('canvas')
    const context =
      canvas.getContext('webgl2') ||
      canvas.getContext('webgl') ||
      canvas.getContext('experimental-webgl')
    if (!context) return false
    // Release it immediately; browsers cap simultaneous contexts and the real
    // one is about to be created.
    const lose = (context as WebGLRenderingContext).getExtension('WEBGL_lose_context')
    lose?.loseContext()
    return true
  } catch {
    return false
  }
}

export function HeroVisual() {
  const reduceMotion = useReducedMotion()
  // Starts null so the first paint is always the static visual: the 3D scene
  // is an upgrade applied after mount, never something the hero waits for.
  const [webglOk, setWebglOk] = useState<boolean | null>(null)
  // Smaller screens are likelier to be the mid-range hardware this has to stay
  // smooth on, and they show less of the field anyway.
  const [budget, setBudget] = useState(8000)

  useEffect(() => {
    if (reduceMotion) return
    setWebglOk(detectWebGL())
    const narrow = window.matchMedia('(max-width: 820px)').matches
    const coarse = window.matchMedia('(pointer: coarse)').matches
    setBudget(narrow || coarse ? 3500 : 8000)
  }, [reduceMotion])

  if (reduceMotion) {
    return <StaticScanField variant="reduced-motion" />
  }

  if (webglOk === false) {
    return <StaticScanField variant="fallback" />
  }

  if (webglOk === null) {
    return <StaticScanField variant="loading" />
  }

  return (
    <CanvasErrorBoundary fallback={<StaticScanField variant="fallback" />}>
      <Suspense fallback={<StaticScanField variant="loading" />}>
        <ScanFieldScene maxPoints={budget} />
      </Suspense>
    </CanvasErrorBoundary>
  )
}
