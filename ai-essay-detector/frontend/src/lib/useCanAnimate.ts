/**
 * Whether an *entrance* animation may run right now.
 *
 * Entrance animations start from a hidden state — opacity 0, a translate, a
 * mask — and only the animation removes it. That makes them a liability the
 * moment they don't run, because what's left on screen is invisible content.
 * Two cases where they don't run:
 *
 *   prefers-reduced-motion    the user asked for the static end state.
 *   a background tab          requestAnimationFrame is suspended, so a
 *                             Framer Motion animation started there never
 *                             ticks. Analysis takes seconds and switching
 *                             tabs while waiting is the normal thing to do,
 *                             so this is a routine case, not an exotic one.
 *
 * Both return false, and every caller must then render the finished state
 * directly rather than the hidden one. This is deliberately a plain function
 * of render-time conditions, not state: it is read while deciding a component's
 * `initial` prop, which Framer Motion only honours on mount.
 *
 * Keep this as the single place that decision is made. It was previously
 * duplicated per component, and the copy that got missed shipped a hero whose
 * body copy stayed at opacity 0.
 */

import { useReducedMotion } from 'framer-motion'

export function useCanAnimate(): boolean {
  const reduceMotion = useReducedMotion()
  return (
    !reduceMotion &&
    typeof document !== 'undefined' &&
    document.visibilityState === 'visible'
  )
}
