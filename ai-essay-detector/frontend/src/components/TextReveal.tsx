/**
 * Masked word-by-word headline reveal.
 *
 * Built on the "text reveal (mask)" pattern that 21st.dev catalogues — each
 * word sits in an `overflow: hidden` box and slides up into it on a stagger —
 * rather than copied from it: the listings there require an account to read the
 * source, so this is the same construction rewritten against Framer Motion and
 * this app's tokens. Nothing to strip out afterwards, and no second animation
 * library fighting the rest of the UI.
 *
 * **The text is always in the DOM as text.** Words are real text nodes inside
 * a single heading, split only for masking, so the accessible name is the whole
 * sentence and selection/copy still work. The mask is applied only when there
 * is something running to remove it — same rule the analyzer's sentence stagger
 * follows, for the same reason: an entrance effect must never be what makes
 * content visible.
 */

import { motion, useReducedMotion } from 'framer-motion'
import type { ReactNode } from 'react'

interface Props {
  /** Plain text. Split on whitespace for the per-word mask. */
  text: string
  /** Rendered after the animated words, unmasked (e.g. an emphasised clause). */
  children?: ReactNode
  className?: string
  /** Seconds between each word. */
  stagger?: number
  delay?: number
  as?: 'h1' | 'h2' | 'p'
}

export function TextReveal({
  text,
  children,
  className = '',
  stagger = 0.045,
  delay = 0,
  as: Tag = 'h1',
}: Props) {
  const reduceMotion = useReducedMotion()

  // requestAnimationFrame is suspended in a background tab, so a reveal that
  // starts there never runs and would leave the headline permanently blank.
  const canAnimate =
    !reduceMotion &&
    typeof document !== 'undefined' &&
    document.visibilityState === 'visible'

  const words = text.split(/\s+/).filter(Boolean)

  if (!canAnimate) {
    return (
      <Tag className={`reveal is-static ${className}`}>
        {text}
        {children}
      </Tag>
    )
  }

  return (
    <Tag className={`reveal ${className}`}>
      <motion.span
        className="reveal__inner"
        initial="hidden"
        animate="visible"
        variants={{ visible: { transition: { staggerChildren: stagger, delayChildren: delay } } }}
      >
        {words.map((word, i) => (
          <span className="reveal__mask" key={`${word}-${i}`}>
            <motion.span
              className="reveal__word"
              variants={{
                hidden: { y: '110%' },
                visible: {
                  y: '0%',
                  transition: { duration: 0.62, ease: [0.16, 1, 0.3, 1] },
                },
              }}
            >
              {word}
            </motion.span>{' '}
          </span>
        ))}
      </motion.span>
      {children}
    </Tag>
  )
}
