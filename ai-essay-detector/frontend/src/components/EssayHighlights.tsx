import { useCallback, useEffect, useRef } from 'react'
import { motion, useReducedMotion } from 'framer-motion'
import type { Sentence } from '../types'

interface Props {
  sentences: Sentence[]
  openIndex: number | null
  onOpen: (index: number) => void
  onClose: () => void
  panelId: string
}

/**
 * Local scores are z-scores; in practice they sit between 0 and ~3. Anything
 * past this is clamped to full intensity rather than being allowed to make
 * every other sentence look blank by comparison.
 */
const INTENSITY_CEILING = 2.5

function intensityOf(localScore: number | null): number {
  if (localScore === null || localScore <= 0) return 0
  return Math.min(localScore / INTENSITY_CEILING, 1)
}

/**
 * Renders the essay with each sentence shaded by how far it deviates from the
 * essay's own norm.
 *
 * **Single hue, deliberately.** A red/green diverging scale is unreadable for
 * a significant share of colourblind readers, and it frames the result as
 * good/bad. One amber ramp reads as "how much signal is here", which is what
 * the number actually means.
 *
 * **Colour is never the only channel.** Flagged sentences also carry an
 * underline and a marker glyph, so the distinction survives greyscale, a
 * monochrome display, or colour-vision deficiency.
 */
export function EssayHighlights({
  sentences,
  openIndex,
  onOpen,
  onClose,
  panelId,
}: Props) {
  const reduceMotion = useReducedMotion()
  const refs = useRef(new Map<number, HTMLElement>())

  // Escape closes from anywhere and returns focus to the sentence that opened
  // the panel — otherwise a keyboard user is dumped at the top of the document.
  useEffect(() => {
    if (openIndex === null) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.stopPropagation()
      const el = refs.current.get(openIndex)
      onClose()
      el?.focus()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [openIndex, onClose])

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent, index: number, isOpen: boolean) => {
      if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
        // Space would otherwise scroll the page.
        event.preventDefault()
        if (isOpen) {
          onClose()
        } else {
          onOpen(index)
        }
      }
    },
    [onOpen, onClose],
  )

  // The stagger is an entrance effect only. Under reduced motion the list is
  // rendered without motion wrappers at all — a static end state, not a slow
  // version of the same animation.
  //
  // `hidden` writes opacity:0 inline on the first render, so it is only ever
  // introduced when something is guaranteed to take it away again. Without that
  // the markup carries no hidden state at all and every sentence — text,
  // reason, intensity, ARIA — is readable exactly as rendered. The animation is
  // layered on correct content; it is never what makes the content appear.
  //
  // The visibility check is not paranoia. requestAnimationFrame is suspended in
  // a background tab, so a stagger that starts there never ticks: someone who
  // switches away while the request is in flight would come back to an essay
  // whose every sentence is still at opacity 0. Analysis takes seconds, which
  // makes switching away the normal thing to do, not the edge case.
  const canAnimate =
    !reduceMotion &&
    typeof document !== 'undefined' &&
    document.visibilityState === 'visible'

  // Bounded, not per-item-fixed. The backend returns up to
  // MAX_EXPLAINED_SENTENCES (200) sentences; a flat 0.035s step would leave the
  // last one invisible for seven seconds. Spreading a fixed budget instead
  // keeps the whole entrance under ~1.2s however long the essay is.
  const stagger = Math.min(0.035, 1.2 / Math.max(sentences.length, 1))

  const container = canAnimate
    ? { visible: { transition: { staggerChildren: stagger } } }
    : undefined
  const item = canAnimate
    ? {
        hidden: { opacity: 0, y: 6 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.32 } },
      }
    : undefined

  return (
    <motion.p
      className={`essay ${canAnimate ? '' : 'is-static'}`}
      variants={container}
      initial={canAnimate ? 'hidden' : false}
      animate="visible"
    >
      {sentences.map((sentence) => {
        const unscored = sentence.local_score === null
        const intensity = intensityOf(sentence.local_score)
        const isOpen = openIndex === sentence.index
        // Only flagged sentences are interactive. An unscored sentence's
        // reason is identical for every sentence in the essay, so it is shown
        // once as a notice rather than behind N identical buttons.
        const interactive = sentence.flagged && sentence.reason !== null

        const classes = [
          'sentence',
          unscored ? 'is-unscored' : '',
          sentence.flagged ? 'is-flagged' : '',
          isOpen ? 'is-open' : '',
        ]
          .filter(Boolean)
          .join(' ')

        const style = unscored
          ? undefined
          : ({ '--intensity': intensity.toFixed(3) } as React.CSSProperties)

        const content = (
          <>
            {sentence.flagged && (
              <span className="sentence__marker" aria-hidden="true">
                ▲
              </span>
            )}
            {sentence.text}{' '}
          </>
        )

        if (!interactive) {
          return (
            <motion.span
              key={sentence.index}
              className={classes}
              style={style}
              variants={item}
            >
              {content}
            </motion.span>
          )
        }

        return (
          <motion.span
            key={sentence.index}
            ref={(el: HTMLElement | null) => {
              if (el) refs.current.set(sentence.index, el)
              else refs.current.delete(sentence.index)
            }}
            className={classes}
            style={style}
            variants={item}
            role="button"
            tabIndex={0}
            aria-expanded={isOpen}
            aria-controls={panelId}
            // The reason is on the element itself, so assistive tech reaches
            // the evidence without having to open the panel at all.
            aria-label={`Flagged sentence: ${sentence.text} — ${sentence.reason}`}
            onClick={() => (isOpen ? onClose() : onOpen(sentence.index))}
            onKeyDown={(event) => handleKeyDown(event, sentence.index, isOpen)}
          >
            {content}
          </motion.span>
        )
      })}
    </motion.p>
  )
}
