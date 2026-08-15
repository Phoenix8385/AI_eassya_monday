import { useEffect, useState } from 'react'
import { animate, useMotionValue } from 'framer-motion'
import { useCanAnimate } from '../lib/useCanAnimate'
import type { Label } from '../types'

interface Props {
  probability: number
  confidenceLabel: string
  label: Label
}

const SIZE = 200
const STROKE = 14
const RADIUS = (SIZE - STROKE) / 2
/** 270° arc, leaving a gap at the bottom so the ends read as a scale. */
const SWEEP = 0.75
const CIRCUMFERENCE = 2 * Math.PI * RADIUS
const ARC = CIRCUMFERENCE * SWEEP

/**
 * Radial arc counting up to the essay's probability.
 *
 * The number is never the whole message: `confidence_label`'s full sentence
 * renders directly beneath it, and the accessible name for the whole component
 * is that sentence — so a screen reader hears the framing, not a bare figure.
 */
export function ProbabilityGauge({ probability, confidenceLabel, label }: Props) {
  // Same rule as every other entrance here: if the count-up cannot run, the
  // gauge shows the real number immediately rather than a 0% that never moves.
  const canCountUp = useCanAnimate()
  const reduceMotion = !canCountUp
  const progress = useMotionValue(reduceMotion ? probability : 0)
  // Mirrored into state so the digits can render as plain text: real text is
  // selectable and survives if the animation never runs.
  const [shown, setShown] = useState(reduceMotion ? probability : 0)

  useEffect(() => {
    // The readout has to be correct whether or not a single frame ever
    // renders. requestAnimationFrame is suspended in a background tab, so a
    // count-up started there never ticks — and someone who switched away
    // during the request comes back to a gauge reading 0% sitting next to a
    // sentence saying 99%. Same rule as the sentence stagger: the animation
    // decorates a value that is already right, it never produces one.
    if (!canCountUp) {
      progress.set(probability)
      setShown(probability)
      return
    }

    progress.set(0)
    setShown(0)
    const unsubscribe = progress.on('change', setShown)
    const controls = animate(progress, probability, {
      duration: 1.1,
      ease: [0.22, 1, 0.36, 1],
      // Commits the exact target even if the last frame lands a hair short,
      // or if the tab was hidden mid-flight and the animation resumed past
      // its own duration.
      onComplete: () => setShown(probability),
    })
    return () => {
      controls.stop()
      unsubscribe()
    }
  }, [probability, reduceMotion, canCountUp, progress])

  const percent = Math.round(shown * 100)
  const offset = ARC * (1 - shown)
  const leaning = label === 'Likely AI'

  return (
    <figure className="gauge" role="img" aria-label={confidenceLabel}>
      <div className="gauge__dial">
        <svg
          width={SIZE}
          height={SIZE}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          aria-hidden="true"
          focusable="false"
        >
          {/* Rotated so the 270° arc opens at the bottom. */}
          <g transform={`rotate(135 ${SIZE / 2} ${SIZE / 2})`}>
            <circle
              className="gauge__track"
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={RADIUS}
              strokeWidth={STROKE}
              strokeDasharray={`${ARC} ${CIRCUMFERENCE}`}
              strokeLinecap="round"
            />
            <circle
              className={`gauge__value ${leaning ? 'is-ai' : 'is-human'}`}
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={RADIUS}
              strokeWidth={STROKE}
              strokeDasharray={`${ARC} ${CIRCUMFERENCE}`}
              strokeDashoffset={offset}
              strokeLinecap="round"
            />
          </g>
        </svg>

        <div className="gauge__readout" aria-hidden="true">
          <span className="gauge__percent">{percent}%</span>
          <span className="gauge__caption">resemblance to
            <br />AI-generated examples</span>
        </div>
      </div>

      {/* The tag is a leaning, not a verdict — both values carry "Likely". */}
      <p className={`gauge__tag ${leaning ? 'is-ai' : 'is-human'}`}>{label}</p>

      {/* The sentence, not the number, is the result. */}
      <figcaption className="gauge__sentence">{confidenceLabel}</figcaption>
    </figure>
  )
}
