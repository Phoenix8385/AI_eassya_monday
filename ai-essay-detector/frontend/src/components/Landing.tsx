import { motion, useReducedMotion } from 'framer-motion'

interface Props {
  onStart: () => void
}

const SCAN_LINES = [92, 76, 88, 61, 84, 70]

/**
 * Hero. The paragraph is the point: the brief distinguishes a real signal
 * pipeline from an LLM asked for a verdict, so the mechanism is stated
 * outright rather than gestured at with "AI-powered".
 */
export function Landing({ onStart }: Props) {
  const reduceMotion = useReducedMotion()

  return (
    <main className="landing">
      <section className="landing__intro">
        <p className="eyebrow">Admissions essay analysis</p>
        <h1>
          Measured signals,
          <br />
          not a second opinion from a chatbot.
        </h1>

        <p className="lede">
          This tool runs a local <strong>GPT-2</strong> over your essay and records
          how predictable each token is, how much sentence rhythm varies from one
          sentence to the next (<em>burstiness</em>), and a set of stylistic
          measurements — vocabulary variety, punctuation habits, how heavily the
          writing leans on connectives like <em>moreover</em> and{' '}
          <em>furthermore</em>. Those numbers feed a logistic-regression
          classifier trained here, on labelled human and AI essays.{' '}
          <strong>
            No language model is asked whether the essay is AI-written.
          </strong>{' '}
          Nothing is sent to a chatbot for judgement — the verdict-shaped
          question is never asked, because a model's opinion about authorship
          isn't evidence. What comes back is a probability, the signals behind
          it, and which sentences moved it.
        </p>

        <button type="button" className="cta" onClick={onStart}>
          Analyze an essay
        </button>

        <p className="landing__caveat">
          Results are a likelihood, not a verdict. Read{' '}
          <span className="nowrap">what this can and cannot show</span> before
          acting on one.
        </p>
      </section>

      {/*
        Decorative. Under prefers-reduced-motion this renders as the finished
        state — every line at full opacity with the scan bar parked — rather
        than the same animation played slowly.
      */}
      <div className="scanner" aria-hidden="true">
        <div className="scanner__frame">
          {SCAN_LINES.map((width, i) => (
            <motion.span
              key={i}
              className="scanner__line"
              style={{ width: `${width}%` }}
              initial={reduceMotion ? false : { opacity: 0.25 }}
              animate={reduceMotion ? { opacity: 1 } : { opacity: [0.25, 1, 0.25] }}
              transition={
                reduceMotion
                  ? { duration: 0 }
                  : {
                      duration: 2.4,
                      repeat: Infinity,
                      delay: i * 0.28,
                      ease: 'easeInOut',
                    }
              }
            />
          ))}
          {reduceMotion ? (
            <span className="scanner__bar is-parked" />
          ) : (
            <motion.span
              className="scanner__bar"
              initial={{ y: '0%' }}
              animate={{ y: ['0%', '520%', '0%'] }}
              transition={{ duration: 4.2, repeat: Infinity, ease: 'easeInOut' }}
            />
          )}
        </div>
      </div>
    </main>
  )
}
