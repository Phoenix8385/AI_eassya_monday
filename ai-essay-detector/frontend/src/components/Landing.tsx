import { motion } from 'framer-motion'
import { HeroVisual } from './hero/HeroVisual'
import { TextReveal } from './TextReveal'
import { useCanAnimate } from '../lib/useCanAnimate'

interface Props {
  onStart: () => void
}

/**
 * Hero. The paragraph is the point: the brief distinguishes a real signal
 * pipeline from an LLM asked for a verdict, so the mechanism is stated
 * outright rather than gestured at with "AI-powered".
 *
 * The 3D scan field sits behind this content in its own layer. It is decorative
 * and `aria-hidden`; every word here renders and every control here works
 * before WebGL has been asked for, let alone finished loading.
 */
export function Landing({ onStart }: Props) {
  const canAnimate = useCanAnimate()

  // Copy and CTA fade in on their own short timeline, independent of the 3D
  // layer — they are never waiting on it. When the entrance cannot run, the
  // variants are dropped entirely so nothing is left holding opacity 0.
  const fade = canAnimate
    ? ({
        hidden: { opacity: 0, y: 12 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
      } as const)
    : undefined

  return (
    <main className="landing">
      {/* Decorative layer. Never focusable, never announced. */}
      <div className="landing__visual">
        <HeroVisual />
      </div>

      <section className="landing__intro">
        <motion.p
          className="eyebrow"
          variants={fade}
          initial={fade ? 'hidden' : false}
          animate="visible"
        >
          Admissions essay analysis
        </motion.p>

        <TextReveal
          className="landing__headline"
          text="Measured signals, not a second opinion from a chatbot."
          delay={0.1}
        />

        <motion.p
          className="lede"
          variants={fade}
          initial={fade ? 'hidden' : false}
          animate="visible"
          transition={{ delay: 0.45 }}
        >
          This tool runs a local <strong>GPT-2</strong> over your essay and records
          how predictable each token is, how much sentence rhythm varies from one
          sentence to the next (<em>burstiness</em>), and a set of stylistic
          measurements — vocabulary variety, punctuation habits, how heavily the
          writing leans on connectives like <em>moreover</em> and{' '}
          <em>furthermore</em>. Those numbers feed a logistic-regression
          classifier trained here, on labelled human and AI essays.{' '}
          <strong>No language model is asked whether the essay is AI-written.</strong>{' '}
          Nothing is sent to a chatbot for judgement — the verdict-shaped
          question is never asked, because a model's opinion about authorship
          isn't evidence. What comes back is a probability, the signals behind
          it, and which sentences moved it.
        </motion.p>

        <motion.div
          variants={fade}
          initial={fade ? 'hidden' : false}
          animate="visible"
          transition={{ delay: 0.55 }}
        >
          <button type="button" className="cta cta--primary" onClick={onStart}>
            Analyze an essay
          </button>
        </motion.div>

        <motion.p
          className="landing__caveat"
          variants={fade}
          initial={fade ? 'hidden' : false}
          animate="visible"
          transition={{ delay: 0.65 }}
        >
          Results are a likelihood, not a verdict. Read{' '}
          <span className="nowrap">what this can and cannot show</span> before
          acting on one.
        </motion.p>
      </section>
    </main>
  )
}
