/**
 * The 24-second looping intro/showcase from the design handoff, rebuilt on
 * this app's stack.
 *
 * It is one element tree rendered as a function of a single looping clock —
 * nothing mounts or unmounts at a scene boundary. The prototype ran on an
 * authored-time engine (`animations-v3.jsx`) that does not exist here; the
 * timing, easing curves and per-element offsets below are transcribed from it,
 * but the mechanism is Framer Motion:
 *
 *   - one `MotionValue<number>` holds elapsed seconds, advanced by
 *     `useAnimationFrame` and wrapped at LOOP_SECONDS;
 *   - every animated property is a `useTransform` off that value, so frames
 *     are applied straight to the DOM and React never re-renders during the
 *     loop. At 60fps with 46 particles, re-rendering the tree would be the
 *     whole cost of the piece.
 *
 * **The result data is not animated into existence.** Following the same rule
 * as ProbabilityGauge and the sentence stagger: when entrance animation cannot
 * run — `prefers-reduced-motion`, a background tab, or server rendering — the
 * clock is never started and every scene renders at its own settled time, laid
 * out as a static poster (see `.is-static` in ShowcaseIntro.css). The real
 * probability, label and sentence are in the markup either way.
 *
 * Styles live in ShowcaseIntro.css, pulled in by App.css rather than imported
 * here: the Node-side render check bundles components with plain rolldown,
 * which cannot load a stylesheet, and a component that can't be rendered by
 * that check is a component whose static output nobody is verifying.
 *
 * Deliberately passive: no click handlers, matching the reference. The closing
 * "Analyze an essay" pill is a picture of a button — the real control belongs
 * to the page that hosts this piece. That is also why the whole thing is one
 * `role="img"` with the result as its accessible name: which words are on
 * screen at any instant depends on where the loop happens to be, so exposing
 * the cycling text to a screen reader would announce a different thing every
 * time it was read.
 */

import { useRef } from 'react'
import { motion, useAnimationFrame, useMotionValue, useTransform } from 'framer-motion'
import type { MotionValue } from 'framer-motion'
import { useCanAnimate } from '../lib/useCanAnimate'
import type { AnalyzeResponse } from '../types'

/**
 * Exactly the fields the piece displays, so a whole `AnalyzeResponse` can be
 * passed straight in but the component's real dependency stays visible.
 */
export type ShowcaseResult = Pick<
  AnalyzeResponse,
  'overall_probability' | 'confidence_label' | 'label'
>

/** One bar in the evidence scene. `value` is a 0–1 proportion of the track. */
export interface EvidenceBar {
  name: string
  value: number
}

export interface ShowcaseIntroProps {
  /**
   * The result to show in the gauge scene — pass the `/analyze` response.
   * The dial fills to `overall_probability`, the pill is `label`, and the
   * sentence under it is `confidence_label` verbatim.
   */
  result: ShowcaseResult
  /**
   * Heights of the three evidence bars, 0–1.
   *
   * These are illustrative proportions in the design, not per-essay data, and
   * they default to the reference's values. The document-level `signals` the
   * backend returns are not on `AnalyzeResponse` here and are unbounded raw
   * measurements (perplexity, burstiness, …), so there is no honest mapping to
   * a 0–1 bar height to apply automatically — pass real ones when a caller has
   * decided how to normalise them.
   */
  evidence?: EvidenceBar[]
  /** The drifting 3D particle field. Off leaves the photo and scrim. */
  particles?: boolean
  /** Particle glow radius multiplier, 0–2. */
  glowIntensity?: number
  className?: string
}

/* ---------------------------------------------------------------------------
   Timeline.

   The scene list is the outline, and the cue table is derived from it, so the
   two cannot drift. Every time below is authored seconds from the top of the
   loop, exactly as in the handoff's timeline table.
--------------------------------------------------------------------------- */

const SCENES = [
  { name: 'opening', seconds: 3 },
  { name: 'signals', seconds: 6 },
  { name: 'gauge', seconds: 6 },
  { name: 'evidence', seconds: 5 },
  { name: 'close', seconds: 4 },
] as const

type SceneName = (typeof SCENES)[number]['name']

const CUE: Record<SceneName, number> = (() => {
  const table = {} as Record<SceneName, number>
  let at = 0
  for (const scene of SCENES) {
    table[scene.name] = at
    at += scene.seconds
  }
  return table
})()

const LOOP_SECONDS = SCENES.reduce((total, scene) => total + scene.seconds, 0)

/* ---------------------------------------------------------------------------
   Motion.

   Three curves, and nothing outside them eases anything — the same discipline
   the reference followed, and the reason the piece reads as one continuous
   video rather than a stack of independent effects.
--------------------------------------------------------------------------- */

const easeOutCubic = (t: number) => {
  const u = t - 1
  return u * u * u + 1
}

const easeInOutCubic = (t: number) =>
  t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1

const easeOutBack = (t: number) => {
  const c1 = 1.70158
  const c3 = c1 + 1
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2)
}

/** Single-segment tween. Holds `from` before `start` and `to` after `end`. */
function tween(
  from: number,
  to: number,
  start: number,
  end: number,
  ease: (t: number) => number,
) {
  return (T: number) => {
    if (T <= start) return from
    if (T >= end) return to
    return from + (to - from) * ease((T - start) / (end - start))
  }
}

const MOTION = {
  /** Fade/rise in. */
  enter: (T: number, start: number, end: number) => tween(0, 1, start, end, easeOutCubic)(T),
  /** Fill or grow along a track. */
  draw: (T: number, start: number, end: number) => tween(0, 1, start, end, easeInOutCubic)(T),
  /** Scale pop with a slight overshoot. */
  pop: (T: number, start: number, end: number) => tween(0.85, 1, start, end, easeOutBack)(T),
}

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value))

/* ---------------------------------------------------------------------------
   Frame plumbing.
--------------------------------------------------------------------------- */

/** A value that is either live off the clock, or a settled constant. */
type Animated<T> = T | MotionValue<T>

/**
 * One animated property.
 *
 * `settleAt` is the whole static story: null means "follow the clock", and a
 * number means "this scene is frozen at that second", which is what every
 * scene gets when animation cannot run. The transform is still created either
 * way — hooks are unconditional — it simply never receives a tick.
 */
function useFrameValue<O>(
  clock: MotionValue<number>,
  settleAt: number | null,
  compute: (seconds: number) => O,
): Animated<O> {
  const live = useTransform(clock, compute)
  return settleAt === null ? live : compute(settleAt)
}

/**
 * An authored hard cut: children are hidden outside [from, to). A settled
 * scene is always inside its own window, so this resolves to `visible`.
 */
function useShot(
  clock: MotionValue<number>,
  settleAt: number | null,
  from: number,
  to: number,
): Animated<'visible' | 'hidden'> {
  return useFrameValue(clock, settleAt, (T) => (T >= from && T < to ? 'visible' : 'hidden'))
}

/**
 * The clock, as a component so that `useAnimationFrame` is only ever
 * registered when the piece is actually playing — a reduced-motion reader
 * should not be paying for a 60fps callback that does nothing.
 *
 * Elapsed time is measured from the first frame's timestamp rather than
 * accumulated from each frame's delta. Accumulating looks equivalent and is
 * not: the frame loop clamps delta so that a stall cannot produce one huge
 * jump, so under any throttling — an occluded window, a battery saver, a
 * long task — the piece silently plays in slow motion, minutes of wall clock
 * covering seconds of the timeline. Reading the timestamp keeps the loop on
 * real time no matter how few frames it is given, and starting from the first
 * one means the piece opens at 0 rather than wherever the page's clock
 * happened to be.
 */
function LoopClock({ clock }: { clock: MotionValue<number> }) {
  const startedAt = useRef<number | null>(null)
  useAnimationFrame((timestamp) => {
    if (startedAt.current === null) startedAt.current = timestamp
    clock.set(((timestamp - startedAt.current) / 1000) % LOOP_SECONDS)
  })
  return null
}

/* ---------------------------------------------------------------------------
   Particle field.

   46 particles on a shared perspective, positions derived from a sine hash so
   the field is identical on every render and every machine. Size rides in the
   transform rather than width/height: one composited property per particle
   instead of a layout pass, and the glow scales with it for free.
--------------------------------------------------------------------------- */

const PARTICLE_COUNT = 46

const PARTICLES = Array.from({ length: PARTICLE_COUNT }, (_, i) => ({
  baseX: ((Math.sin(i * 12.9898) * 43758.5453) % 1) * 100,
  baseY: ((Math.sin(i * 78.233) * 12543.5453) % 1) * 100,
  z: ((Math.sin(i * 37.719) * 5647.234) % 1) * 600 - 300,
  radius: 1.4 + (Math.sin(i * 4.31) * 0.5 + 0.5) * 2.6,
  speed: 0.15 + (Math.sin(i * 2.11) * 0.5 + 0.5) * 0.35,
  phase: i * 0.618,
}))

type ParticleSeed = (typeof PARTICLES)[number]

function Particle({
  seed,
  clock,
  settleAt,
  glow,
}: {
  seed: ParticleSeed
  clock: MotionValue<number>
  settleAt: number | null
  glow: number
}) {
  const angleAt = (T: number) => (T / LOOP_SECONDS) * Math.PI * 2
  const depthAt = (T: number) =>
    (seed.z + Math.sin(angleAt(T) + seed.phase) * 60 + 300) / 600

  const left = useFrameValue(clock, settleAt, (T) => {
    const drift = Math.sin(angleAt(T) * seed.speed + seed.phase) * 18
    return `${(seed.baseX + drift + 200) % 100}%`
  })
  const top = useFrameValue(clock, settleAt, (T) => {
    const drift = Math.cos(angleAt(T) * seed.speed + seed.phase) * 12
    return `${(seed.baseY + drift + 200) % 100}%`
  })
  const z = useFrameValue(clock, settleAt, (T) => seed.z + Math.sin(angleAt(T) + seed.phase) * 60)
  const scale = useFrameValue(clock, settleAt, (T) => 0.6 + depthAt(T))
  const opacity = useFrameValue(clock, settleAt, (T) => 0.15 + depthAt(T) * 0.55)

  // Blur is fixed at the particle's own depth and then rides the scale above,
  // which reproduces the reference's near-particles-glow-harder relationship
  // without a sixth value to write every frame.
  const restDepth = clamp((seed.z + 300) / 600, 0, 1)

  return (
    <motion.i
      className="showcase__particle"
      style={{
        left,
        top,
        z,
        scale,
        opacity,
        width: seed.radius,
        height: seed.radius,
        boxShadow: `0 0 ${(6 + restDepth * 10) * glow}px var(--amber-hot)`,
      }}
    />
  )
}

function ParticleField({
  clock,
  settleAt,
  glow,
}: {
  clock: MotionValue<number>
  settleAt: number | null
  glow: number
}) {
  return (
    <div className="showcase__particles">
      <div className="showcase__depth">
        {PARTICLES.map((seed, i) => (
          <Particle key={i} seed={seed} clock={clock} settleAt={settleAt} glow={glow} />
        ))}
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------------------
   Scenes.
--------------------------------------------------------------------------- */

function OpeningScene({
  clock,
  settleAt,
}: {
  clock: MotionValue<number>
  settleAt: number | null
}) {
  const start = CUE.opening
  const visibility = useShot(clock, settleAt, start, CUE.signals)
  const opacity = useFrameValue(clock, settleAt, (T) =>
    MOTION.enter(T, start + 0.1, start + 1),
  )
  const eyebrowOpacity = useFrameValue(clock, settleAt, (T) =>
    MOTION.enter(T, start + 0.5, start + 1.4),
  )

  return (
    <motion.div
      className="showcase__scene showcase__scene--opening"
      style={{ visibility, opacity }}
    >
      <motion.p className="showcase__eyebrow" style={{ opacity: eyebrowOpacity }}>
        Admissions essay analysis
      </motion.p>
      <p className="showcase__title">AI Essay Detector</p>
    </motion.div>
  )
}

const SIGNAL_CARDS = [
  {
    name: 'Token perplexity',
    sub: 'How predictable each token is, scored by a local GPT-2 pass.',
  },
  {
    name: 'Sentence-rhythm burstiness',
    sub: 'How much rhythm varies from one sentence to the next.',
  },
  {
    name: 'Stylistic features',
    sub: 'Vocabulary variety, punctuation habits, connective use.',
  },
]

function SignalCard({
  card,
  index,
  clock,
  settleAt,
}: {
  card: (typeof SIGNAL_CARDS)[number]
  index: number
  clock: MotionValue<number>
  settleAt: number | null
}) {
  const start = CUE.signals
  const end = CUE.gauge
  const enterFrom = start + index * 0.25

  const opacity = useFrameValue(clock, settleAt, (T) => {
    const enter = MOTION.enter(T, enterFrom, enterFrom + 1.1)
    const exit = 1 - MOTION.enter(T, end + 0.1 + index * 0.1, end + 0.7 + index * 0.1)
    return enter * (T < end ? 1 : exit)
  })
  const y = useFrameValue(
    clock,
    settleAt,
    (T) => (1 - MOTION.enter(T, enterFrom, enterFrom + 1.1)) * 40,
  )
  const scale = useFrameValue(clock, settleAt, (T) => MOTION.pop(T, enterFrom, enterFrom + 1))
  // The slow orbital wobble is motion, not layout: a poster frame keeps the
  // cards square rather than freezing three arbitrary rotations.
  const rotateY = useFrameValue(clock, settleAt, (T) =>
    settleAt === null ? Math.sin((T - start) * 0.6 + index * 2.1) * 6 : 0,
  )

  return (
    <motion.article className="showcase__card" style={{ opacity, y, scale, rotateY }}>
      <span className="showcase__card-rule" />
      <h3 className="showcase__card-name">{card.name}</h3>
      <p className="showcase__card-sub">{card.sub}</p>
    </motion.article>
  )
}

function SignalScene({
  clock,
  settleAt,
}: {
  clock: MotionValue<number>
  settleAt: number | null
}) {
  const visibility = useShot(clock, settleAt, CUE.signals - 0.5, CUE.gauge + 0.8)

  return (
    <motion.div
      className="showcase__scene showcase__scene--signals"
      style={{ visibility }}
    >
      {SIGNAL_CARDS.map((card, i) => (
        <SignalCard key={card.name} card={card} index={i} clock={clock} settleAt={settleAt} />
      ))}
    </motion.div>
  )
}

/** --amber-hot, as channels, for the one shadow whose alpha is animated. */
const AMBER_HOT_RGB = '240, 169, 60'

function GaugeScene({
  clock,
  settleAt,
  result,
}: {
  clock: MotionValue<number>
  settleAt: number | null
  result: ShowcaseResult
}) {
  const start = CUE.gauge
  const end = CUE.evidence
  // The one number the whole scene is built around. The reference filled to a
  // sample 5.2%; this fills to whatever the analyzer actually returned.
  const probability = clamp(result.overall_probability, 0, 1)
  const fillAt = tween(0, probability, start + 0.5, start + 3.5, easeOutCubic)

  const visibility = useShot(clock, settleAt, start - 0.6, end + 0.7)
  const opacity = useFrameValue(clock, settleAt, (T) => {
    const enter = MOTION.enter(T, start + 0.2, start + 1.2)
    const exit = 1 - MOTION.enter(T, end + 0.1, end + 0.7)
    return enter * (T < end ? 1 : exit)
  })
  const scale = useFrameValue(clock, settleAt, (T) => MOTION.pop(T, start + 0.2, start + 1.1))
  const background = useFrameValue(clock, settleAt, (T) => {
    const degrees = fillAt(T) * 360
    return `conic-gradient(var(--amber-hot) ${degrees}deg, var(--ink-700) ${degrees}deg 360deg)`
  })
  const boxShadow = useFrameValue(
    clock,
    settleAt,
    (T) => `0 0 60px 6px rgba(${AMBER_HOT_RGB}, ${0.18 + fillAt(T)})`,
  )
  const readout = useFrameValue(clock, settleAt, (T) => `${(fillAt(T) * 100).toFixed(1)}%`)

  return (
    <motion.div
      className="showcase__scene showcase__scene--gauge"
      style={{ visibility, opacity }}
    >
      <motion.div
        className="showcase__dial"
        style={{ scale, background, boxShadow, rotateX: 46 }}
      >
        <div className="showcase__dial-face">
          <motion.span className="showcase__percent">{readout}</motion.span>
          <span className="showcase__percent-caption">AI likelihood</span>
        </div>
      </motion.div>

      {/* A leaning, not a verdict — the API's own tag, which always says "Likely". */}
      <p className="showcase__pill">{result.label}</p>

      {/* The sentence is the result; the dial is how it arrives. */}
      <p className="showcase__sentence">{result.confidence_label}</p>
    </motion.div>
  )
}

const DEFAULT_EVIDENCE: EvidenceBar[] = [
  { name: 'Perplexity', value: 0.72 },
  { name: 'Burstiness', value: 0.38 },
  { name: 'Stylistic', value: 0.51 },
]

function EvidenceBarView({
  bar,
  index,
  clock,
  settleAt,
}: {
  bar: EvidenceBar
  index: number
  clock: MotionValue<number>
  settleAt: number | null
}) {
  const growFrom = CUE.evidence + 0.5 + index * 0.2
  const height = useFrameValue(clock, settleAt, (T) => {
    const grow = MOTION.draw(T, growFrom, growFrom + 1.7)
    return `${grow * clamp(bar.value, 0, 1) * 100}%`
  })

  return (
    <div className="showcase__bar">
      <div className="showcase__bar-track">
        <motion.div className="showcase__bar-fill" style={{ height }} />
      </div>
      <p className="showcase__bar-name">{bar.name}</p>
    </div>
  )
}

function EvidenceScene({
  clock,
  settleAt,
  evidence,
}: {
  clock: MotionValue<number>
  settleAt: number | null
  evidence: EvidenceBar[]
}) {
  const start = CUE.evidence
  const end = CUE.close

  const visibility = useShot(clock, settleAt, start - 0.4, end + 0.6)
  const opacity = useFrameValue(clock, settleAt, (T) => {
    const enter = MOTION.enter(T, start, start + 0.8)
    const exit = 1 - MOTION.enter(T, end + 0.1, end + 0.6)
    return enter * (T < end ? 1 : exit)
  })

  return (
    <motion.div
      className="showcase__scene showcase__scene--evidence"
      style={{ visibility, opacity }}
    >
      <p className="showcase__evidence-label">
        Sentence-level evidence, not a bare percentage
      </p>
      <div className="showcase__bars">
        {evidence.map((bar, i) => (
          <EvidenceBarView
            key={bar.name}
            bar={bar}
            index={i}
            clock={clock}
            settleAt={settleAt}
          />
        ))}
      </div>
    </motion.div>
  )
}

function CloseScene({
  clock,
  settleAt,
}: {
  clock: MotionValue<number>
  settleAt: number | null
}) {
  const start = CUE.close

  const visibility = useShot(clock, settleAt, start - 0.3, LOOP_SECONDS)
  // Settles back to nothing just before the loop point, so the last authored
  // frame matches the first and the seam does not read as a cut.
  const opacity = useFrameValue(
    clock,
    settleAt,
    (T) =>
      MOTION.enter(T, start + 0.2, start + 1.3) *
      (1 - MOTION.enter(T, LOOP_SECONDS - 0.8, LOOP_SECONDS - 0.05)),
  )

  return (
    <motion.div
      className="showcase__scene showcase__scene--close"
      style={{ visibility, opacity }}
    >
      <p className="showcase__headline">
        Measured signals, not a second opinion from a chatbot.
      </p>
      {/* A picture of the CTA, not the CTA — see the note at the top. */}
      <span className="showcase__cta">Analyze an essay</span>
    </motion.div>
  )
}

/* ---------------------------------------------------------------------------
   The piece.
--------------------------------------------------------------------------- */

/**
 * Where each scene parks when the loop cannot run. Every one is a moment
 * after that scene has fully arrived and before it starts to leave.
 */
const SETTLED: Record<SceneName, number> = {
  opening: CUE.opening + 2,
  signals: CUE.signals + 3,
  gauge: CUE.gauge + 4,
  evidence: CUE.evidence + 3,
  close: CUE.close + 2,
}

export function ShowcaseIntro({
  result,
  evidence = DEFAULT_EVIDENCE,
  particles = true,
  glowIntensity = 1,
  className = '',
}: ShowcaseIntroProps) {
  const canAnimate = useCanAnimate()
  const clock = useMotionValue(0)
  const glow = clamp(glowIntensity, 0, 2)

  // One decision, read once per mount, passed down as each scene's settled
  // time. Null is the only value that means "play".
  const settle = (scene: SceneName) => (canAnimate ? null : SETTLED[scene])

  return (
    <div
      className={`showcase ${canAnimate ? '' : 'is-static'} ${className}`.trim()}
      role="img"
      aria-label={`AI Essay Detector — ${result.label}. ${result.confidence_label}`}
    >
      {canAnimate && <LoopClock clock={clock} />}

      <div className="showcase__backdrop" />
      <div className="showcase__scrim" />

      {particles && (
        <ParticleField clock={clock} settleAt={settle('opening')} glow={glow} />
      )}

      <OpeningScene clock={clock} settleAt={settle('opening')} />
      <SignalScene clock={clock} settleAt={settle('signals')} />
      <GaugeScene clock={clock} settleAt={settle('gauge')} result={result} />
      <EvidenceScene clock={clock} settleAt={settle('evidence')} evidence={evidence} />
      <CloseScene clock={clock} settleAt={settle('close')} />
    </div>
  )
}
