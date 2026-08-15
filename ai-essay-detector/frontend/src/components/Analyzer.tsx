import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  ApiError,
  analyzeEssay,
  blocksSubmission,
  countWords,
  lengthState,
  WORD_LIMITS,
} from '../api'
import type { AnalyzeResponse } from '../types'
import { useCanAnimate } from '../lib/useCanAnimate'
import { ProbabilityGauge } from './ProbabilityGauge'
import { EssayHighlights } from './EssayHighlights'

interface Props {
  onBack: () => void
}

type Status = 'idle' | 'loading' | 'done' | 'error'

export function Analyzer({ onBack }: Props) {
  const [essay, setEssay] = useState('')
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  const canAnimate = useCanAnimate()
  const reduceMotion = !canAnimate
  // Entrance/exit target for the reason panel; undefined disables it entirely.
  const panelMotion = canAnimate ? { opacity: 0, y: 4 } : undefined
  const abortRef = useRef<AbortController | null>(null)
  const panelId = useId()
  const counterId = useId()

  const words = useMemo(() => countWords(essay), [essay])
  const length = lengthState(words)
  const lengthBlocks = blocksSubmission(length)
  const isLoading = status === 'loading'
  // Functionally disabled, not merely styled: a second submit cannot be
  // dispatched while one is in flight, so two responses can't race.
  const submitDisabled = isLoading || lengthBlocks

  useEffect(() => () => abortRef.current?.abort(), [])

  // Continues the "<n> words" already rendered beside it, so the count is
  // stated once rather than repeated back in every branch.
  const lengthMessage = (() => {
    switch (length) {
      case 'empty':
        return null
      case 'too-short':
        return `${WORD_LIMITS.MIN - words} more needed. Below ${WORD_LIMITS.MIN} words there aren't enough sentences for the per-sentence signals to mean anything.`
      case 'long':
        return `longer than a typical admissions essay. This will still be analysed, but scoring time grows with length.`
      case 'too-long':
        return `over the ${WORD_LIMITS.HARD_MAX.toLocaleString()} word maximum. Submit a single essay rather than a combined document.`
      default:
        return null
    }
  })()

  const onSubmit = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault()
      if (submitDisabled) return

      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setStatus('loading')
      setError(null)
      setOpenIndex(null)

      try {
        const data = await analyzeEssay(essay, controller.signal)
        setResult(data)
        setStatus('done')
      } catch (cause) {
        if (cause instanceof DOMException && cause.name === 'AbortError') return
        // Prefer the backend's own wording — its 400s already say precisely
        // what is wrong, which beats a generic failure message.
        setError(
          cause instanceof ApiError
            ? cause.message
            : 'Something went wrong running the analysis.',
        )
        setResult(null)
        setStatus('error')
      } finally {
        abortRef.current = null
      }
    },
    [essay, submitDisabled],
  )

  // Announced once per resolution. Assertive would interrupt; polite lets a
  // screen reader finish its sentence and then report the outcome.
  const announcement =
    status === 'loading'
      ? 'Analyzing essay, please wait.'
      : status === 'done' && result
        ? `Analysis complete. ${result.confidence_label} ${result.flagged_sentence_count} of ${result.sentences.length} sentences flagged.`
        : status === 'error' && error
          ? `Analysis failed. ${error}`
          : ''

  // Keyed off "any sentence unscored" rather than "all of them". The backend
  // skips localization for the whole essay or not at all, so today these agree
  // — but a partially-localized response would still get its hatched sentences
  // explained instead of leaving them silently unaccounted for. The reason is
  // read off an actually-unscored sentence, so a flagged sentence's reason can
  // never be shown here in its place. No trailing period: the notice supplies
  // one, and the backend's text does not carry its own.
  const firstUnscored = result?.sentences.find((s) => s.local_score === null) ?? null
  const unscoredReason = firstUnscored
    ? (firstUnscored.reason ?? 'This essay had too few sentences to score individually')
    : null

  const openSentence =
    openIndex !== null
      ? (result?.sentences.find((s) => s.index === openIndex) ?? null)
      : null

  return (
    <main className="analyzer">
      <button type="button" className="back" onClick={onBack}>
        ← Back
      </button>

      <h1>Analyze an essay</h1>

      <form onSubmit={onSubmit}>
        <label htmlFor="essay" className="field-label">
          Paste the essay
        </label>
        <div className="editor">
          <textarea
            id="essay"
            className="editor__input"
            value={essay}
            onChange={(e) => setEssay(e.target.value)}
            placeholder="Paste the full essay here…"
            rows={12}
            spellCheck={false}
            aria-describedby={counterId}
            aria-invalid={lengthBlocks && length !== 'empty'}
            disabled={isLoading}
          />

          {/* Scanning overlay. Static under reduced motion. */}
          <AnimatePresence>
            {isLoading && (
              <motion.div
                className="scanoverlay"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                aria-hidden="true"
              >
                {reduceMotion ? (
                  <span className="scanoverlay__bar is-parked" />
                ) : (
                  <motion.span
                    className="scanoverlay__bar"
                    initial={{ top: '0%' }}
                    animate={{ top: ['0%', '100%', '0%'] }}
                    transition={{ duration: 2.2, repeat: Infinity, ease: 'easeInOut' }}
                  />
                )}
                <span className="scanoverlay__label">Analyzing…</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="editor__footer">
          <p id={counterId} className={`counter is-${length}`}>
            <strong>{words.toLocaleString()}</strong> words
            {lengthMessage && <span className="counter__note"> — {lengthMessage}</span>}
          </p>

          <button type="submit" className="cta" disabled={submitDisabled}>
            {isLoading ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
      </form>

      {/*
        One region for every resolution — completion and failure both land
        here, so a screen reader user never has to poll the page to find out
        whether the request finished.
      */}
      <div className="visually-hidden" role="status" aria-live="polite" aria-atomic="true">
        {announcement}
      </div>

      {/*
        Visible twin of the announcement above. Deliberately NOT role="alert":
        the polite region already carries this exact text, and an assertive
        alert on top of it makes a screen reader read the failure twice.
      */}
      {status === 'error' && error && (
        <p className="alert">
          <strong>Analysis failed.</strong> {error}
        </p>
      )}

      {status === 'done' && result && (
        <section className="results" aria-label="Analysis result">
          <ProbabilityGauge
            probability={result.overall_probability}
            confidenceLabel={result.confidence_label}
            label={result.label}
          />

          <div className="results__body">
            <div className="legend">
              <span className="legend__label">Per-sentence signal</span>
              <span className="legend__ramp" aria-hidden="true">
                <i style={{ '--intensity': '0' } as React.CSSProperties} />
                <i style={{ '--intensity': '0.33' } as React.CSSProperties} />
                <i style={{ '--intensity': '0.66' } as React.CSSProperties} />
                <i style={{ '--intensity': '1' } as React.CSSProperties} />
              </span>
              <span className="legend__scale">
                closer to this essay's norm → further from it
              </span>
              <span className="legend__flag">
                <span className="sentence__marker" aria-hidden="true">
                  ▲
                </span>{' '}
                flagged — select for the reason
              </span>
            </div>

            {unscoredReason && (
              <p className="notice">
                <strong>Not scored per sentence.</strong> {unscoredReason}. The
                essay-level probability above is unaffected — it never depended on
                per-sentence variance.
              </p>
            )}

            {result.flagged_sentence_count === 0 && !unscoredReason && (
              <p className="notice">
                <strong>No individual sentences stood out.</strong> Either nothing
                deviated far enough from this essay's own norm, or the essay itself
                didn't score AI-leaning — flags are only raised inside an essay
                that did.
              </p>
            )}

            <EssayHighlights
              sentences={result.sentences}
              openIndex={openIndex}
              onOpen={setOpenIndex}
              onClose={() => setOpenIndex(null)}
              panelId={panelId}
            />

            {/*
              Always in the DOM so aria-controls has a valid target. The panel
              is supplementary: every reason is already on its sentence's
              aria-label, so nothing here is the only route to the evidence.
            */}
            <div id={panelId} className="panel" aria-live="polite">
              {/*
                The container stays mounted so aria-controls always has a valid
                target and the live region is never torn down mid-announcement;
                only the contents animate. `mode="wait"` keeps the old reason
                from overlapping the new one when moving between sentences.
              */}
              <AnimatePresence mode="wait" initial={false}>
                {openSentence ? (
                  <motion.div
                    key={openSentence.index}
                    initial={panelMotion}
                    animate={panelMotion ? { opacity: 1, y: 0 } : undefined}
                    exit={panelMotion ? { opacity: 0, y: -4 } : undefined}
                    transition={{ duration: 0.18, ease: 'easeOut' }}
                  >
                    <div className="panel__head">
                      <h2>Why this sentence</h2>
                      <button
                        type="button"
                        className="panel__close"
                        onClick={() => setOpenIndex(null)}
                      >
                        Close
                      </button>
                    </div>
                    <blockquote>{openSentence.text}</blockquote>
                    <p className="panel__reason">{openSentence.reason}</p>
                    <dl className="panel__stats">
                      <div>
                        <dt>Deviation from this essay's norm</dt>
                        <dd>{openSentence.local_score?.toFixed(2) ?? '—'}</dd>
                      </div>
                      <div>
                        <dt>Predictability (z)</dt>
                        <dd>{openSentence.ppl_z?.toFixed(2) ?? '—'}</dd>
                      </div>
                      <div>
                        <dt>Length (z)</dt>
                        <dd>{openSentence.length_z?.toFixed(2) ?? '—'}</dd>
                      </div>
                    </dl>
                    <p className="panel__hint">Press Escape to close.</p>
                  </motion.div>
                ) : (
                  <motion.p key="empty" className="panel__empty" initial={false}>
                    Select a flagged sentence to see why it was flagged.
                  </motion.p>
                )}
              </AnimatePresence>
            </div>

            <p className="results__meta">
              Model <code>{result.model_version}</code> · scored in{' '}
              {Math.round(result.processing_ms)} ms ·{' '}
              {result.flagged_sentence_count} of {result.sentences.length} sentences
              flagged
            </p>
          </div>
        </section>
      )}
    </main>
  )
}
