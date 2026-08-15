/**
 * Renders the result components with NO browser and NO animation frames, then
 * asserts the content is already complete in the markup.
 *
 * This is the progressive-enhancement requirement stated directly: the stagger
 * is an entrance effect on top of correct content, never the mechanism that
 * makes content exist. If this file fails, a reader whose animation never ran
 * is looking at a blank result panel.
 *
 * Run with `npm run check:ssr`. It lives outside `src/` on purpose — it is a
 * Node program, not part of the browser bundle.
 */
import { renderToStaticMarkup } from 'react-dom/server'
import { EssayHighlights } from '../src/components/EssayHighlights'
import { ProbabilityGauge } from '../src/components/ProbabilityGauge'
import { Landing } from '../src/components/Landing'
import type { Sentence } from '../src/types'

function sentence(over: Partial<Sentence> & { index: number; text: string }): Sentence {
  return {
    start_char: 0,
    end_char: 0,
    word_count: 12,
    perplexity: 40,
    mean_logprob: -3.5,
    ai_likelihood: 0.3,
    phrase: 'resembles human-written examples',
    flagged: false,
    reason: null,
    local_score: 0.4,
    ppl_z: -0.3,
    length_z: 0.2,
    ...over,
  }
}

const scored: Sentence[] = [
  sentence({ index: 0, text: 'First sentence of the essay.', local_score: 0.2 }),
  sentence({
    index: 1,
    text: 'A flagged sentence that stood out.',
    flagged: true,
    local_score: 1.9,
    reason:
      "This sentence resembles the AI-pattern examples in the training data — it's unusually predictable word-to-word compared to the rest of the essay.",
  }),
  sentence({ index: 2, text: 'Third sentence here.', local_score: 0.9 }),
]

const unscored: Sentence[] = [
  sentence({
    index: 0,
    text: 'Only three sentences here.',
    local_score: null,
    ppl_z: null,
    length_z: null,
    reason: 'Too few sentences in this essay to reliably localize signals',
  }),
  sentence({
    index: 1,
    text: 'So nothing is localized.',
    local_score: null,
    ppl_z: null,
    length_z: null,
    reason: 'Too few sentences in this essay to reliably localize signals',
  }),
]

const noop = () => {}
const results: string[] = []
let failures = 0

function check(name: string, pass: boolean, detail = '') {
  results.push(`  ${pass ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`)
  if (!pass) failures += 1
}

// --- Sentences ------------------------------------------------------------
const html = renderToStaticMarkup(
  <EssayHighlights
    sentences={scored}
    openIndex={null}
    onOpen={noop}
    onClose={noop}
    panelId="panel"
  />,
)

check('every sentence text is in the markup', scored.every((s) => html.includes(s.text)))
check(
  'flagged sentence is a real button',
  html.includes('role="button"') && html.includes('tabindex="0"'),
)
check('aria-expanded present', html.includes('aria-expanded="false"'))
check('aria-controls points at the panel', html.includes('aria-controls="panel"'))
check(
  'reason reachable without opening the panel',
  html.includes('unusually predictable word-to-word'),
)
check(
  'flagged carries a non-colour cue (marker glyph)',
  html.includes('sentence__marker'),
)
check(
  'intensity is a per-sentence custom property',
  html.includes('--intensity:1.000') || html.includes('--intensity:0.760'),
  (html.match(/--intensity:[\d.]+/g) ?? []).join(' '),
)
// The critical pair: with no JS runtime to run the entrance animation, nothing
// may be left hidden by that animation's start state.
check('no opacity:0 anywhere in the static markup', !/opacity:\s*0(?![.\d])/.test(html))
check('no entrance transform left applied', !/translateY/.test(html))

// --- Null local_score -----------------------------------------------------
const unscoredHtml = renderToStaticMarkup(
  <EssayHighlights
    sentences={unscored}
    openIndex={null}
    onOpen={noop}
    onClose={noop}
    panelId="panel"
  />,
)
check(
  'null-score sentences still render their text',
  unscoredHtml.includes('Only three sentences here.'),
)
check('null-score uses the distinct unscored style', unscoredHtml.includes('is-unscored'))
check(
  'null-score sets no intensity (not treated as zero signal)',
  !unscoredHtml.includes('--intensity'),
)

// --- Gauge ----------------------------------------------------------------
const gaugeHtml = renderToStaticMarkup(
  <ProbabilityGauge
    probability={0.87}
    confidenceLabel="This essay shares 87% of its measured patterns with AI-generated examples in the training data."
    label="Likely AI"
  />,
)
check(
  'gauge exposes the full sentence as its accessible name',
  gaugeHtml.includes('aria-label="This essay shares 87%'),
)
check(
  'confidence_label sentence is rendered as visible text',
  gaugeHtml.includes('<figcaption'),
)
check('short tag rendered alongside, not alone', gaugeHtml.includes('Likely AI'))

// --- Landing --------------------------------------------------------------
const landingHtml = renderToStaticMarkup(<Landing onStart={noop} />)
check('landing states the mechanism (GPT-2)', landingHtml.includes('GPT-2'))
check('landing names burstiness', landingHtml.includes('burstiness'))
check(
  'landing rules out an LLM verdict explicitly',
  landingHtml.includes('No language model is asked whether the essay is AI-written'),
)
check('landing has the CTA', landingHtml.includes('Analyze an essay'))
check(
  'decorative visual is hidden from assistive tech',
  landingHtml.includes('aria-hidden="true"'),
)

console.log(results.join('\n'))
console.log(failures === 0 ? '\nALL SSR/DOM CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`)
if (failures > 0) process.exit(1)
