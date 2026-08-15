/**
 * Mirrors the backend's actual response shape (app/models/schemas.py).
 *
 * Note what is NOT here: there is no `overall_score`. The API reports a
 * probability plus the sentence that frames it, and the UI is built around
 * that pairing — a bare number never ships without `confidence_label`.
 */

/** Short tag shown beside the probability. Never on its own. */
export type Label = 'Likely AI' | 'Likely Human'

export interface Sentence {
  index: number
  text: string
  start_char: number
  end_char: number
  word_count: number
  perplexity: number
  mean_logprob: number
  /** Corpus-calibrated 0–1 score. Distinct from local_score. */
  ai_likelihood: number
  /** Resemblance phrase — always present, so a score never stands alone. */
  phrase: string
  flagged: boolean
  /**
   * Templated explanation, or null when nothing cleared the threshold.
   * Also carries the "too few sentences" notice when localization was skipped.
   */
  reason: string | null
  /**
   * Deviation from THIS essay's own norm, on a z-scale.
   * `null` when the essay had too few sentences to localize against
   * (backend: MIN_SENTENCES_FOR_LOCALIZATION, default 4).
   */
  local_score: number | null
  ppl_z: number | null
  length_z: number | null
}

export interface AnalyzeResponse {
  overall_probability: number
  confidence_label: string
  label: Label
  confidence: number
  sentences: Sentence[]
  flagged_sentence_count: number
  model_version: string
  processing_ms: number
}

/** Every non-2xx from the backend uses this envelope. */
export interface ApiErrorBody {
  error: string
}
