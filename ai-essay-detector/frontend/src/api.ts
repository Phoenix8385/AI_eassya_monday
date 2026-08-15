import type { AnalyzeResponse, ApiErrorBody } from './types'

/**
 * The backend base URL. Set VITE_API_URL for a deployed backend; the default
 * matches `uvicorn app.main:app --port 8000` locally.
 */
const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(
  /\/$/,
  '',
)

/** Word count, matching the backend's `len(essay.split())` exactly. */
export function countWords(text: string): number {
  const trimmed = text.trim()
  return trimmed === '' ? 0 : trimmed.split(/\s+/).length
}

/**
 * Bounds mirrored from the backend so a person gets feedback while typing
 * rather than after waiting on a request that was always going to 400.
 *
 * MIN and HARD_MAX are the backend's real limits (MIN_ESSAY_WORDS,
 * MAX_ESSAY_WORDS). SOFT_MAX is a UI-only advisory: 2,000 words is far past a
 * real admissions essay and scoring time grows with length, so it is worth
 * warning about — but the backend will still accept it, so it must not block.
 */
export const WORD_LIMITS = {
  MIN: 50,
  SOFT_MAX: 2_000,
  HARD_MAX: 20_000,
} as const

export type LengthState = 'empty' | 'too-short' | 'ok' | 'long' | 'too-long'

export function lengthState(words: number): LengthState {
  if (words === 0) return 'empty'
  if (words < WORD_LIMITS.MIN) return 'too-short'
  if (words > WORD_LIMITS.HARD_MAX) return 'too-long'
  if (words > WORD_LIMITS.SOFT_MAX) return 'long'
  return 'ok'
}

/** True when the backend would reject this length outright. */
export function blocksSubmission(state: LengthState): boolean {
  return state === 'empty' || state === 'too-short' || state === 'too-long'
}

export class ApiError extends Error {
  readonly status: number | null
  constructor(message: string, status: number | null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function isErrorBody(value: unknown): value is ApiErrorBody {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as ApiErrorBody).error === 'string'
  )
}

/**
 * POST /analyze.
 *
 * Surfaces the backend's own message wherever it sends one — its 400s already
 * say exactly what is wrong ("Essay is 20 words — at least 50 words are
 * needed…"), which is far more useful than replacing it with a generic
 * "something went wrong".
 */
export async function analyzeEssay(
  essay: string,
  signal?: AbortSignal,
): Promise<AnalyzeResponse> {
  let response: Response
  try {
    response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ essay }),
      signal,
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    // Network-level: server down, DNS, CORS rejection, offline.
    throw new ApiError(
      `Couldn't reach the analyzer at ${API_URL}. Check that the backend is running and that its CORS_ORIGIN allows this page.`,
      null,
    )
  }

  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    // Fall through — an empty or non-JSON body is handled below.
  }

  if (!response.ok) {
    if (isErrorBody(body)) throw new ApiError(body.error, response.status)
    if (response.status === 429) {
      throw new ApiError(
        'Too many requests in a short window. Wait a moment and try again.',
        429,
      )
    }
    throw new ApiError(
      `The analyzer returned ${response.status} ${response.statusText || ''}`.trim(),
      response.status,
    )
  }

  if (!body || typeof body !== 'object' || !('overall_probability' in body)) {
    throw new ApiError(
      'The analyzer returned a response this page could not read.',
      response.status,
    )
  }

  return body as AnalyzeResponse
}
