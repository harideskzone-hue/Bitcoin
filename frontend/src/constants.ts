// frontend/src/constants.ts
// ============================
// Single source of truth for risk-label thresholds and colours on the frontend.
//
// Per SIH26146_Final_Backlog_v5.1_FROZEN.md correction #28:
//   "Backend and frontend MUST import from these constants, never hard-code inline."
//
// Backend equivalent: backend/constants.py → RISK_LABEL_THRESHOLDS
// These values MUST stay in sync with backend/constants.py.

export type RiskLabel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

// ── Risk-label thresholds ───────────────────────────────────────────────────
// Presentation categories for prioritization consistency.
// NOT probability thresholds. NOT enforcement decision rules.
// See: backend/constants.py comment block for full rationale.
export const RISK_LABEL_THRESHOLDS: Record<RiskLabel, [number, number]> = {
  LOW:      [0,  24],   // informational; monitor
  MEDIUM:   [25, 49],   // elevated; review when capacity allows
  HIGH:     [50, 74],   // prioritised; review within 24h
  CRITICAL: [75, 100],  // urgent; review immediately
}

export const RISK_LABEL_ORDER: RiskLabel[] = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']

export function scoreToLabel(score: number): RiskLabel {
  if (score <= 24)  return 'LOW'
  if (score <= 49)  return 'MEDIUM'
  if (score <= 74)  return 'HIGH'
  return 'CRITICAL'
}

// ── Risk-label colours ────────────────────────────────────────────────────────
// Must match backend/constants.py → RISK_LABEL_COLOURS
// Must match index.css CSS variables (--risk-low, --risk-medium, etc.)
export const RISK_COLOURS: Record<RiskLabel, string> = {
  LOW:      '#22c55e',
  MEDIUM:   '#eab308',
  HIGH:     '#f97316',
  CRITICAL: '#ef4444',
}

export const RISK_BG_COLOURS: Record<RiskLabel, string> = {
  LOW:      'rgba(34, 197, 94,  0.12)',
  MEDIUM:   'rgba(234, 179, 8,  0.12)',
  HIGH:     'rgba(249, 115, 22, 0.12)',
  CRITICAL: 'rgba(239, 68,  68, 0.12)',
}

// ── Score disclaimer ──────────────────────────────────────────────────────────
// Must match backend/constants.py → SCORE_DISCLAIMER (P0.4.7)
export const SCORE_DISCLAIMER =
  'Model-derived risk ranking. For prioritization and human review only. ' +
  'Not a calibrated probability.'

// ── API defaults ─────────────────────────────────────────────────────────────
// Per frozen spec: min_risk default = 70 (not 50)
export const DEFAULT_MIN_RISK = 70
export const DEFAULT_ALERT_LIMIT = 20
