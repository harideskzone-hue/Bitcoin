// frontend/src/components/RiskBadge.tsx
// Risk score badge + ring — used by AlertList, WalletDetail

import type { RiskLabel } from '../api'

interface Props {
  label: RiskLabel
  score?: number
  size?: 'sm' | 'md' | 'lg'
}

const COLOURS: Record<RiskLabel, string> = {
  LOW:      'var(--risk-low)',
  MEDIUM:   'var(--risk-medium)',
  HIGH:     'var(--risk-high)',
  CRITICAL: 'var(--risk-critical)',
}

export function RiskBadge({ label, score, size = 'md' }: Props) {
  const colour = COLOURS[label] ?? 'var(--text-muted)'
  return (
    <span className={`badge badge-${label}`} style={{ fontSize: size === 'sm' ? '0.68rem' : undefined }}>
      <span style={{
        width: 7, height: 7, borderRadius: '50%',
        background: colour, display: 'inline-block', flexShrink: 0,
        boxShadow: `0 0 6px ${colour}`,
      }} />
      {label}{score !== undefined && ` · ${score}`}
    </span>
  )
}

// ── Score ring (SVG progress circle) ──────────────────────────────────────

interface RingProps { score: number; label: RiskLabel }

export function ScoreRing({ score, label }: RingProps) {
  const r  = 28
  const cx = 36
  const circumference = 2 * Math.PI * r
  const offset = circumference - (score / 100) * circumference
  const colour = COLOURS[label]

  return (
    <div className="score-ring">
      <svg width="72" height="72" viewBox="0 0 72 72">
        {/* Track */}
        <circle cx={cx} cy={cx} r={r} fill="none" stroke="var(--border)" strokeWidth="6" />
        {/* Progress */}
        <circle
          cx={cx} cy={cx} r={r} fill="none"
          stroke={colour} strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 0.6s ease, stroke 0.3s ease', filter: `drop-shadow(0 0 4px ${colour})` }}
        />
      </svg>
      <div className="score-num" style={{ color: colour, fontSize: '1rem' }}>
        {score}
      </div>
    </div>
  )
}
