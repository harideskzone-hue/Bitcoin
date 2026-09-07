// frontend/src/components/AlertList.tsx
// Track 3 — AlertList with live threshold slider + real-time stats panel

import { useEffect, useRef, useState } from 'react'
import type { WalletAlert } from '../api'
import { api } from '../api'
import { DEFAULT_MIN_RISK, RISK_COLOURS } from '../constants'
import { RiskBadge } from './RiskBadge'

interface Props {
  onSelect: (address: string) => void
  selected: string | null
}

// Pre-computed stats at various thresholds (based on 17,660 heuristic-scored addresses)
// Mirrors the percentile bands from heuristic_scorer.py
const THRESHOLD_STATS: Record<number, { count: number; label: string; description: string }> = {
  0:  { count: 17660, label: 'ALL',      description: 'All 17,660 addresses in snapshot' },
  25: { count: 13333, label: 'MEDIUM+',  description: 'Medium, High, and Critical risk addresses' },
  50: { count: 4416,  label: 'HIGH+',    description: 'High and Critical risk addresses' },
  70: { count: 810,   label: 'CRITICAL', description: 'Highest-confidence risk addresses' },
  75: { count: 810,   label: 'CRITICAL', description: 'CRITICAL band — review immediately' },
  90: { count: 323,   label: 'TOP 2%',   description: 'Extremely high-confidence — top 2% of addresses' },
  95: { count: 88,    label: 'TOP 0.5%', description: 'Absolute highest-ranked addresses' },
}

function getStatsForThreshold(t: number) {
  const keys = Object.keys(THRESHOLD_STATS).map(Number).sort((a, b) => b - a)
  for (const k of keys) {
    if (t >= k) return THRESHOLD_STATS[k]
  }
  return THRESHOLD_STATS[0]
}

export function AlertList({ onSelect, selected }: Props) {
  const [alerts, setAlerts]         = useState<WalletAlert[]>([])
  const [disclaimer, setDisclaimer] = useState('')
  const [loading, setLoading]       = useState(true)
  const [error, setError]           = useState<string | null>(null)
  const [minRisk, setMinRisk]       = useState(DEFAULT_MIN_RISK)
  const [totalCount, setTotal]      = useState(0)
  const debounceRef                 = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setLoading(true)
      setError(null)
      api.alerts({ min_risk: minRisk, limit: 50 })
        .then(r => {
          setAlerts(r.alerts)
          setDisclaimer(r.score_disclaimer)
          setTotal(r.total ?? r.alerts.length)
        })
        .catch(e => setError(e.message))
        .finally(() => setLoading(false))
    }, 150)
  }, [minRisk])

  const stats = getStatsForThreshold(minRisk)

  // Band colour for the live count pill
  const pillColour = minRisk >= 75 ? 'var(--risk-critical)'
    : minRisk >= 50 ? 'var(--risk-high)'
    : minRisk >= 25 ? 'var(--risk-medium)'
    : 'var(--risk-low)'
  const pillBg = minRisk >= 75 ? 'var(--risk-critical-bg)'
    : minRisk >= 50 ? 'var(--risk-high-bg)'
    : minRisk >= 25 ? 'var(--risk-medium-bg)'
    : 'var(--risk-low-bg)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between mb-3">
          <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Alert Feed</span>
          <span style={{
            background: pillBg, color: pillColour,
            border: `1px solid ${pillColour}`, borderRadius: 999,
            fontSize: '0.70rem', fontWeight: 700, padding: '2px 9px',
            transition: 'all 0.2s ease',
          }}>
            {totalCount > 0 ? totalCount : alerts.length} above threshold
          </span>
        </div>

        {/* ── Threshold slider ─────────────────────────────────────────────── */}
        <div style={{ marginBottom: 8 }}>
          <div className="flex items-center gap-2" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: 4 }}>
            <span>Risk threshold</span>
            <input
              type="range" min={0} max={100} step={1} value={minRisk}
              onChange={e => setMinRisk(Number(e.target.value))}
              style={{ flex: 1, accentColor: pillColour, cursor: 'pointer', transition: 'accent-color 0.2s' }}
            />
            <span style={{
              fontFamily: 'JetBrains Mono, monospace',
              minWidth: 28, color: pillColour, fontWeight: 700, fontSize: '0.85rem',
              transition: 'color 0.2s',
            }}>{minRisk}</span>
          </div>

          {/* ── Live stats bar ───────────────────────────────────────────────── */}
          <div style={{
            background: 'var(--bg-card)',
            border: `1px solid ${pillColour}22`,
            borderRadius: 8,
            padding: '8px 12px',
            fontSize: '0.72rem',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            transition: 'border-color 0.2s',
          }}>
            {/* Band indicator */}
            <span style={{
              background: pillBg, color: pillColour,
              border: `1px solid ${pillColour}`,
              borderRadius: 6,
              padding: '2px 7px',
              fontWeight: 700,
              fontSize: '0.68rem',
              flexShrink: 0,
              transition: 'all 0.2s',
            }}>{stats.label}</span>

            {/* Description */}
            <span style={{ color: 'var(--text-secondary)', flex: 1 }}>
              {stats.description}
            </span>

            {/* Mini distribution bar */}
            <div style={{ display: 'flex', gap: 2, flexShrink: 0, alignItems: 'center' }}>
              {[
                { w: 40, colour: RISK_COLOURS.LOW,      label: 'L' },
                { w: 35, colour: RISK_COLOURS.MEDIUM,   label: 'M' },
                { w: 20, colour: RISK_COLOURS.HIGH,     label: 'H' },
                { w: 5,  colour: RISK_COLOURS.CRITICAL, label: 'C' },
              ].map(({ w, colour, label }) => (
                <div
                  key={label}
                  title={`${label}: ${w}%`}
                  style={{
                    width: w * 0.6,
                    height: 10,
                    borderRadius: 3,
                    background: colour,
                    opacity: minRisk <= (label === 'C' ? 75 : label === 'H' ? 50 : label === 'M' ? 25 : 0) ? 1 : 0.25,
                    transition: 'opacity 0.2s',
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Disclaimer strip ───────────────────────────────────────────────── */}
      {disclaimer && (
        <div style={{
          padding: '7px 14px',
          background: 'rgba(99,102,241,0.08)',
          borderBottom: '1px solid var(--border)',
          fontSize: '0.70rem', color: 'var(--text-muted)',
          fontStyle: 'italic', lineHeight: 1.4,
        }}>
          {disclaimer}
        </div>
      )}

      {/* ── Content ───────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
        {loading && (
          <div style={{ padding: 24 }}>
            {[...Array(5)].map((_, i) => (
              <div key={i} className="loading-shimmer mb-3" style={{ height: 64, borderRadius: 10 }} />
            ))}
          </div>
        )}

        {error && (
          <div className="empty-state">
            <div className="icon">⚠️</div>
            <div style={{ fontSize: '0.85rem' }}>{error}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Ensure the API is running: <code>USE_SQLITE_FALLBACK=true uvicorn backend.app.main:app</code>
            </div>
          </div>
        )}

        {!loading && !error && alerts.length === 0 && (
          <div className="empty-state">
            <div className="icon">🔍</div>
            <div>No addresses above risk {minRisk}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 6 }}>
              Lower the threshold to see more addresses
            </div>
          </div>
        )}

        {!loading && !error && alerts.map(alert => (
          <button
            key={alert.address}
            onClick={() => onSelect(alert.address)}
            className={`card ${selected === alert.address ? 'selected' : ''}`}
            style={{
              width: '100%', display: 'flex', alignItems: 'center',
              gap: 10, padding: '10px 12px', marginBottom: 6,
              cursor: 'pointer', textAlign: 'left', border: 'none',
              background: selected === alert.address ? 'var(--bg-hover)' : 'var(--bg-card)',
              transition: 'background 0.15s, transform 0.1s',
            }}
            onMouseEnter={e => (e.currentTarget.style.transform = 'translateX(2px)')}
            onMouseLeave={e => (e.currentTarget.style.transform = 'translateX(0)')}
          >
            {/* Score bubble */}
            <div style={{
              width: 44, height: 44, borderRadius: 10, flexShrink: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 800, fontSize: '0.95rem',
              background: `var(--risk-${alert.risk_label.toLowerCase()}-bg)`,
              color: `var(--risk-${alert.risk_label.toLowerCase()})`,
              border: `1px solid var(--risk-${alert.risk_label.toLowerCase()})`,
              boxShadow: `0 0 12px var(--risk-${alert.risk_label.toLowerCase()})22`,
            }}>
              {alert.risk_score}
            </div>

            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="flex items-center gap-2 mb-3" style={{ marginBottom: 3 }}>
                <RiskBadge label={alert.risk_label} size="sm" />
              </div>
              <div className="truncate font-mono text-xs" style={{ color: 'var(--text-secondary)', marginBottom: 3 }}>
                {alert.address}
              </div>
              <div className="truncate text-xs text-muted">
                {alert.top_reason}
              </div>
            </div>

            {/* Arrow indicator */}
            <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', flexShrink: 0 }}>›</div>
          </button>
        ))}
      </div>
    </div>
  )
}
