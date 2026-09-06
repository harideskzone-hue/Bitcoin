// frontend/src/components/AlertList.tsx
// HP5 — AlertList with risk badges and live API data

import { useEffect, useState } from 'react'
import type { WalletAlert } from '../api'
import { api } from '../api'
import { RiskBadge } from './RiskBadge'

interface Props {
  onSelect: (address: string) => void
  selected: string | null
}

export function AlertList({ onSelect, selected }: Props) {
  const [alerts, setAlerts]         = useState<WalletAlert[]>([])
  const [disclaimer, setDisclaimer] = useState('')
  const [loading, setLoading]       = useState(true)
  const [error, setError]           = useState<string | null>(null)
  const [minRisk, setMinRisk]       = useState(50)

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.alerts({ min_risk: minRisk, limit: 50 })
      .then(r => { setAlerts(r.alerts); setDisclaimer(r.score_disclaimer) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [minRisk])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header */}
      <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)' }}>
        <div className="flex items-center justify-between mb-3">
          <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>Alert Feed</span>
          <span style={{
            background: 'var(--risk-critical-bg)', color: 'var(--risk-critical)',
            border: '1px solid var(--risk-critical)', borderRadius: 999,
            fontSize: '0.70rem', fontWeight: 700, padding: '2px 9px',
          }}>
            {alerts.length} alerts
          </span>
        </div>

        {/* Min-risk slider */}
        <div className="flex items-center gap-2" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
          <span>Min risk</span>
          <input
            type="range" min={0} max={100} value={minRisk}
            onChange={e => setMinRisk(Number(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--accent-indigo)', cursor: 'pointer' }}
          />
          <span style={{ fontFamily: 'JetBrains Mono', minWidth: 28 }}>{minRisk}</span>
        </div>
      </div>

      {/* Disclaimer strip */}
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

      {/* Content */}
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
              Ensure the API or Prism mock is running on port 8000 / 4010
            </div>
          </div>
        )}

        {!loading && !error && alerts.length === 0 && (
          <div className="empty-state">
            <div className="icon">🔍</div>
            <div>No alerts above risk {minRisk}</div>
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
            }}
          >
            {/* Score bubble */}
            <div style={{
              width: 40, height: 40, borderRadius: 8, flexShrink: 0,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 700, fontSize: '0.9rem',
              background: `var(--risk-${alert.risk_label.toLowerCase()}-bg)`,
              color: `var(--risk-${alert.risk_label.toLowerCase()})`,
              border: `1px solid var(--risk-${alert.risk_label.toLowerCase()})`,
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
          </button>
        ))}
      </div>
    </div>
  )
}
