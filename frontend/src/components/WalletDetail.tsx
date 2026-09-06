// frontend/src/components/WalletDetail.tsx
// HP5 — Full risk profile for a single address

import { useEffect, useState } from 'react'
import type { WalletDetailResponse } from '../api'
import { api } from '../api'
import { RiskBadge, ScoreRing } from './RiskBadge'
import { P2PPanel } from './P2PPanel'

interface Props {
  address: string | null
}

export function WalletDetail({ address }: Props) {
  const [data, setData]     = useState<WalletDetailResponse | null>(null)
  const [loading, setLoad]  = useState(false)
  const [error, setError]   = useState<string | null>(null)

  useEffect(() => {
    if (!address) { setData(null); return }
    setLoad(true); setError(null)
    api.wallet(address)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoad(false))
  }, [address])

  if (!address) return (
    <div className="empty-state" style={{ height: '100%' }}>
      <div className="icon">🔎</div>
      <div>Select an alert or search for an address</div>
    </div>
  )

  if (loading) return (
    <div style={{ padding: 20 }}>
      {[...Array(6)].map((_, i) => (
        <div key={i} className="loading-shimmer mb-3" style={{ height: i === 0 ? 80 : 40, borderRadius: 8 }} />
      ))}
    </div>
  )

  if (error) return (
    <div className="empty-state" style={{ height: '100%' }}>
      <div className="icon">⚠️</div>
      <div style={{ fontSize: '0.85rem' }}>{error}</div>
    </div>
  )

  if (!data) return null

  return (
    <div style={{ padding: '16px' }}>

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-4" style={{ marginBottom: 16 }}>
        <ScoreRing score={data.risk_score} label={data.risk_label} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="flex items-center gap-2" style={{ marginBottom: 6 }}>
            <RiskBadge label={data.risk_label} score={data.risk_score} />
          </div>
          <div className="font-mono text-xs truncate" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            {data.address}
          </div>
        </div>
      </div>

      {/* ── Disclaimer ───────────────────────────────────────────────────── */}
      <div style={{
        padding: '8px 12px', marginBottom: 16,
        background: 'rgba(99,102,241,0.07)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic', lineHeight: 1.5,
      }}>
        {data.score_disclaimer}
      </div>

      {/* ── Reasons ──────────────────────────────────────────────────────── */}
      <div className="section-heading">Why flagged</div>
      {data.reasons.length === 0 ? (
        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 16 }}>
          No reason tokens computed yet (run explainer pipeline)
        </div>
      ) : (
        <div style={{ marginBottom: 16 }}>
          {data.reasons.map((r, i) => (
            <div key={i} className="reason-bullet">
              <span className="dot" />
              <span>{r}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Feature signals ──────────────────────────────────────────────── */}
      <div className="section-heading">Feature signals</div>
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: '8px 16px', marginBottom: 16,
      }}>
        <FeatureRow label="Hops to flagged" value={
          data.hops_to_nearest_flagged !== null
            ? String(data.hops_to_nearest_flagged)
            : 'N/A (no seed coverage)'
        } />
        <FeatureRow label="Burst score"     value={data.tx_burst_score.toFixed(2)} />
        <FeatureRow label="Timing anomaly"  value={data.timing_anomaly_score.toFixed(2)} />
        <FeatureRow label="Cluster"         value={data.cluster_id ?? '—'} mono />
        <FeatureRow label="Cluster conf."   value={data.cluster_confidence ?? '—'} />
        <FeatureRow label="Change addr?"    value={data.candidate_change_address ? '✓ Yes' : 'No'} />
      </div>

      {/* ── P2P Signals ──────────────────────────────────────────────────── */}
      <P2PPanel signals={data.p2p_signals} />
    </div>
  )
}

function FeatureRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 2 }}>
        {label}
      </div>
      <div style={{
        fontSize: '0.82rem', color: 'var(--text-secondary)',
        fontFamily: mono ? 'JetBrains Mono, monospace' : undefined,
        wordBreak: 'break-all',
      }}>
        {value}
      </div>
    </div>
  )
}
