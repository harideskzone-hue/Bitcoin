import { useEffect, useRef, useState } from 'react'
import type { WalletDetailResponse } from '../api'
import { api } from '../api'
import { RiskBadge, ScoreRing } from './RiskBadge'
import { P2PPanel } from './P2PPanel'
import { TemporalChart } from './TemporalChart'

interface Props {
  address: string | null
}

export function WalletDetail({ address }: Props) {
  const [data, setData]         = useState<WalletDetailResponse | null>(null)
  const [loading, setLoad]      = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [exporting, setExport]  = useState(false)
  const [exportDone, setExDone] = useState(false)
  const exportTimer              = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!address) { setData(null); return }
    setLoad(true); setError(null); setExDone(false)
    api.wallet(address)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoad(false))
  }, [address])

  // ── Export handler (Track 6) ─────────────────────────────────────────────
  const handleExport = async () => {
    if (!address || exporting) return
    setExport(true)
    try {
      const res = await fetch(`/api/v1/wallets/${address}/report`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `investigation_${address.slice(0, 12)}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setExDone(true)
      if (exportTimer.current) clearTimeout(exportTimer.current)
      exportTimer.current = setTimeout(() => setExDone(false), 3000)
    } catch {
      // silently ignore
    } finally {
      setExport(false)
    }
  }

  if (!address) return (
    <div className="empty-state" style={{ height: '100%' }}>
      <div className="icon">🔎</div>
      <div>Select an alert or search for an address</div>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 6 }}>
        Try: <code style={{ fontSize: '0.7rem' }}>17ebdd724dbfe21e</code>
      </div>
    </div>
  )

  if (loading) return (
    <div style={{ padding: 20 }}>
      {[...Array(7)].map((_, i) => (
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
    <div style={{ padding: '16px', overflowY: 'auto', height: '100%' }}>

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-4" style={{ marginBottom: 12 }}>
        <ScoreRing score={data.risk_score} label={data.risk_label} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="flex items-center gap-2" style={{ marginBottom: 4 }}>
            <RiskBadge label={data.risk_label} score={data.risk_score} />
            {/* Score type badge */}
            {(data.score_type) && (
              <span style={{
                background: 'rgba(99,102,241,0.12)', color: 'var(--accent-indigo)',
                border: '1px solid rgba(99,102,241,0.3)',
                borderRadius: 6, fontSize: '0.62rem', fontWeight: 600, padding: '1px 6px',
              }}>
                {(data.score_type) === 'heuristic_fallback' ? 'HEURISTIC' : 'GCN'}
              </span>
            )}
          </div>
          <div className="font-mono text-xs truncate" style={{ color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            {data.address}
          </div>
        </div>

        {/* ── Export button (Track 6) ─────────────────────────────────────── */}
        <button
          onClick={handleExport}
          disabled={exporting}
          title="Export investigation report"
          style={{
            flexShrink: 0,
            padding: '6px 12px',
            background: exportDone ? 'rgba(34,197,94,0.15)' : 'var(--bg-card)',
            color: exportDone ? '#22c55e' : 'var(--text-secondary)',
            border: `1px solid ${exportDone ? '#22c55e' : 'var(--border)'}`,
            borderRadius: 8,
            fontSize: '0.72rem',
            fontWeight: 600,
            cursor: exporting ? 'wait' : 'pointer',
            display: 'flex', alignItems: 'center', gap: 5,
            transition: 'all 0.2s',
          }}
        >
          {exporting ? '⏳' : exportDone ? '✓' : '⬇'}&nbsp;
          {exportDone ? 'Saved' : 'Report'}
        </button>
      </div>

      {/* ── Disclaimer ──────────────────────────────────────────────────────── */}
      <div style={{
        padding: '8px 12px', marginBottom: 14,
        background: 'rgba(99,102,241,0.07)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic', lineHeight: 1.5,
      }}>
        {data.score_disclaimer}
      </div>

      {/* ── Reasons ─────────────────────────────────────────────────────────── */}
      <div className="section-heading">Why flagged</div>
      {data.reasons.length === 0 ? (
        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: 14 }}>
          No reason tokens computed
        </div>
      ) : (
        <div style={{ marginBottom: 14 }}>
          {data.reasons.map((r, i) => (
            <div key={i} className="reason-bullet">
              <span className="dot" />
              <span>{r}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Temporal Risk Profile (Track 4) ─────────────────────────────────────── */}
      {data.temporal_profile && (
        <TemporalChart profile={data.temporal_profile} />
      )}

      {/* ── Feature signals ─────────────────────────────────────────────────── */}
      <div className="section-heading">Feature signals</div>
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: '8px 16px', marginBottom: 14,
      }}>
        <FeatureRow label="Hops to flagged" value={
          data.hops_to_nearest_flagged !== null && data.hops_to_nearest_flagged !== undefined
            ? String(data.hops_to_nearest_flagged)
            : '— (50k snapshot pending)'
        } />
        <FeatureRow label="Burst score"    value={data.tx_burst_score.toFixed(2)} />
        <FeatureRow label="Timing anomaly" value={data.timing_anomaly_score.toFixed(2)} />
        <FeatureRow label="Cluster"        value={data.cluster_id ?? '—'} mono />
        <FeatureRow label="Cluster conf."  value={data.cluster_confidence ?? '—'} />
        <FeatureRow label="Change addr?"   value={data.candidate_change_address ? '✓ Candidate' : 'No'} />
      </div>

      {/* ── P2P Signals ─────────────────────────────────────────────────────── */}
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
