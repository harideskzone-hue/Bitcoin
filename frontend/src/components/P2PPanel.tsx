// frontend/src/components/P2PPanel.tsx
// HP5 — P2P signal panel
// ⚠ SIMULATED DATA banner MUST be visible without scrolling (frozen spec P0.8.5)

import type { P2PSignal } from '../api'

interface Props {
  signals: P2PSignal[] | null
}

export function P2PPanel({ signals }: Props) {
  return (
    <div style={{ marginTop: 20 }}>
      <div className="section-heading">P2P Network Signals</div>

      {/* P0.8.5 — SIMULATED DATA banner: always first, always visible */}
      <div className="simulated-banner mb-4" style={{ marginBottom: 12 }}>
        <span className="icon">⚠</span>
        <div>
          <div style={{ fontSize: '0.80rem', lineHeight: 1.2 }}>
            NETWORK SIGNAL — SIMULATED DATA
          </div>
          <div style={{ fontSize: '0.72rem', fontWeight: 400, opacity: 0.85, marginTop: 2 }}>
            Demonstration only. Not real evidence. Not real network telemetry.
          </div>
        </div>
      </div>

      {signals === null && (
        <div className="empty-state" style={{ padding: '20px 0' }}>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            No P2P signals for this address
          </div>
        </div>
      )}

      {signals && signals.length === 0 && (
        <div className="empty-state" style={{ padding: '20px 0' }}>
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            No simulated signals for this address
          </div>
        </div>
      )}

      {signals && signals.map((sig, i) => (
        <SignalCard key={i} signal={sig} />
      ))}
    </div>
  )
}

function SignalCard({ signal }: { signal: P2PSignal }) {
  const ts = new Date(signal.timestamp).toLocaleString('en-GB', {
    dateStyle: 'medium', timeStyle: 'short',
  })

  return (
    <div className="p2p-signal-card mb-3" style={{ marginBottom: 8 }}>
      {/* Row 1 */}
      <div>
        <div className="label">IP Cluster</div>
        <div className="value">{signal.ip_cluster_id}</div>
      </div>
      <div>
        <div className="label">Timestamp</div>
        <div className="value">{ts}</div>
      </div>
      {/* Row 2 */}
      <div style={{ gridColumn: '1 / -1' }}>
        <div className="label">Transaction</div>
        <div className="value" style={{ wordBreak: 'break-all', fontSize: '0.74rem' }}>
          {signal.tx_hash}
        </div>
      </div>
      {/* Disclaimer */}
      <div style={{ gridColumn: '1 / -1', marginTop: 4 }}>
        <div className="label" style={{ color: 'var(--simulated-text)', opacity: 0.7 }}>
          {signal.disclaimer}
        </div>
      </div>
    </div>
  )
}
