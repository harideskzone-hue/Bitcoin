// frontend/src/App.tsx
// SIH26146 — Bitcoin Risk Surveillance Dashboard
// HP5 main app: AlertList + GraphExplorer + WalletDetail + SearchBar

import { useEffect, useState } from 'react'
import { api } from './api'
import { AlertList }     from './components/AlertList'
import { GraphExplorer } from './components/GraphExplorer'
import { SearchBar }     from './components/SearchBar'
import { WalletDetail }  from './components/WalletDetail'
import './index.css'

type ApiStatus = 'checking' | 'ok' | 'error'

export default function App() {
  const [selectedAddress, setSelectedAddress] = useState<string | null>(null)
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')

  // G12 fix: real health check instead of always-green dot
  useEffect(() => {
    api.health()
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'))
    // Re-check every 30 s
    const id = setInterval(() => {
      api.health()
        .then(() => setApiStatus('ok'))
        .catch(() => setApiStatus('error'))
    }, 30_000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="app-layout">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="app-header" style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 20px',
        background: 'var(--bg-panel)',
        borderBottom: '1px solid var(--border)',
        boxShadow: '0 1px 12px rgba(0,0,0,0.4)',
        zIndex: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: 'linear-gradient(135deg, var(--accent-indigo), var(--accent-purple))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '1rem',
          }}>₿</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: '0.95rem', lineHeight: 1.2 }}>
              Bitcoin Risk Surveillance
            </div>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              SIH26146 · Weak-Supervision GNN · Pipeline B
            </div>
          </div>
        </div>

        <div style={{ flex: 1, maxWidth: 380, margin: '0 32px' }}>
          <SearchBar onSearch={setSelectedAddress} />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <StatusDot status={apiStatus} />
          <div style={{
            fontSize: '0.70rem', color: 'var(--text-muted)',
            textAlign: 'right', lineHeight: 1.5,
          }}>
            <div>Phase 0 · Demo-ready</div>
            <div>50k validation pending</div>
          </div>
        </div>
      </header>

      {/* ── Sidebar: Alert feed ──────────────────────────────────────────── */}
      <aside className="app-sidebar">
        <AlertList onSelect={setSelectedAddress} selected={selectedAddress} />
      </aside>

      {/* ── Main: Graph explorer ─────────────────────────────────────────── */}
      <main className="app-main">
        <GraphExplorer centreAddress={selectedAddress} />
      </main>

      {/* ── Detail: Wallet deep-dive ─────────────────────────────────────── */}
      <aside className="app-detail" style={{ background: 'var(--bg-panel)' }}>
        <WalletDetail address={selectedAddress} />
      </aside>

    </div>
  )
}

// ── Status dot ─────────────────────────────────────────────────────────────
// G12: reflects actual /api/v1/health response, not always-green

function StatusDot({ status }: { status: ApiStatus }) {
  const config = {
    checking: { colour: 'var(--text-muted)',    label: 'Connecting…', pulse: false },
    ok:       { colour: 'var(--risk-low)',       label: 'API ready',   pulse: true  },
    error:    { colour: 'var(--risk-critical)',  label: 'API offline', pulse: false },
  }[status]

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: config.colour,
        boxShadow: `0 0 6px ${config.colour}`,
        display: 'inline-block',
        animation: config.pulse ? 'pulse 2s infinite' : 'none',
      }} />
      <style>{`@keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: 0.4 } }`}</style>
      {config.label}
    </div>
  )
}
