// frontend/src/components/TemporalChart.tsx
// Track 4 — Hour-of-day transaction activity chart
// Pure CSS/SVG — no extra dependency required

import { RISK_COLOURS } from '../constants'

interface TemporalProfile {
  avg_time_between_tx_hrs: number
  tx_burst_score: number
  weekday_vs_weekend_ratio: number
  avg_tx_value_btc: number
  total_sent_btc: number
  tx_count: number
  is_offhours_active: boolean
  velocity_label: string
  velocity_risk: string
  peak_hour_utc: number
  hours_distribution: number[]  // 24 floats
}

interface Props {
  profile: TemporalProfile
}

function VelocityBadge({ risk, label }: { risk: string; label: string }) {
  const colour = risk === 'CRITICAL' ? RISK_COLOURS.CRITICAL
    : risk === 'HIGH' ? RISK_COLOURS.HIGH
    : risk === 'MEDIUM' ? RISK_COLOURS.MEDIUM
    : RISK_COLOURS.LOW
  return (
    <span style={{
      background: colour + '22', color: colour,
      border: `1px solid ${colour}`,
      borderRadius: 6, padding: '2px 8px',
      fontSize: '0.68rem', fontWeight: 700,
    }}>
      {label}
    </span>
  )
}

export function TemporalChart({ profile }: Props) {
  const dist = profile.hours_distribution
  const max  = Math.max(...dist, 1)

  const barColour = (h: number) => {
    // Off-hours: 0–6, 22–23 UTC (grey daytime, red night)
    const isNight = h < 6 || h >= 22
    if (isNight && profile.is_offhours_active) return RISK_COLOURS.CRITICAL
    if (isNight) return '#6366f1'
    return '#4b5563'
  }

  return (
    <div style={{ marginBottom: 16 }}>
      <div className="section-heading" style={{ marginBottom: 10 }}>
        Temporal Risk Profile
      </div>

      {/* ── Velocity badge row ─────────────────────────────────────── */}
      <div className="flex items-center gap-2" style={{ marginBottom: 12, flexWrap: 'wrap' }}>
        <VelocityBadge risk={profile.velocity_risk} label={profile.velocity_label} />
        {profile.is_offhours_active && (
          <span style={{
            background: 'rgba(239,68,68,0.12)', color: RISK_COLOURS.CRITICAL,
            border: `1px solid ${RISK_COLOURS.CRITICAL}`,
            borderRadius: 6, padding: '2px 8px', fontSize: '0.68rem', fontWeight: 700,
          }}>
            ⚠ OFF-HOURS ACTIVE
          </span>
        )}
      </div>

      {/* ── Hour-of-day bar chart ──────────────────────────────────── */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 10,
        padding: '12px 12px 8px',
        marginBottom: 10,
      }}>
        <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginBottom: 6 }}>
          Transaction activity by hour (UTC) — peak hour: {String(profile.peak_hour_utc).padStart(2,'0')}:00
        </div>

        {/* Bars */}
        <div style={{
          display: 'flex', alignItems: 'flex-end', gap: 2, height: 48,
        }}>
          {dist.map((v, h) => (
            <div key={h} title={`${String(h).padStart(2,'0')}:00 UTC — activity: ${v.toFixed(1)}`}
              style={{
                flex: 1,
                height: `${Math.max(4, (v / max) * 48)}px`,
                background: barColour(h),
                borderRadius: '2px 2px 0 0',
                opacity: v === 0 ? 0.15 : 0.9,
                transition: 'height 0.3s ease, opacity 0.2s',
                cursor: 'default',
              }}
            />
          ))}
        </div>

        {/* Hour labels (every 4 hours) */}
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          {[0, 4, 8, 12, 16, 20].map(h => (
            <span key={h} style={{ fontSize: '0.6rem', color: 'var(--text-muted)', width: 24, textAlign: 'center' }}>
              {String(h).padStart(2,'0')}h
            </span>
          ))}
        </div>

        {/* Legend */}
        <div className="flex items-center gap-3" style={{ marginTop: 8, fontSize: '0.6rem', color: 'var(--text-muted)' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: RISK_COLOURS.CRITICAL, display: 'inline-block' }} />
            Off-hours peak
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: '#6366f1', display: 'inline-block' }} />
            Night (00–06 / 22–23 UTC)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: '#4b5563', display: 'inline-block' }} />
            Day
          </span>
        </div>
      </div>

      {/* ── Key stats row ──────────────────────────────────────────── */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr 1fr',
        gap: '6px 12px',
        fontSize: '0.72rem',
      }}>
        {[
          { label: 'Avg interval', value: profile.avg_time_between_tx_hrs < 1
              ? `${Math.round(profile.avg_time_between_tx_hrs * 60)} min`
              : `${profile.avg_time_between_tx_hrs.toFixed(1)} hrs` },
          { label: 'Burst score', value: profile.tx_burst_score.toFixed(2) },
          { label: 'Wkdy / wknd', value: profile.weekday_vs_weekend_ratio.toFixed(2) },
          { label: 'Tx count', value: String(profile.tx_count) },
          { label: 'Total sent', value: `${profile.total_sent_btc.toFixed(3)} BTC` },
          { label: 'Avg value', value: `${profile.avg_tx_value_btc.toFixed(4)} BTC` },
        ].map(({ label, value }) => (
          <div key={label}>
            <div style={{ color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', fontSize: '0.6rem', marginBottom: 2 }}>
              {label}
            </div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-secondary)', fontWeight: 600 }}>
              {value}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
