// frontend/src/api.ts
// Typed API client — wraps the five endpoints defined in backend/openapi.yaml

const BASE = '/api/v1'

export type RiskLabel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export interface WalletAlert {
  address: string
  risk_score: number
  risk_label: RiskLabel
  top_reason: string
  reasons: string[]
  cluster_id: string | null
  cluster_confidence: string | null
}

export interface AlertListResponse {
  alerts: WalletAlert[]
  total: number
  score_disclaimer: string
}

export interface P2PSignal {
  signal_source: 'SIMULATED'
  ip_cluster_id: string
  timestamp: string
  tx_hash: string
  disclaimer: string
}

export interface TemporalProfile {
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
  hours_distribution: number[]
}

export interface WalletDetailResponse {
  address: string
  risk_score: number
  risk_label: RiskLabel
  reasons: string[]
  score_type: string
  score_disclaimer: string
  hops_to_nearest_flagged: number | null
  tx_burst_score: number
  timing_anomaly_score: number
  cluster_id: string | null
  cluster_confidence: string | null
  candidate_change_address: boolean
  p2p_signals: P2PSignal[] | null
  temporal_profile: TemporalProfile | null
}

export interface GraphNode {
  id: string
  risk_score: number | null
  risk_label: RiskLabel | null
  node_type: 'address' | 'transaction'
}

export interface GraphEdge {
  source: string
  target: string
  edge_type: 'INPUT_TO' | 'OUTPUT_TO'
  value_btc: number | null
  timestamp: string | null
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
  disclaimer: string
}

export interface HealthResponse {
  status: string
  model_version: string
  last_updated: string
  offline_mode: boolean
}

// ── Fetch helpers ──────────────────────────────────────────────────────────

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail?.detail ?? `HTTP ${res.status}`)
  }
  return res.json()
}

// ── API functions ──────────────────────────────────────────────────────────

export const api = {
  health: () => get<HealthResponse>('/health'),

  alerts: (params?: { min_risk?: number; limit?: number }) =>
    get<AlertListResponse>('/alerts', params as Record<string, number>),

  wallet: (addressHash: string) =>
    get<WalletDetailResponse>(`/wallets/${encodeURIComponent(addressHash)}`),

  graph: (address: string, depth: 1 | 2 = 1) =>
    get<GraphResponse>('/graph', { address, depth }),
}
