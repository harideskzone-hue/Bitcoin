// frontend/src/components/GraphExplorer.tsx
// HP5 — Force-directed graph visualisation using Cytoscape.js
// Node click expands 1-hop neighbourhood (HP5 acceptance test)

import cytoscape from 'cytoscape'
import { useEffect, useRef, useState } from 'react'
import type { GraphEdge, GraphNode } from '../api'
import { api } from '../api'

// ── Colour helpers ─────────────────────────────────────────────────────────

const RISK_COLOUR: Record<string, string> = {
  LOW:      '#22c55e',
  MEDIUM:   '#eab308',
  HIGH:     '#f97316',
  CRITICAL: '#ef4444',
}
const NODE_TYPE_COLOUR: Record<string, string> = {
  address:     '#6366f1',   // indigo  — address nodes
  transaction: '#22d3ee',   // cyan    — transaction nodes
}

function nodeColour(n: GraphNode): string {
  if (n.risk_label && RISK_COLOUR[n.risk_label]) return RISK_COLOUR[n.risk_label]
  return NODE_TYPE_COLOUR[n.node_type] ?? '#8892c8'
}

// ── Cytoscape style ────────────────────────────────────────────────────────

const CY_STYLE = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(colour)',
      'border-width': 2,
      'border-color': 'data(colour)',
      'border-opacity': 0.6,
      'label': 'data(short)',
      'font-size': '9px',
      'color': '#c8d0f0',
      'text-valign': 'bottom',
      'text-margin-y': 4,
      'width': 28,
      'height': 28,
    },
  },
  {
    selector: 'node[type = "transaction"]',
    style: {
      'shape': 'diamond',
      'width': 18,
      'height': 18,
    },
  },
  {
    selector: 'node:selected',
    style: {
      'border-width': 3,
      'border-color': '#fff',
      'border-opacity': 1,
    },
  },
  {
    selector: 'edge',
    style: {
      'line-color': '#2a3560',
      'target-arrow-color': '#2a3560',
      'target-arrow-shape': 'triangle',
      'arrow-scale': 0.8,
      'curve-style': 'bezier',
      'width': 1.5,
    },
  },
  {
    selector: 'edge[edgeType = "OUTPUT_TO"]',
    style: {
      'line-color': '#3b4a80',
      'target-arrow-color': '#3b4a80',
    },
  },
]

// ── Component ──────────────────────────────────────────────────────────────

interface Props {
  centreAddress: string | null
}

export function GraphExplorer({ centreAddress }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef        = useRef<cytoscape.Core | null>(null)

  const [nodes, setNodes]     = useState<GraphNode[]>([])
  const [edges, setEdges]     = useState<GraphEdge[]>([])
  const [disclaimer, setDisc] = useState('')
  const [loading, setLoad]    = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [selected, setSelected] = useState<GraphNode | null>(null)

  // ── Load graph data ──────────────────────────────────────────────────────
  useEffect(() => {
    if (!centreAddress) { setNodes([]); setEdges([]); return }
    setLoad(true); setError(null); setSelected(null)
    api.graph(centreAddress, 1)
      .then(g => { setNodes(g.nodes); setEdges(g.edges); setDisc(g.disclaimer) })
      .catch(e => setError(e.message))
      .finally(() => setLoad(false))
  }, [centreAddress])

  // ── Mount / update Cytoscape ─────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || nodes.length === 0) return

    const elements = [
      ...nodes.map(n => ({
        data: {
          id:     n.id,
          label:  n.id,
          short:  n.id.slice(0, 8) + '…',
          colour: nodeColour(n),
          type:   n.node_type,
        },
      })),
      ...edges.map((e, i) => ({
        data: {
          id:       `e${i}`,
          source:   e.source,
          target:   e.target,
          edgeType: e.edge_type,
        },
      })),
    ]

    // Destroy previous instance
    if (cyRef.current) { cyRef.current.destroy() }

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      style: CY_STYLE as unknown as cytoscape.StylesheetCSS[],
      layout: { name: 'cose', idealEdgeLength: 80, nodeOverlap: 20, animate: false } as cytoscape.LayoutOptions,
      userZoomingEnabled:   true,
      userPanningEnabled:   true,
      boxSelectionEnabled:  false,
      autoungrabify:        false,
    })

    // Node click → expand 1 hop
    cy.on('tap', 'node', evt => {
      const nodeId = evt.target.id() as string
      const nodeData = nodes.find(n => n.id === nodeId) ?? null
      setSelected(nodeData)

      // Highlight: dim all, highlight tapped + neighbours
      cy.elements().addClass('dimmed')
      evt.target.removeClass('dimmed')
      evt.target.neighborhood().removeClass('dimmed')
    })

    cy.on('tap', evt => {
      // Click on background — clear selection
      if (evt.target === cy) { cy.elements().removeClass('dimmed'); setSelected(null) }
    })

    cyRef.current = cy
    return () => { cy.destroy(); cyRef.current = null }
  }, [nodes, edges])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>

      {/* Header */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <div className="flex items-center justify-between">
          <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>Graph Explorer</span>
          <div style={{ display: 'flex', gap: 12, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#6366f1', display: 'inline-block' }} /> Address
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 10, height: 10, background: '#22d3ee', transform: 'rotate(45deg)', display: 'inline-block' }} /> Tx
            </span>
          </div>
        </div>
        {disclaimer && (
          <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 4, fontStyle: 'italic' }}>
            {disclaimer}
          </div>
        )}
      </div>

      {/* Graph canvas */}
      <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
        {loading && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Loading graph…
          </div>
        )}
        {error && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--risk-high)', fontSize: '0.85rem' }}>
            {error}
          </div>
        )}
        {!centreAddress && !loading && (
          <div className="empty-state" style={{ height: '100%' }}>
            <div className="icon" style={{ fontSize: '2.5rem', opacity: 0.3 }}>⬡</div>
            <div>Select an address to explore its graph</div>
          </div>
        )}
        <div
          ref={containerRef}
          style={{ width: '100%', height: '100%', background: 'var(--bg-panel)' }}
        />
      </div>

      {/* Node info strip */}
      {selected && (
        <div style={{
          padding: '10px 16px', borderTop: '1px solid var(--border)',
          background: 'var(--bg-card)', flexShrink: 0,
          display: 'grid', gridTemplateColumns: 'auto 1fr auto', gap: 12, alignItems: 'center',
        }}>
          <div style={{
            width: 10, height: 10, borderRadius: '50%',
            background: nodeColour(selected),
            boxShadow: `0 0 8px ${nodeColour(selected)}`,
          }} />
          <div className="font-mono text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
            {selected.id}
          </div>
          <div style={{
            fontSize: '0.72rem', fontWeight: 600, padding: '2px 8px', borderRadius: 999,
            background: selected.risk_label ? `var(--risk-${selected.risk_label.toLowerCase()}-bg)` : 'var(--border)',
            color: selected.risk_label ? `var(--risk-${selected.risk_label.toLowerCase()})` : 'var(--text-muted)',
          }}>
            {selected.risk_label ?? selected.node_type}
          </div>
        </div>
      )}
    </div>
  )
}
