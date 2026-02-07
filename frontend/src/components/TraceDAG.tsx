import { useCallback, useMemo } from 'react'
import ReactFlow, {
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  MarkerType,
  Handle,
  Position,
  NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'

// --- Types ---

interface TraceVertex {
  id: string
  type: string // step_type: "thesis", "antithesis", "synthesis", etc.
  content: string
  score: number
  metadata: Record<string, any>
}

interface TraceEdge {
  source: string
  target: string
  label: string
}

interface TraceDAGProps {
  vertices: TraceVertex[]
  edges: TraceEdge[]
  onNodeClick?: (vertex: TraceVertex) => void
}

// --- Step type styling ---

type StepCategory = 'thesis' | 'antithesis' | 'synthesis' | 'evaluation' | 'default'

function categorizeStepType(stepType: string): StepCategory {
  const lower = stepType.toLowerCase()
  if (['thesis', 'initial', 'dream'].includes(lower)) return 'thesis'
  if (['antithesis', 'critique', 'challenge'].includes(lower)) return 'antithesis'
  if (['synthesis', 'final', 'refinement'].includes(lower)) return 'synthesis'
  if (['evaluation', 'judge'].includes(lower)) return 'evaluation'
  return 'default'
}

const categoryStyles: Record<StepCategory, { border: string; badge: string; badgeText: string }> = {
  thesis: {
    border: 'border-accent-blue',
    badge: 'bg-accent-blue',
    badgeText: 'text-white',
  },
  antithesis: {
    border: 'border-accent-red',
    badge: 'bg-accent-red',
    badgeText: 'text-white',
  },
  synthesis: {
    border: 'border-accent-green',
    badge: 'bg-accent-green',
    badgeText: 'text-editor-bg',
  },
  evaluation: {
    border: 'border-accent-yellow',
    badge: 'bg-accent-yellow',
    badgeText: 'text-editor-bg',
  },
  default: {
    border: 'border-sidebar-border',
    badge: 'bg-sidebar-active',
    badgeText: 'text-sidebar-fg',
  },
}

// --- Score bar ---

function ScoreBar({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(1, score))
  const percentage = `${clamped * 100}%`

  let barColor = 'bg-activitybar-fg'
  if (clamped >= 0.8) barColor = 'bg-accent-green'
  else if (clamped >= 0.5) barColor = 'bg-accent-blue'
  else if (clamped >= 0.3) barColor = 'bg-accent-yellow'
  else barColor = 'bg-accent-red'

  return (
    <div className="flex items-center gap-1.5 mt-2">
      <div className="flex-1 h-1 bg-sidebar-active rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${barColor}`} style={{ width: percentage }} />
      </div>
      <span className="text-2xs text-activitybar-fg font-mono w-7 text-right">
        {clamped.toFixed(2)}
      </span>
    </div>
  )
}

// --- Custom node component ---

function StepNode({ data }: NodeProps) {
  const category = categorizeStepType(data.stepType)
  const styles = categoryStyles[category]

  return (
    <div
      className={`bg-sidebar-bg border-2 ${styles.border} rounded-lg px-3 py-2 min-w-[180px] max-w-[240px] shadow-md cursor-pointer hover:brightness-110 transition-all`}
    >
      <Handle type="target" position={Position.Top} className="!bg-activitybar-fg !w-2 !h-2" />

      {/* Step type badge */}
      <div className="mb-1.5">
        <span
          className={`inline-block text-2xs px-1.5 py-0.5 rounded ${styles.badge} ${styles.badgeText} font-medium uppercase tracking-wide`}
        >
          {data.stepType}
        </span>
      </div>

      {/* Content preview */}
      <p className="text-xs text-editor-fg leading-relaxed line-clamp-3">
        {data.content.length > 100 ? data.content.slice(0, 100) + '...' : data.content}
      </p>

      {/* Score bar */}
      <ScoreBar score={data.score} />

      <Handle type="source" position={Position.Bottom} className="!bg-activitybar-fg !w-2 !h-2" />
    </div>
  )
}

const nodeTypes = {
  step: StepNode,
}

// --- BFS layout algorithm (adapted from TwinGraph's DAGVisualization) ---

function calculateDAGLayout(vertices: TraceVertex[], dagEdges: TraceEdge[]) {
  if (!vertices.length) return { nodes: [], edges: [] }

  // Build adjacency lists
  const outgoing = new Map<string, string[]>()
  const incoming = new Map<string, string[]>()

  vertices.forEach((v) => {
    outgoing.set(v.id, [])
    incoming.set(v.id, [])
  })

  dagEdges.forEach((edge) => {
    outgoing.get(edge.source)?.push(edge.target)
    incoming.get(edge.target)?.push(edge.source)
  })

  // Find root nodes (no incoming edges)
  const roots = vertices.filter((v) => !incoming.get(v.id)?.length)

  // If no roots found (cycle), pick the first vertex as root
  if (roots.length === 0 && vertices.length > 0) {
    roots.push(vertices[0])
  }

  // Calculate levels using BFS
  const levels = new Map<string, number>()
  const queue = [...roots.map((r) => ({ id: r.id, level: 0 }))]

  while (queue.length) {
    const { id, level } = queue.shift()!

    // Only update if this path gives a deeper (or first) level
    if (levels.has(id) && levels.get(id)! >= level) continue
    levels.set(id, level)

    outgoing.get(id)?.forEach((targetId) => {
      queue.push({ id: targetId, level: level + 1 })
    })
  }

  // Ensure all vertices have a level (handle disconnected nodes)
  vertices.forEach((v) => {
    if (!levels.has(v.id)) {
      levels.set(v.id, 0)
    }
  })

  // Group nodes by level
  const nodesByLevel = new Map<number, TraceVertex[]>()
  vertices.forEach((v) => {
    const level = levels.get(v.id) ?? 0
    if (!nodesByLevel.has(level)) nodesByLevel.set(level, [])
    nodesByLevel.get(level)!.push(v)
  })

  // Calculate positions with centering within each level
  const nodeWidth = 220
  const nodeHeight = 120
  const horizontalSpacing = 60
  const verticalSpacing = 80

  const maxNodesInLevel = Math.max(
    ...Array.from(nodesByLevel.values()).map((nodes) => nodes.length),
    1
  )
  const totalWidth = maxNodesInLevel * (nodeWidth + horizontalSpacing)

  const nodes: Node[] = vertices.map((vertex) => {
    const level = levels.get(vertex.id) ?? 0
    const nodesInLevel = nodesByLevel.get(level) || []
    const indexInLevel = nodesInLevel.findIndex((n) => n.id === vertex.id)

    // Center nodes within their level
    const levelWidth = nodesInLevel.length * (nodeWidth + horizontalSpacing)
    const startX = (totalWidth - levelWidth) / 2

    return {
      id: vertex.id,
      type: 'step',
      position: {
        x: startX + indexInLevel * (nodeWidth + horizontalSpacing),
        y: level * (nodeHeight + verticalSpacing),
      },
      data: {
        stepType: vertex.type,
        content: vertex.content,
        score: vertex.score,
        metadata: vertex.metadata,
        // Store original vertex for click handler
        _vertex: vertex,
      },
    }
  })

  // Build ReactFlow edges with labels and arrows
  const edgeColor = '#6b7280'
  const flowEdges: Edge[] = dagEdges.map((edge, index) => ({
    id: `trace-e-${index}`,
    source: edge.source,
    target: edge.target,
    label: edge.label || undefined,
    labelStyle: {
      fill: '#858585',
      fontSize: 10,
      fontFamily: 'Segoe UI, system-ui, sans-serif',
    },
    labelBgStyle: {
      fill: '#252526',
      fillOpacity: 0.9,
    },
    labelBgPadding: [4, 2] as [number, number],
    labelBgBorderRadius: 3,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: edgeColor,
      width: 16,
      height: 16,
    },
    style: {
      stroke: edgeColor,
      strokeWidth: 1.5,
    },
    animated: false,
  }))

  return { nodes, edges: flowEdges }
}

// --- Main component ---

export default function TraceDAG({ vertices, edges: dagEdges, onNodeClick }: TraceDAGProps) {
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => calculateDAGLayout(vertices, dagEdges),
    [vertices, dagEdges]
  )

  const [nodes, , onNodesChange] = useNodesState(initialNodes)
  const [edges, , onEdgesChange] = useEdgesState(initialEdges)

  const handleNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (onNodeClick) {
        const vertex = vertices.find((v) => v.id === node.id)
        if (vertex) {
          onNodeClick(vertex)
        }
      }
    },
    [vertices, onNodeClick]
  )

  if (!vertices.length) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="empty-state">
          <svg
            className="empty-state-icon"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5"
            />
          </svg>
          <h3 className="empty-state-title">No Reasoning Steps</h3>
          <p className="empty-state-description">
            Run a reasoning engine to generate a trace with steps that will be visualized here as a
            directed acyclic graph.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.2}
        maxZoom={2}
        className="bg-editor-bg"
        proOptions={{ hideAttribution: true }}
      >
        <Controls />
        <Background color="#3c3c3c" gap={20} size={1} />
      </ReactFlow>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 panel rounded px-3 py-2">
        <div className="text-2xs text-activitybar-fg uppercase tracking-wide mb-1.5 font-medium">
          Step Types
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {[
            { label: 'Thesis / Initial', color: 'bg-accent-blue' },
            { label: 'Antithesis / Critique', color: 'bg-accent-red' },
            { label: 'Synthesis / Final', color: 'bg-accent-green' },
            { label: 'Evaluation / Judge', color: 'bg-accent-yellow' },
          ].map(({ label, color }) => (
            <div key={label} className="flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-sm ${color}`} />
              <span className="text-2xs text-sidebar-fg">{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
