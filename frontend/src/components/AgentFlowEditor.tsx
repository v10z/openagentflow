import { useCallback, useMemo, useState } from 'react'
import ReactFlow, {
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Node,
  Edge,
  MarkerType,
  Handle,
  Position,
  NodeProps,
} from 'reactflow'
import 'reactflow/dist/style.css'

// --- Types ---

export interface AgentNode {
  id: string
  name: string
  model?: string
  tools?: string[]
  reasoning?: string
}

export interface AgentEdge {
  source: string
  target: string
  label?: string
}

interface AgentFlowEditorProps {
  agents?: AgentNode[]
  edges?: AgentEdge[]
  onSave?: (agents: AgentNode[], edges: AgentEdge[]) => void
  readOnly?: boolean
}

// --- Custom agent node ---

function AgentFlowNode({ data }: NodeProps) {
  return (
    <div className="bg-sidebar-bg border-2 border-accent-blue rounded-lg px-4 py-3 min-w-[200px] max-w-[260px] shadow-md">
      <Handle type="target" position={Position.Top} className="!bg-accent-blue !w-2.5 !h-2.5" />

      {/* Agent name */}
      <div className="flex items-center gap-2 mb-2">
        <svg className="w-4 h-4 text-accent-blue flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" />
        </svg>
        <span className="text-sm font-medium text-editor-fg truncate">{data.name}</span>
      </div>

      {/* Model */}
      {data.model && (
        <div className="text-2xs text-activitybar-fg mb-1.5">
          <span className="text-token-keyword">model:</span>{' '}
          <span className="text-token-string font-mono">{data.model}</span>
        </div>
      )}

      {/* Tools */}
      {data.tools && data.tools.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-1.5">
          {data.tools.slice(0, 3).map((tool: string) => (
            <span key={tool} className="badge text-2xs bg-accent-purple/20 text-accent-purple">
              {tool}
            </span>
          ))}
          {data.tools.length > 3 && (
            <span className="badge text-2xs bg-sidebar-active text-activitybar-fg">
              +{data.tools.length - 3}
            </span>
          )}
        </div>
      )}

      {/* Reasoning engine */}
      {data.reasoning && (
        <div className="text-2xs text-activitybar-fg">
          <span className="text-token-keyword">reasoning:</span>{' '}
          <span className="text-accent-green">{data.reasoning}</span>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-accent-blue !w-2.5 !h-2.5" />
    </div>
  )
}

const nodeTypes = {
  agent: AgentFlowNode,
}

// --- Layout ---

function layoutAgents(agents: AgentNode[], agentEdges: AgentEdge[]) {
  if (!agents.length) return { nodes: [], edges: [] }

  const nodeWidth = 240
  const nodeHeight = 140
  const hGap = 80
  const vGap = 100

  // Build adjacency for BFS
  const incoming = new Map<string, string[]>()
  const outgoing = new Map<string, string[]>()
  agents.forEach((a) => {
    incoming.set(a.id, [])
    outgoing.set(a.id, [])
  })
  agentEdges.forEach((e) => {
    outgoing.get(e.source)?.push(e.target)
    incoming.get(e.target)?.push(e.source)
  })

  // BFS for levels
  const roots = agents.filter((a) => !incoming.get(a.id)?.length)
  if (roots.length === 0 && agents.length > 0) roots.push(agents[0])

  const levels = new Map<string, number>()
  const queue = roots.map((r) => ({ id: r.id, level: 0 }))
  while (queue.length) {
    const { id, level } = queue.shift()!
    if (levels.has(id) && levels.get(id)! >= level) continue
    levels.set(id, level)
    outgoing.get(id)?.forEach((t) => queue.push({ id: t, level: level + 1 }))
  }
  agents.forEach((a) => { if (!levels.has(a.id)) levels.set(a.id, 0) })

  // Group by level
  const byLevel = new Map<number, AgentNode[]>()
  agents.forEach((a) => {
    const lvl = levels.get(a.id) ?? 0
    if (!byLevel.has(lvl)) byLevel.set(lvl, [])
    byLevel.get(lvl)!.push(a)
  })

  const maxPerLevel = Math.max(...Array.from(byLevel.values()).map((n) => n.length), 1)
  const totalWidth = maxPerLevel * (nodeWidth + hGap)

  const nodes: Node[] = agents.map((agent) => {
    const lvl = levels.get(agent.id) ?? 0
    const row = byLevel.get(lvl) || []
    const idx = row.findIndex((n) => n.id === agent.id)
    const rowWidth = row.length * (nodeWidth + hGap)
    const startX = (totalWidth - rowWidth) / 2

    return {
      id: agent.id,
      type: 'agent',
      position: {
        x: startX + idx * (nodeWidth + hGap),
        y: lvl * (nodeHeight + vGap),
      },
      data: {
        name: agent.name,
        model: agent.model,
        tools: agent.tools,
        reasoning: agent.reasoning,
      },
    }
  })

  const edgeColor = '#007acc'
  const flowEdges: Edge[] = agentEdges.map((e, i) => ({
    id: `agent-e-${i}`,
    source: e.source,
    target: e.target,
    label: e.label || undefined,
    labelStyle: { fill: '#858585', fontSize: 10 },
    labelBgStyle: { fill: '#252526', fillOpacity: 0.9 },
    labelBgPadding: [4, 2] as [number, number],
    labelBgBorderRadius: 3,
    markerEnd: { type: MarkerType.ArrowClosed, color: edgeColor, width: 16, height: 16 },
    style: { stroke: edgeColor, strokeWidth: 2 },
  }))

  return { nodes, edges: flowEdges }
}

// --- Main component ---

export default function AgentFlowEditor({
  agents = [],
  edges: agentEdges = [],
  onSave,
  readOnly = false,
}: AgentFlowEditorProps) {
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => layoutAgents(agents, agentEdges),
    [agents, agentEdges]
  )

  const [nodes, , onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onConnect = useCallback(
    (params: Connection) => {
      if (readOnly) return
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            markerEnd: { type: MarkerType.ArrowClosed, color: '#007acc', width: 16, height: 16 },
            style: { stroke: '#007acc', strokeWidth: 2 },
          },
          eds
        )
      )
    },
    [readOnly, setEdges]
  )

  if (!agents.length) {
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
              d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z"
            />
          </svg>
          <h3 className="empty-state-title">No Agent Flow</h3>
          <p className="empty-state-description">
            Register agents and connect them to build a multi-agent workflow. Each agent node shows
            its model, tools, and reasoning engine.
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
        onNodesChange={readOnly ? undefined : onNodesChange}
        onEdgesChange={readOnly ? undefined : onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        minZoom={0.2}
        maxZoom={2}
        className="bg-editor-bg"
        proOptions={{ hideAttribution: true }}
        nodesDraggable={!readOnly}
        nodesConnectable={!readOnly}
      >
        <Controls />
        <Background color="#3c3c3c" gap={20} size={1} />
      </ReactFlow>
    </div>
  )
}
