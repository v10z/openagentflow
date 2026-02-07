import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

interface TraceSummary {
  trace_id: string
  strategy: string
  steps_count: number
  total_llm_calls: number
  total_tokens: number
  duration_ms: number
  final_output_preview: string
  created_at: string
}

interface TraceStep {
  step_id: string
  step_type: string
  content: string
  score: number
  metadata: Record<string, any>
  parent_step_id: string | null
}

interface TraceDetail {
  trace_id: string
  strategy: string
  steps: TraceStep[]
  total_llm_calls: number
  total_tokens: number
  duration_ms: number
  final_output: string
  created_at: string
}

function getStepTypeColor(stepType: string): string {
  const type = stepType.toLowerCase()
  if (type.includes('initial') || type.includes('input')) return 'bg-accent-blue/20 text-accent-blue'
  if (type.includes('analysis') || type.includes('think')) return 'bg-accent-purple/20 text-accent-purple'
  if (type.includes('synthesis') || type.includes('combine')) return 'bg-accent-green/20 text-accent-green'
  if (type.includes('evaluation') || type.includes('score')) return 'bg-accent-yellow/20 text-accent-yellow'
  if (type.includes('final') || type.includes('output')) return 'bg-accent-orange/20 text-accent-orange'
  if (type.includes('error') || type.includes('fail')) return 'bg-accent-red/20 text-accent-red'
  return 'bg-sidebar-active text-activitybar-fg'
}

function formatTimestamp(dateStr: string): string {
  try {
    const date = new Date(dateStr)
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return dateStr
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

export default function Traces() {
  const [selectedTrace, setSelectedTrace] = useState<string | null>(null)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())

  const {
    data: tracesData,
    isLoading: tracesLoading,
    error: tracesError,
  } = useQuery<{ traces: TraceSummary[]; total: number }>({
    queryKey: ['traces'],
    queryFn: () => api.get('/api/traces').then((r) => r.data),
  })

  const {
    data: traceDetail,
    isLoading: detailLoading,
  } = useQuery<TraceDetail>({
    queryKey: ['trace', selectedTrace],
    queryFn: () => api.get(`/api/traces/${selectedTrace}`).then((r) => r.data),
    enabled: !!selectedTrace,
  })

  const traces = tracesData?.traces ?? []

  function toggleStep(stepId: string) {
    setExpandedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(stepId)) {
        next.delete(stepId)
      } else {
        next.add(stepId)
      }
      return next
    })
  }

  if (tracesLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-activitybar-fg text-sm">Loading traces...</div>
      </div>
    )
  }

  if (tracesError) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-accent-red text-sm">
          Failed to load traces: {(tracesError as Error).message}
        </div>
      </div>
    )
  }

  // Empty state
  if (traces.length === 0) {
    return (
      <div className="h-full flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-sidebar-border">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-medium text-editor-fg">Reasoning Traces</h1>
            <span className="badge bg-sidebar-active text-activitybar-fg">0</span>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center">
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-16 h-16">
                <path d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
              </svg>
            </div>
            <p className="empty-state-title">No reasoning traces yet</p>
            <p className="empty-state-description">
              Traces are recorded when you run a reasoning engine. Go to the{' '}
              <span className="text-accent-blue">Reasoning Engines</span> page and execute a
              query to generate your first trace. Each run will be stored here with its
              full step-by-step reasoning path.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-medium text-editor-fg">Reasoning Traces</h1>
          <span className="badge badge-blue">{tracesData?.total ?? traces.length}</span>
        </div>
      </div>

      {/* Two-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: trace list */}
        <div className="w-80 flex-shrink-0 border-r border-sidebar-border overflow-y-auto">
          <div className="py-1">
            {traces.map((trace) => {
              const isSelected = selectedTrace === trace.trace_id

              return (
                <div
                  key={trace.trace_id}
                  className={`list-item ${isSelected ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedTrace(trace.trace_id)
                    setExpandedSteps(new Set())
                  }}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-editor-fg truncate">
                      {trace.strategy}
                    </span>
                    <span className="text-2xs text-activitybar-fg flex-shrink-0 ml-2">
                      {formatDuration(trace.duration_ms)}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-2xs text-activitybar-fg mb-1">
                    <span>{trace.steps_count} steps</span>
                    <span className="text-sidebar-border">|</span>
                    <span>{trace.total_llm_calls} calls</span>
                    <span className="text-sidebar-border">|</span>
                    <span>{trace.total_tokens.toLocaleString()} tokens</span>
                  </div>

                  <p className="text-xs text-activitybar-fg line-clamp-2">
                    {trace.final_output_preview}
                  </p>

                  <div className="text-2xs text-activitybar-fg mt-1">
                    {formatTimestamp(trace.created_at)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right panel: trace detail */}
        <div className="flex-1 overflow-y-auto">
          {!selectedTrace ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <p className="text-sm text-activitybar-fg">
                  Select a trace from the list to view its details.
                </p>
              </div>
            </div>
          ) : detailLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-activitybar-fg text-sm">Loading trace detail...</div>
            </div>
          ) : !traceDetail ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-accent-red text-sm">Failed to load trace detail.</div>
            </div>
          ) : (
            <div className="p-4 space-y-4 animate-fade-in">
              {/* Summary stats */}
              <div className="card p-4">
                <div className="flex items-center gap-3 mb-3">
                  <h2 className="text-sm font-medium text-editor-fg">
                    {traceDetail.strategy}
                  </h2>
                  <span className="badge badge-blue">
                    {traceDetail.trace_id.slice(0, 8)}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="bg-editor-bg rounded p-2">
                    <div className="text-2xs text-activitybar-fg mb-0.5">Steps</div>
                    <div className="text-lg font-mono text-token-number">
                      {traceDetail.steps.length}
                    </div>
                  </div>
                  <div className="bg-editor-bg rounded p-2">
                    <div className="text-2xs text-activitybar-fg mb-0.5">LLM Calls</div>
                    <div className="text-lg font-mono text-token-number">
                      {traceDetail.total_llm_calls}
                    </div>
                  </div>
                  <div className="bg-editor-bg rounded p-2">
                    <div className="text-2xs text-activitybar-fg mb-0.5">Tokens</div>
                    <div className="text-lg font-mono text-token-number">
                      {traceDetail.total_tokens.toLocaleString()}
                    </div>
                  </div>
                  <div className="bg-editor-bg rounded p-2">
                    <div className="text-2xs text-activitybar-fg mb-0.5">Duration</div>
                    <div className="text-lg font-mono text-token-number">
                      {formatDuration(traceDetail.duration_ms)}
                    </div>
                  </div>
                </div>

                <div className="text-2xs text-activitybar-fg mt-2">
                  Created: {formatTimestamp(traceDetail.created_at)}
                </div>
              </div>

              {/* Final output */}
              <div className="card p-4">
                <h3 className="text-xs font-medium text-sidebar-fg mb-2">
                  Final Output
                </h3>
                <div className="p-3 rounded bg-editor-bg text-sm text-editor-fg whitespace-pre-wrap leading-relaxed">
                  {traceDetail.final_output}
                </div>
              </div>

              {/* Step timeline */}
              <div className="card p-4">
                <h3 className="text-xs font-medium text-sidebar-fg mb-3">
                  Step-by-Step Timeline ({traceDetail.steps.length} steps)
                </h3>

                <div className="relative">
                  {/* Vertical line connector */}
                  <div className="absolute left-3 top-0 bottom-0 w-px bg-sidebar-border" />

                  <div className="space-y-2">
                    {traceDetail.steps.map((step, idx) => {
                      const isStepExpanded = expandedSteps.has(step.step_id)

                      return (
                        <div key={step.step_id} className="relative pl-8">
                          {/* Timeline dot */}
                          <div className="absolute left-1.5 top-2.5 w-3 h-3 rounded-full bg-sidebar-bg border-2 border-accent-blue z-10" />

                          <div className="rounded bg-editor-bg border border-sidebar-border overflow-hidden">
                            {/* Step header */}
                            <div
                              className="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-list-hover"
                              onClick={() => toggleStep(step.step_id)}
                            >
                              <svg
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                className={`w-3 h-3 flex-shrink-0 text-activitybar-fg transition-transform ${
                                  isStepExpanded ? 'rotate-90' : ''
                                }`}
                              >
                                <path d="M9 5l7 7-7 7" />
                              </svg>

                              <span className="text-2xs text-activitybar-fg font-mono w-5 text-center flex-shrink-0">
                                {idx + 1}
                              </span>

                              <span
                                className={`badge text-2xs flex-shrink-0 ${getStepTypeColor(step.step_type)}`}
                              >
                                {step.step_type}
                              </span>

                              <span className="text-xs text-sidebar-fg truncate flex-1">
                                {step.content.length > 80
                                  ? step.content.slice(0, 80) + '...'
                                  : step.content}
                              </span>

                              {/* Score bar */}
                              <div className="flex items-center gap-1.5 flex-shrink-0">
                                <div className="w-12 h-1.5 bg-sidebar-border rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-accent-green rounded-full transition-all"
                                    style={{
                                      width: `${Math.max(0, Math.min(100, step.score * 100))}%`,
                                    }}
                                  />
                                </div>
                                <span className="text-2xs text-activitybar-fg font-mono w-8 text-right">
                                  {(step.score * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>

                            {/* Step expanded content */}
                            {isStepExpanded && (
                              <div className="px-3 pb-3 border-t border-sidebar-border animate-fade-in">
                                <div className="mt-2 space-y-2">
                                  {/* Full content */}
                                  <div>
                                    <span className="text-2xs text-activitybar-fg block mb-1">
                                      Content
                                    </span>
                                    <div className="text-xs text-editor-fg whitespace-pre-wrap bg-sidebar-bg p-2 rounded">
                                      {step.content}
                                    </div>
                                  </div>

                                  {/* Step metadata row */}
                                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-2xs">
                                    <span className="text-activitybar-fg">
                                      Step ID:{' '}
                                      <span className="text-token-variable font-mono">
                                        {step.step_id}
                                      </span>
                                    </span>
                                    <span className="text-activitybar-fg">
                                      Score:{' '}
                                      <span className="text-token-number">
                                        {step.score.toFixed(4)}
                                      </span>
                                    </span>
                                    {step.parent_step_id && (
                                      <span className="text-activitybar-fg">
                                        Parent:{' '}
                                        <span className="text-token-variable font-mono">
                                          {step.parent_step_id}
                                        </span>
                                      </span>
                                    )}
                                  </div>

                                  {/* Extra metadata */}
                                  {step.metadata &&
                                    Object.keys(step.metadata).length > 0 && (
                                      <div>
                                        <span className="text-2xs text-activitybar-fg block mb-1">
                                          Metadata
                                        </span>
                                        <pre className="text-2xs text-editor-fg bg-sidebar-bg p-2 rounded font-mono overflow-x-auto">
                                          {JSON.stringify(step.metadata, null, 2)}
                                        </pre>
                                      </div>
                                    )}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
