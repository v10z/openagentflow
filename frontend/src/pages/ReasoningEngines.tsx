import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api/client'

interface Engine {
  name: string
  description: string
  category: string
}

interface EnginesResponse {
  engines: Engine[]
  total: number
}

interface ReasoningStep {
  step_id: string
  step_type: string
  content: string
  score: number
  metadata: Record<string, any>
  parent_step_id: string | null
}

interface RunResult {
  engine_name: string
  query: string
  final_output: string
  steps: ReasoningStep[]
  total_llm_calls: number
  total_tokens: number
  duration_ms: number
}

const CATEGORY_STYLES: Record<string, { badge: string; border: string; label: string }> = {
  core: {
    badge: 'bg-accent-blue/20 text-accent-blue',
    border: 'border-accent-blue/40',
    label: 'Core Cognitive',
  },
  neuroscience: {
    badge: 'bg-accent-green/20 text-accent-green',
    border: 'border-accent-green/40',
    label: 'Neuroscience',
  },
  physics: {
    badge: 'bg-accent-purple/20 text-accent-purple',
    border: 'border-accent-purple/40',
    label: 'Physics',
  },
}

function getCategoryStyle(category: string) {
  const key = category.toLowerCase()
  for (const [match, style] of Object.entries(CATEGORY_STYLES)) {
    if (key.includes(match)) return style
  }
  return {
    badge: 'bg-sidebar-active text-activitybar-fg',
    border: 'border-sidebar-border',
    label: category,
  }
}

function getStepTypeBadgeStyle(stepType: string): string {
  const type = stepType.toLowerCase()
  if (type.includes('initial') || type.includes('input')) return 'bg-accent-blue/20 text-accent-blue'
  if (type.includes('analysis') || type.includes('think')) return 'bg-accent-purple/20 text-accent-purple'
  if (type.includes('synthesis') || type.includes('combine')) return 'bg-accent-green/20 text-accent-green'
  if (type.includes('evaluation') || type.includes('score')) return 'bg-accent-yellow/20 text-accent-yellow'
  if (type.includes('final') || type.includes('output')) return 'bg-accent-orange/20 text-accent-orange'
  if (type.includes('error') || type.includes('fail')) return 'bg-accent-red/20 text-accent-red'
  return 'bg-sidebar-active text-activitybar-fg'
}

export default function ReasoningEngines() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<RunResult | null>(null)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())

  const { data, isLoading, error } = useQuery<EnginesResponse>({
    queryKey: ['reasoning-engines'],
    queryFn: () => api.get('/api/reasoning/engines').then((r) => r.data),
  })

  const runMutation = useMutation<RunResult, Error, { engine: string; query: string }>({
    mutationFn: ({ engine, query: q }) =>
      api
        .post('/api/reasoning/run', {
          engine_name: engine,
          query: q,
          provider: 'mock',
        })
        .then((r) => r.data),
    onSuccess: (data) => {
      setResult(data)
    },
  })

  const engines = data?.engines ?? []

  const filteredEngines = selectedCategory
    ? engines.filter((e) => {
        const style = getCategoryStyle(e.category)
        return style.label === selectedCategory
      })
    : engines

  // Build category counts
  const categoryCounts: Record<string, number> = {}
  for (const engine of engines) {
    const style = getCategoryStyle(engine.category)
    categoryCounts[style.label] = (categoryCounts[style.label] || 0) + 1
  }

  const categoryTabs = Object.entries(categoryCounts).map(([label, count]) => ({
    label,
    count,
  }))

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

  function handleRun() {
    if (!selectedEngine || !query.trim()) return
    setResult(null)
    setExpandedSteps(new Set())
    runMutation.mutate({ engine: selectedEngine, query: query.trim() })
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-activitybar-fg text-sm">Loading reasoning engines...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-accent-red text-sm">
          Failed to load engines: {(error as Error).message}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-medium text-editor-fg">Reasoning Engines</h1>
          <span className="badge badge-blue">{data?.total ?? 0}</span>
        </div>
      </div>

      {/* Category tabs */}
      <div className="flex items-center gap-1 px-4 py-2 border-b border-sidebar-border overflow-x-auto">
        <button
          className={`btn text-xs whitespace-nowrap ${
            selectedCategory === null ? 'btn-primary' : 'btn-ghost'
          }`}
          onClick={() => setSelectedCategory(null)}
        >
          All ({data?.total ?? 0})
        </button>
        {categoryTabs.map(({ label, count }) => (
          <button
            key={label}
            className={`btn text-xs whitespace-nowrap ${
              selectedCategory === label ? 'btn-primary' : 'btn-ghost'
            }`}
            onClick={() =>
              setSelectedCategory(selectedCategory === label ? null : label)
            }
          >
            {label} ({count})
          </button>
        ))}
      </div>

      {/* Main content area */}
      <div className="flex-1 overflow-y-auto">
        {/* Engine grid */}
        <div className="p-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {filteredEngines.map((engine) => {
              const style = getCategoryStyle(engine.category)
              const isSelected = selectedEngine === engine.name

              return (
                <div
                  key={engine.name}
                  className={`card p-3 transition-all ${
                    isSelected
                      ? `${style.border} border-2 bg-sidebar-active`
                      : 'hover:bg-sidebar-hover'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="text-sm font-medium text-editor-fg truncate">
                          {engine.name}
                        </h3>
                        <span
                          className={`badge text-2xs flex-shrink-0 ${style.badge}`}
                        >
                          {style.label}
                        </span>
                      </div>
                      <p className="text-xs text-activitybar-fg line-clamp-2">
                        {engine.description}
                      </p>
                    </div>
                    <button
                      className={`btn text-xs flex-shrink-0 ${
                        isSelected ? 'btn-secondary' : 'btn-primary'
                      }`}
                      onClick={() => {
                        if (isSelected) {
                          setSelectedEngine(null)
                          setResult(null)
                          setQuery('')
                        } else {
                          setSelectedEngine(engine.name)
                          setResult(null)
                        }
                      }}
                    >
                      {isSelected ? 'Close' : 'Run'}
                    </button>
                  </div>

                  {/* Inline run form */}
                  {isSelected && (
                    <div className="mt-3 pt-3 border-t border-sidebar-border animate-fade-in">
                      <label className="block text-xs text-activitybar-fg mb-1.5">
                        Query
                      </label>
                      <textarea
                        className="input w-full min-h-[80px] resize-y text-xs"
                        placeholder={`Enter a query for ${engine.name}...`}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                            handleRun()
                          }
                        }}
                      />
                      <div className="flex items-center gap-2 mt-2">
                        <button
                          className="btn btn-primary text-xs"
                          onClick={handleRun}
                          disabled={runMutation.isPending || !query.trim()}
                        >
                          {runMutation.isPending ? (
                            <span className="flex items-center gap-2">
                              <svg
                                className="animate-spin w-3 h-3"
                                viewBox="0 0 24 24"
                                fill="none"
                              >
                                <circle
                                  className="opacity-25"
                                  cx="12"
                                  cy="12"
                                  r="10"
                                  stroke="currentColor"
                                  strokeWidth="4"
                                />
                                <path
                                  className="opacity-75"
                                  fill="currentColor"
                                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                                />
                              </svg>
                              Running...
                            </span>
                          ) : (
                            'Run with MockProvider'
                          )}
                        </button>
                        <span className="text-2xs text-activitybar-fg">
                          Ctrl+Enter to run
                        </span>
                      </div>

                      {/* Error */}
                      {runMutation.isError && (
                        <div className="mt-3 p-2 rounded bg-accent-red/10 border border-accent-red/30">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="status-dot status-error" />
                            <span className="text-xs text-accent-red font-medium">
                              Error
                            </span>
                          </div>
                          <p className="text-xs text-accent-red">
                            {runMutation.error?.message || 'Execution failed'}
                          </p>
                        </div>
                      )}

                      {/* Results panel */}
                      {result && result.engine_name === engine.name && (
                        <div className="mt-3 space-y-3 animate-slide-up">
                          {/* Stats bar */}
                          <div className="flex items-center gap-4 p-2 rounded bg-editor-bg">
                            <div className="flex items-center gap-1.5">
                              <span className="status-dot status-success" />
                              <span className="text-xs text-accent-green font-medium">
                                Complete
                              </span>
                            </div>
                            <div className="flex items-center gap-3 ml-auto text-2xs text-activitybar-fg">
                              <span>
                                <span className="text-token-number">{result.total_llm_calls}</span> LLM calls
                              </span>
                              <span>
                                <span className="text-token-number">{result.total_tokens.toLocaleString()}</span> tokens
                              </span>
                              <span>
                                <span className="text-token-number">{result.duration_ms}</span>ms
                              </span>
                            </div>
                          </div>

                          {/* Final output */}
                          <div>
                            <h4 className="text-xs font-medium text-sidebar-fg mb-1.5">
                              Final Output
                            </h4>
                            <div className="p-3 rounded bg-editor-bg border border-sidebar-border text-sm text-editor-fg whitespace-pre-wrap leading-relaxed">
                              {result.final_output}
                            </div>
                          </div>

                          {/* Reasoning steps */}
                          {result.steps.length > 0 && (
                            <div>
                              <h4 className="text-xs font-medium text-sidebar-fg mb-1.5">
                                Reasoning Steps ({result.steps.length})
                              </h4>
                              <div className="space-y-1">
                                {result.steps.map((step, idx) => {
                                  const isStepExpanded = expandedSteps.has(step.step_id)

                                  return (
                                    <div
                                      key={step.step_id}
                                      className="rounded bg-editor-bg border border-sidebar-border overflow-hidden"
                                    >
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
                                          className={`badge text-2xs flex-shrink-0 ${getStepTypeBadgeStyle(step.step_type)}`}
                                        >
                                          {step.step_type}
                                        </span>

                                        <span className="text-xs text-sidebar-fg truncate flex-1">
                                          {step.content.length > 100
                                            ? step.content.slice(0, 100) + '...'
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

                                      {/* Step detail */}
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

                                            {/* Metadata */}
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
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>

        {filteredEngines.length === 0 && (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-16 h-16">
                <path d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5M14.25 3.104c.251.023.501.05.75.082M19 14.5l-2.47 4.526a2.25 2.25 0 01-1.99 1.224H9.46a2.25 2.25 0 01-1.99-1.224L5 14.5m14 0H5" />
              </svg>
            </div>
            <p className="empty-state-title">No engines found</p>
            <p className="empty-state-description">
              Try selecting a different category.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
