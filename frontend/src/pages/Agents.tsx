import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  UserGroupIcon,
  PlayIcon,
  XMarkIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline'

interface Agent {
  name: string
  description: string
  model_id: string
  provider: string
  tools: string[]
  reasoning_strategy: string
}

interface AgentsResponse {
  agents: Agent[]
  total: number
}

interface RunResult {
  output: unknown
  trace_id?: string
  duration_ms?: number
  error?: string
}

export default function Agents() {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)
  const [runInput, setRunInput] = useState<string>('{}')
  const [runResults, setRunResults] = useState<Record<string, RunResult | null>>({})
  const [runErrors, setRunErrors] = useState<Record<string, string | null>>({})

  const { data, isLoading, isError, error } = useQuery<AgentsResponse>({
    queryKey: ['agents'],
    queryFn: () => api.get('/api/agents').then((r) => r.data),
  })

  const runMutation = useMutation({
    mutationFn: ({ name, inputData }: { name: string; inputData: Record<string, unknown> }) =>
      api.post(`/api/agents/${name}/run`, { input_data: inputData }).then((r) => r.data),
    onSuccess: (result: RunResult, variables) => {
      setRunResults((prev) => ({ ...prev, [variables.name]: result }))
      setRunErrors((prev) => ({ ...prev, [variables.name]: null }))
    },
    onError: (err: Error, variables) => {
      setRunResults((prev) => ({ ...prev, [variables.name]: null }))
      setRunErrors((prev) => ({
        ...prev,
        [variables.name]: err instanceof Error ? err.message : 'An unknown error occurred',
      }))
    },
  })

  const handleRun = (agentName: string) => {
    // Clear previous results
    setRunErrors((prev) => ({ ...prev, [agentName]: null }))
    setRunResults((prev) => ({ ...prev, [agentName]: null }))

    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(runInput)
    } catch {
      setRunErrors((prev) => ({ ...prev, [agentName]: 'Invalid JSON input' }))
      return
    }

    runMutation.mutate({ name: agentName, inputData: parsed })
  }

  const toggleExpanded = (agentName: string) => {
    if (expandedAgent === agentName) {
      setExpandedAgent(null)
    } else {
      setExpandedAgent(agentName)
      setRunInput('{}')
    }
  }

  const agents = data?.agents ?? []

  if (isError) {
    return (
      <div className="space-y-6">
        <div className="card p-8">
          <div className="empty-state">
            <UserGroupIcon className="empty-state-icon text-accent-red" />
            <h3 className="empty-state-title text-accent-red">Failed to Load Agents</h3>
            <p className="empty-state-description">
              {error instanceof Error ? error.message : 'Could not fetch agents from the server.'}
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-xl font-semibold text-white">Agents</h1>
        <p className="text-sm text-activitybar-fg mt-1">
          Browse and interact with registered AI agents. Each agent has its own model, tools, and
          reasoning strategy.
        </p>
      </div>

      {/* Summary Bar */}
      <div className="flex items-center gap-4 text-sm text-activitybar-fg">
        <span>
          <span className="text-white font-medium">{isLoading ? '...' : data?.total ?? 0}</span>{' '}
          agents registered
        </span>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="card p-6 animate-pulse">
              <div className="h-5 bg-sidebar-active rounded w-1/3 mb-3" />
              <div className="h-4 bg-sidebar-active rounded w-2/3 mb-4" />
              <div className="flex gap-2">
                <div className="h-5 bg-sidebar-active rounded w-16" />
                <div className="h-5 bg-sidebar-active rounded w-20" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && agents.length === 0 && (
        <div className="card p-8">
          <div className="empty-state">
            <UserGroupIcon className="empty-state-icon" />
            <h3 className="empty-state-title">No Agents Registered</h3>
            <p className="empty-state-description">
              No agents have been registered yet. Register an agent through the API or
              configuration file to get started.
            </p>
          </div>
        </div>
      )}

      {/* Agent Cards Grid */}
      {!isLoading && agents.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {agents.map((agent) => {
            const isExpanded = expandedAgent === agent.name
            const result = runResults[agent.name]
            const runError = runErrors[agent.name]
            const isRunning =
              runMutation.isPending && runMutation.variables?.name === agent.name

            return (
              <div
                key={agent.name}
                className="card flex flex-col overflow-hidden"
              >
                {/* Card Header */}
                <div className="p-4 border-b border-sidebar-border">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <h3 className="text-sm font-semibold text-white truncate">{agent.name}</h3>
                      <p className="text-xs text-activitybar-fg mt-1 line-clamp-2">
                        {agent.description || 'No description provided'}
                      </p>
                    </div>
                  </div>

                  {/* Badges */}
                  <div className="flex flex-wrap items-center gap-1.5 mt-3">
                    <span className="badge badge-blue">{agent.model_id}</span>
                    <span className="badge bg-accent-purple/20 text-accent-purple">
                      {agent.provider}
                    </span>
                    <span className="badge bg-accent-green/20 text-accent-green">
                      {agent.tools.length} tool{agent.tools.length !== 1 ? 's' : ''}
                    </span>
                    {agent.reasoning_strategy && (
                      <span className="badge bg-accent-yellow/20 text-accent-yellow">
                        {agent.reasoning_strategy}
                      </span>
                    )}
                  </div>
                </div>

                {/* Tools List (collapsed) */}
                {agent.tools.length > 0 && (
                  <div className="px-4 py-2 border-b border-sidebar-border">
                    <p className="text-2xs text-activitybar-fg uppercase tracking-wide mb-1">
                      Tools
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {agent.tools.map((tool) => (
                        <span
                          key={tool}
                          className="text-2xs px-1.5 py-0.5 rounded bg-sidebar-active text-sidebar-fg font-mono"
                        >
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Run Section Toggle */}
                <div className="px-4 py-2 flex items-center gap-2">
                  <button
                    onClick={() => toggleExpanded(agent.name)}
                    className="btn btn-primary flex items-center gap-1.5"
                  >
                    {isExpanded ? (
                      <ChevronUpIcon className="w-3.5 h-3.5" />
                    ) : (
                      <PlayIcon className="w-3.5 h-3.5" />
                    )}
                    {isExpanded ? 'Close' : 'Run'}
                  </button>
                </div>

                {/* Expanded Run Form */}
                {isExpanded && (
                  <div className="border-t border-sidebar-border">
                    <div className="p-4 space-y-3">
                      <div>
                        <label className="block text-2xs text-activitybar-fg uppercase tracking-wide mb-1">
                          Input Data (JSON)
                        </label>
                        <textarea
                          value={runInput}
                          onChange={(e) => setRunInput(e.target.value)}
                          className="input w-full font-mono text-xs resize-y"
                          rows={4}
                          placeholder='{"key": "value"}'
                          spellCheck={false}
                        />
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleRun(agent.name)}
                          disabled={isRunning}
                          className="btn btn-primary flex items-center gap-1.5 disabled:opacity-50"
                        >
                          <PlayIcon className="w-3.5 h-3.5" />
                          {isRunning ? 'Running...' : 'Execute'}
                        </button>
                        <button
                          onClick={() => toggleExpanded(agent.name)}
                          className="btn btn-ghost flex items-center gap-1.5"
                        >
                          <XMarkIcon className="w-3.5 h-3.5" />
                          Cancel
                        </button>
                      </div>

                      {/* Validation Error */}
                      {runError && (
                        <div className="rounded border border-accent-red/30 bg-accent-red/5 p-3">
                          <p className="text-xs text-accent-red font-medium">Error</p>
                          <p className="text-xs text-accent-red/80 mt-1">{runError}</p>
                        </div>
                      )}

                      {/* Run Result */}
                      {result && (
                        <div className="rounded border border-accent-green/30 bg-accent-green/5 p-3 space-y-2">
                          <div className="flex items-center justify-between">
                            <p className="text-xs text-accent-green font-medium">Result</p>
                            {result.duration_ms !== undefined && (
                              <span className="text-2xs text-activitybar-fg">
                                {result.duration_ms}ms
                              </span>
                            )}
                          </div>
                          {result.trace_id && (
                            <p className="text-2xs text-activitybar-fg font-mono">
                              Trace: {result.trace_id}
                            </p>
                          )}
                          <pre className="text-xs text-sidebar-fg font-mono bg-editor-bg rounded p-2 overflow-x-auto max-h-48 overflow-y-auto">
                            {JSON.stringify(result.output ?? result, null, 2)}
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
      )}
    </div>
  )
}
