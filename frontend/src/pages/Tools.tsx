import { useState, useMemo } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { api } from '../api/client'

interface Tool {
  name: string
  description: string
  category: string
  parameters: Record<string, any>
}

interface ToolsResponse {
  tools: Tool[]
  total: number
  categories: string[]
}

interface ExecuteResult {
  tool_name: string
  result: any
  duration_ms: number
}

export default function Tools() {
  const [search, setSearch] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [expandedTool, setExpandedTool] = useState<string | null>(null)
  const [toolArgs, setToolArgs] = useState<Record<string, Record<string, string>>>({})

  const { data, isLoading, error } = useQuery<ToolsResponse>({
    queryKey: ['tools'],
    queryFn: () => api.get('/api/tools').then((r) => r.data),
  })

  const executeMutation = useMutation<ExecuteResult, Error, { name: string; args: Record<string, any> }>({
    mutationFn: ({ name, args }) =>
      api.post(`/api/tools/${name}/execute`, { arguments: args }).then((r) => r.data),
  })

  const filteredTools = useMemo(() => {
    if (!data?.tools) return []
    let tools = data.tools

    if (selectedCategory) {
      tools = tools.filter((t) => t.category === selectedCategory)
    }

    if (search.trim()) {
      const q = search.toLowerCase()
      tools = tools.filter(
        (t) =>
          t.name.toLowerCase().includes(q) ||
          t.description.toLowerCase().includes(q)
      )
    }

    return tools
  }, [data?.tools, selectedCategory, search])

  const groupedTools = useMemo(() => {
    const groups: Record<string, Tool[]> = {}
    for (const tool of filteredTools) {
      if (!groups[tool.category]) groups[tool.category] = []
      groups[tool.category].push(tool)
    }
    return groups
  }, [filteredTools])

  const categoryCount = useMemo(() => {
    if (!data?.tools) return {}
    const counts: Record<string, number> = {}
    for (const tool of data.tools) {
      counts[tool.category] = (counts[tool.category] || 0) + 1
    }
    return counts
  }, [data?.tools])

  function getArgsForTool(toolName: string): Record<string, string> {
    return toolArgs[toolName] || {}
  }

  function setArgForTool(toolName: string, paramName: string, value: string) {
    setToolArgs((prev) => ({
      ...prev,
      [toolName]: {
        ...(prev[toolName] || {}),
        [paramName]: value,
      },
    }))
  }

  function handleExecute(tool: Tool) {
    const args = getArgsForTool(tool.name)
    const parsedArgs: Record<string, any> = {}
    for (const [key, value] of Object.entries(args)) {
      try {
        parsedArgs[key] = JSON.parse(value)
      } catch {
        parsedArgs[key] = value
      }
    }
    executeMutation.mutate({ name: tool.name, args: parsedArgs })
  }

  function renderParameterFields(tool: Tool) {
    const params = tool.parameters
    if (!params || Object.keys(params).length === 0) {
      return (
        <p className="text-activitybar-fg text-xs italic">
          No parameters required
        </p>
      )
    }

    const args = getArgsForTool(tool.name)

    return (
      <div className="space-y-2">
        {Object.entries(params).map(([paramName, paramDef]) => {
          const def = paramDef as Record<string, any>
          return (
            <div key={paramName}>
              <label className="block text-xs text-activitybar-fg mb-1">
                <span className="font-mono text-token-variable">{paramName}</span>
                {def.type && (
                  <span className="ml-2 text-token-comment">({def.type})</span>
                )}
                {def.required && (
                  <span className="ml-1 text-accent-red">*</span>
                )}
              </label>
              {def.description && (
                <p className="text-2xs text-activitybar-fg mb-1">{def.description}</p>
              )}
              <input
                type="text"
                className="input w-full text-xs"
                placeholder={def.default !== undefined ? `Default: ${def.default}` : `Enter ${paramName}...`}
                value={args[paramName] || ''}
                onChange={(e) => setArgForTool(tool.name, paramName, e.target.value)}
              />
            </div>
          )
        })}
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-activitybar-fg text-sm">Loading tools...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-accent-red text-sm">
          Failed to load tools: {(error as Error).message}
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-sidebar-border">
        <div className="flex items-center gap-3">
          <h1 className="text-base font-medium text-editor-fg">Tools</h1>
          <span className="badge badge-blue">{data?.total ?? 0}</span>
        </div>
      </div>

      {/* Search */}
      <div className="px-4 py-2 border-b border-sidebar-border">
        <input
          type="text"
          className="input w-full"
          placeholder="Search tools by name or description..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
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
        {(data?.categories ?? []).map((cat) => (
          <button
            key={cat}
            className={`btn text-xs whitespace-nowrap ${
              selectedCategory === cat ? 'btn-primary' : 'btn-ghost'
            }`}
            onClick={() =>
              setSelectedCategory(selectedCategory === cat ? null : cat)
            }
          >
            {cat} ({categoryCount[cat] || 0})
          </button>
        ))}
      </div>

      {/* Tool list */}
      <div className="flex-1 overflow-y-auto">
        {filteredTools.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-16 h-16">
                <path d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
              </svg>
            </div>
            <p className="empty-state-title">No tools found</p>
            <p className="empty-state-description">
              Try adjusting your search or category filter.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-sidebar-border">
            {Object.entries(groupedTools).map(([category, tools]) => (
              <div key={category}>
                {/* Category group header */}
                <div className="sticky top-0 z-10 bg-editor-bg px-4 py-2 flex items-center gap-2 border-b border-sidebar-border">
                  <span className="text-xs font-medium text-token-type uppercase tracking-wider">
                    {category}
                  </span>
                  <span className="text-2xs text-activitybar-fg">
                    ({tools.length})
                  </span>
                </div>

                {/* Tools in this category */}
                {tools.map((tool) => {
                  const isExpanded = expandedTool === tool.name
                  const isExecuting =
                    executeMutation.isPending &&
                    executeMutation.variables?.name === tool.name

                  return (
                    <div
                      key={tool.name}
                      className={`border-b border-sidebar-border transition-colors ${
                        isExpanded ? 'bg-sidebar-bg' : 'hover:bg-list-hover'
                      }`}
                    >
                      {/* Tool summary row */}
                      <div
                        className="px-4 py-2.5 cursor-pointer flex items-start gap-3"
                        onClick={() =>
                          setExpandedTool(isExpanded ? null : tool.name)
                        }
                      >
                        <svg
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          className={`w-3 h-3 mt-1.5 flex-shrink-0 text-activitybar-fg transition-transform ${
                            isExpanded ? 'rotate-90' : ''
                          }`}
                        >
                          <path d="M9 5l7 7-7 7" />
                        </svg>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-sm text-token-function truncate">
                              {tool.name}
                            </span>
                            <span className="badge text-2xs bg-sidebar-active text-activitybar-fg flex-shrink-0">
                              {tool.category}
                            </span>
                          </div>
                          <p className="text-xs text-activitybar-fg mt-0.5 line-clamp-2">
                            {tool.description}
                          </p>
                        </div>
                        <div className="text-2xs text-activitybar-fg flex-shrink-0 mt-0.5">
                          {Object.keys(tool.parameters || {}).length} params
                        </div>
                      </div>

                      {/* Expanded panel */}
                      {isExpanded && (
                        <div className="px-4 pb-4 pl-10 animate-fade-in">
                          <div className="card p-3 space-y-3">
                            <h3 className="text-xs font-medium text-sidebar-fg">
                              Parameters
                            </h3>

                            {renderParameterFields(tool)}

                            <div className="flex items-center gap-2 pt-2">
                              <button
                                className="btn btn-primary text-xs"
                                onClick={() => handleExecute(tool)}
                                disabled={isExecuting}
                              >
                                {isExecuting ? (
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
                                    Executing...
                                  </span>
                                ) : (
                                  'Execute'
                                )}
                              </button>
                              <button
                                className="btn btn-ghost text-xs"
                                onClick={() => {
                                  setToolArgs((prev) => {
                                    const next = { ...prev }
                                    delete next[tool.name]
                                    return next
                                  })
                                }}
                              >
                                Clear
                              </button>
                            </div>

                            {/* Execution result */}
                            {executeMutation.isSuccess &&
                              executeMutation.variables?.name === tool.name && (
                                <div className="mt-3 border-t border-sidebar-border pt-3">
                                  <div className="flex items-center gap-2 mb-2">
                                    <span className="status-dot status-success" />
                                    <span className="text-xs text-accent-green font-medium">
                                      Success
                                    </span>
                                    <span className="text-2xs text-activitybar-fg ml-auto">
                                      {executeMutation.data.duration_ms}ms
                                    </span>
                                  </div>
                                  <pre className="text-xs text-editor-fg bg-editor-bg rounded p-2 overflow-x-auto max-h-48 overflow-y-auto font-mono">
                                    {typeof executeMutation.data.result === 'string'
                                      ? executeMutation.data.result
                                      : JSON.stringify(executeMutation.data.result, null, 2)}
                                  </pre>
                                </div>
                              )}

                            {executeMutation.isError &&
                              executeMutation.variables?.name === tool.name && (
                                <div className="mt-3 border-t border-sidebar-border pt-3">
                                  <div className="flex items-center gap-2 mb-2">
                                    <span className="status-dot status-error" />
                                    <span className="text-xs text-accent-red font-medium">
                                      Error
                                    </span>
                                  </div>
                                  <pre className="text-xs text-accent-red bg-editor-bg rounded p-2 overflow-x-auto font-mono">
                                    {executeMutation.error?.message || 'Execution failed'}
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
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
