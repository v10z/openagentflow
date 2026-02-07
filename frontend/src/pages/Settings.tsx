import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  ServerIcon,
  Cog6ToothIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline'

interface HealthData {
  status: string
  version: string
  agents_count: number
  tools_count: number
  engines_count: number
  uptime_seconds: number
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  if (hours < 24) return `${hours}h ${minutes}m`
  const days = Math.floor(hours / 24)
  const remainingHours = hours % 24
  return `${days}d ${remainingHours}h ${minutes}m`
}

function getStatusColor(status: string): { text: string; dot: string } {
  switch (status?.toLowerCase()) {
    case 'healthy':
    case 'ok':
    case 'running':
      return { text: 'text-accent-green', dot: 'status-success' }
    case 'degraded':
    case 'warning':
      return { text: 'text-accent-yellow', dot: 'status-warning' }
    case 'unhealthy':
    case 'error':
      return { text: 'text-accent-red', dot: 'status-error' }
    default:
      return { text: 'text-activitybar-fg', dot: 'status-pending' }
  }
}

const LLM_PROVIDERS = [
  { name: 'OpenAI', models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'], color: 'text-accent-green' },
  { name: 'Anthropic', models: ['claude-opus-4-20250514', 'claude-sonnet-4-20250514', 'claude-3-haiku'], color: 'text-accent-orange' },
  { name: 'Mock', models: ['mock-model'], color: 'text-activitybar-fg' },
]

export default function Settings() {
  const { data: health, isLoading, isError, error } = useQuery<HealthData>({
    queryKey: ['health'],
    queryFn: () => api.get('/api/health').then((r) => r.data),
    refetchInterval: 30000,
  })

  const statusColor = getStatusColor(health?.status ?? '')

  if (isError) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-xl font-semibold text-white">Settings</h1>
          <p className="text-sm text-activitybar-fg mt-1">System configuration and information</p>
        </div>
        <div className="card p-8">
          <div className="empty-state">
            <ServerIcon className="empty-state-icon text-accent-red" />
            <h3 className="empty-state-title text-accent-red">Connection Error</h3>
            <p className="empty-state-description">
              {error instanceof Error ? error.message : 'Unable to reach the server.'}
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
        <h1 className="text-xl font-semibold text-white">Settings</h1>
        <p className="text-sm text-activitybar-fg mt-1">
          System configuration, server information, and project details
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Server Information Card */}
        <div className="card flex flex-col">
          <div className="p-4 border-b border-sidebar-border flex items-center gap-2">
            <ServerIcon className="w-4 h-4 text-accent-blue" />
            <h2 className="text-sm font-semibold text-white">Server Information</h2>
          </div>
          <div className="flex-1">
            <table className="w-full">
              <tbody className="divide-y divide-sidebar-border">
                <tr className="hover:bg-sidebar-hover/50">
                  <td className="px-4 py-3 text-sm text-activitybar-fg w-40">Version</td>
                  <td className="px-4 py-3 text-sm text-sidebar-fg font-mono">
                    {isLoading ? '...' : health?.version ?? 'N/A'}
                  </td>
                </tr>
                <tr className="hover:bg-sidebar-hover/50">
                  <td className="px-4 py-3 text-sm text-activitybar-fg w-40">Status</td>
                  <td className="px-4 py-3 text-sm">
                    <div className="flex items-center gap-2">
                      <span className={`status-dot ${statusColor.dot}`} />
                      <span className={statusColor.text}>
                        {isLoading ? '...' : health?.status ?? 'Unknown'}
                      </span>
                    </div>
                  </td>
                </tr>
                <tr className="hover:bg-sidebar-hover/50">
                  <td className="px-4 py-3 text-sm text-activitybar-fg w-40">Uptime</td>
                  <td className="px-4 py-3 text-sm text-sidebar-fg">
                    {isLoading ? '...' : health ? formatUptime(health.uptime_seconds) : 'N/A'}
                  </td>
                </tr>
                <tr className="hover:bg-sidebar-hover/50">
                  <td className="px-4 py-3 text-sm text-activitybar-fg w-40">Agents</td>
                  <td className="px-4 py-3 text-sm text-sidebar-fg">
                    {isLoading ? '...' : health?.agents_count ?? 0}
                  </td>
                </tr>
                <tr className="hover:bg-sidebar-hover/50">
                  <td className="px-4 py-3 text-sm text-activitybar-fg w-40">Tools</td>
                  <td className="px-4 py-3 text-sm text-sidebar-fg">
                    {isLoading ? '...' : health?.tools_count ?? 0}
                  </td>
                </tr>
                <tr className="hover:bg-sidebar-hover/50">
                  <td className="px-4 py-3 text-sm text-activitybar-fg w-40">Engines</td>
                  <td className="px-4 py-3 text-sm text-sidebar-fg">
                    {isLoading ? '...' : health?.engines_count ?? 0}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Configuration Card */}
        <div className="card flex flex-col">
          <div className="p-4 border-b border-sidebar-border flex items-center gap-2">
            <Cog6ToothIcon className="w-4 h-4 text-accent-purple" />
            <h2 className="text-sm font-semibold text-white">Configuration</h2>
            <span className="badge bg-sidebar-active text-activitybar-fg ml-auto">Read-only</span>
          </div>
          <div className="flex-1 p-4 space-y-4">
            {/* LLM Providers */}
            <div>
              <h3 className="text-2xs text-activitybar-fg uppercase tracking-wide mb-2">
                Available LLM Providers
              </h3>
              <div className="space-y-3">
                {LLM_PROVIDERS.map((provider) => (
                  <div
                    key={provider.name}
                    className="rounded border border-sidebar-border p-3"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`status-dot status-success`} />
                      <span className={`text-sm font-medium ${provider.color}`}>
                        {provider.name}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {provider.models.map((model) => (
                        <span
                          key={model}
                          className="text-2xs px-1.5 py-0.5 rounded bg-sidebar-active text-sidebar-fg font-mono"
                        >
                          {model}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* API Configuration */}
            <div>
              <h3 className="text-2xs text-activitybar-fg uppercase tracking-wide mb-2">
                API Configuration
              </h3>
              <div className="rounded border border-sidebar-border">
                <table className="w-full">
                  <tbody className="divide-y divide-sidebar-border">
                    <tr className="hover:bg-sidebar-hover/50">
                      <td className="px-3 py-2 text-xs text-activitybar-fg">Base URL</td>
                      <td className="px-3 py-2 text-xs text-sidebar-fg font-mono">
                        {window.location.origin}
                      </td>
                    </tr>
                    <tr className="hover:bg-sidebar-hover/50">
                      <td className="px-3 py-2 text-xs text-activitybar-fg">Health Endpoint</td>
                      <td className="px-3 py-2 text-xs text-sidebar-fg font-mono">/api/health</td>
                    </tr>
                    <tr className="hover:bg-sidebar-hover/50">
                      <td className="px-3 py-2 text-xs text-activitybar-fg">WebSocket</td>
                      <td className="px-3 py-2 text-xs text-sidebar-fg font-mono">/ws/events</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* About Card */}
      <div className="card">
        <div className="p-4 border-b border-sidebar-border flex items-center gap-2">
          <InformationCircleIcon className="w-4 h-4 text-accent-green" />
          <h2 className="text-sm font-semibold text-white">About OpenAgentFlow</h2>
        </div>
        <div className="p-4">
          <div className="max-w-2xl space-y-4">
            <p className="text-sm text-sidebar-fg leading-relaxed">
              OpenAgentFlow is a modular, open-source framework for building, orchestrating, and
              observing AI agent workflows. It provides a unified interface for managing agents,
              tools, reasoning engines, and memory systems -- enabling rapid development of
              sophisticated multi-agent applications.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div className="rounded border border-sidebar-border p-3">
                <p className="text-2xs text-activitybar-fg uppercase tracking-wide mb-1">License</p>
                <p className="text-sm text-sidebar-fg font-medium">MIT</p>
              </div>
              <div className="rounded border border-sidebar-border p-3">
                <p className="text-2xs text-activitybar-fg uppercase tracking-wide mb-1">Source</p>
                <a
                  href="https://github.com/openagentflow/openagentflow"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-accent-blue hover:underline"
                >
                  GitHub Repository
                </a>
              </div>
              <div className="rounded border border-sidebar-border p-3">
                <p className="text-2xs text-activitybar-fg uppercase tracking-wide mb-1">
                  Framework
                </p>
                <p className="text-sm text-sidebar-fg font-medium">Python + React</p>
              </div>
            </div>

            <div className="rounded border border-sidebar-border p-3">
              <p className="text-2xs text-activitybar-fg uppercase tracking-wide mb-2">
                Key Features
              </p>
              <ul className="space-y-1.5 text-sm text-sidebar-fg">
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 shrink-0" />
                  Agent orchestration with configurable reasoning strategies
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 shrink-0" />
                  Pluggable tool system for extending agent capabilities
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 shrink-0" />
                  Multiple reasoning engines inspired by neuroscience and physics
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 shrink-0" />
                  Tiered memory architecture (fleeting, short-term, long-term)
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 shrink-0" />
                  Full execution tracing and observability
                </li>
                <li className="flex items-start gap-2">
                  <span className="status-dot status-success mt-1.5 shrink-0" />
                  Real-time WebSocket event streaming
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
