import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { Link } from 'react-router-dom'
import {
  UserGroupIcon,
  WrenchScrewdriverIcon,
  CpuChipIcon,
  ServerIcon,
  PlayIcon,
  BeakerIcon,
  CubeTransparentIcon,
  ArrowRightIcon,
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

function getStatusColor(status: string): { text: string; bg: string; dot: string } {
  switch (status?.toLowerCase()) {
    case 'healthy':
    case 'ok':
    case 'running':
      return { text: 'text-accent-green', bg: 'bg-accent-green/10', dot: 'status-success' }
    case 'degraded':
    case 'warning':
      return { text: 'text-accent-yellow', bg: 'bg-accent-yellow/10', dot: 'status-warning' }
    case 'unhealthy':
    case 'error':
      return { text: 'text-accent-red', bg: 'bg-accent-red/10', dot: 'status-error' }
    default:
      return { text: 'text-activitybar-fg', bg: 'bg-activitybar-fg/10', dot: 'status-pending' }
  }
}

export default function Dashboard() {
  const { data: health, isLoading, isError, error } = useQuery<HealthData>({
    queryKey: ['health'],
    queryFn: () => api.get('/api/health').then((r) => r.data),
    refetchInterval: 15000,
  })

  const statusColor = getStatusColor(health?.status ?? '')

  const statCards = [
    {
      name: 'Agents',
      value: health?.agents_count ?? '-',
      icon: UserGroupIcon,
      color: 'text-accent-blue',
      bgColor: 'bg-accent-blue/10',
      href: '/agents',
    },
    {
      name: 'Tools',
      value: health?.tools_count ?? '-',
      icon: WrenchScrewdriverIcon,
      color: 'text-accent-purple',
      bgColor: 'bg-accent-purple/10',
      href: '/tools',
    },
    {
      name: 'Reasoning Engines',
      value: health?.engines_count ?? '-',
      icon: CpuChipIcon,
      color: 'text-accent-green',
      bgColor: 'bg-accent-green/10',
      href: '/reasoning',
    },
    {
      name: 'Server Status',
      value: health ? formatUptime(health.uptime_seconds) : '-',
      icon: ServerIcon,
      color: statusColor.text,
      bgColor: statusColor.bg,
      subtitle: health?.status ?? 'Unknown',
    },
  ]

  const quickActions = [
    {
      label: 'Run Agent',
      description: 'Execute a registered agent with custom input',
      icon: PlayIcon,
      href: '/agents',
      color: 'text-accent-blue',
      bgColor: 'bg-accent-blue/10',
    },
    {
      label: 'Try Reasoning Engine',
      description: 'Run a query through a reasoning engine',
      icon: BeakerIcon,
      href: '/reasoning',
      color: 'text-accent-green',
      bgColor: 'bg-accent-green/10',
    },
    {
      label: 'Explore Tools',
      description: 'Browse available tools and capabilities',
      icon: CubeTransparentIcon,
      href: '/tools',
      color: 'text-accent-purple',
      bgColor: 'bg-accent-purple/10',
    },
  ]

  if (isError) {
    return (
      <div className="space-y-6">
        <div className="card p-8">
          <div className="empty-state">
            <ServerIcon className="empty-state-icon text-accent-red" />
            <h3 className="empty-state-title text-accent-red">Connection Error</h3>
            <p className="empty-state-description">
              Unable to connect to the OpenAgentFlow server.
              {error instanceof Error ? ` ${error.message}` : ''}
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
        <h1 className="text-xl font-semibold text-white">Dashboard</h1>
        <p className="text-sm text-activitybar-fg mt-1">
          OpenAgentFlow system overview and quick actions
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat) => {
          const content = (
            <div className="card p-6 hover:border-sidebar-hover transition-colors">
              <div className="flex items-center">
                <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                  <stat.icon className={`w-6 h-6 ${stat.color}`} />
                </div>
                <div className="ml-4">
                  <p className="text-sm text-activitybar-fg">{stat.name}</p>
                  <p className="text-2xl font-semibold text-white">
                    {isLoading ? '...' : stat.value}
                  </p>
                  {stat.subtitle && (
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className={`status-dot ${statusColor.dot}`} />
                      <span className={`text-xs ${statusColor.text}`}>{stat.subtitle}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )

          if (stat.href) {
            return (
              <Link key={stat.name} to={stat.href}>
                {content}
              </Link>
            )
          }

          return <div key={stat.name}>{content}</div>
        })}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <div className="p-4 border-b border-sidebar-border">
          <h2 className="text-sm font-semibold text-white">Quick Actions</h2>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {quickActions.map((action) => (
              <Link
                key={action.label}
                to={action.href}
                className="flex items-center gap-3 p-3 rounded border border-sidebar-border hover:border-sidebar-hover hover:bg-sidebar-hover/50 transition-colors group"
              >
                <div className={`p-2 rounded-lg ${action.bgColor}`}>
                  <action.icon className={`w-5 h-5 ${action.color}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-sidebar-fg group-hover:text-white transition-colors">
                    {action.label}
                  </p>
                  <p className="text-xs text-activitybar-fg truncate">{action.description}</p>
                </div>
                <ArrowRightIcon className="w-4 h-4 text-activitybar-fg group-hover:text-sidebar-fg transition-colors shrink-0" />
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* System Info */}
      <div className="card">
        <div className="p-4 border-b border-sidebar-border">
          <h2 className="text-sm font-semibold text-white">System Information</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <tbody className="divide-y divide-sidebar-border">
              <tr className="hover:bg-sidebar-hover/50">
                <td className="px-4 py-3 text-sm text-activitybar-fg w-48">Version</td>
                <td className="px-4 py-3 text-sm text-sidebar-fg font-mono">
                  {isLoading ? '...' : health?.version ?? 'N/A'}
                </td>
              </tr>
              <tr className="hover:bg-sidebar-hover/50">
                <td className="px-4 py-3 text-sm text-activitybar-fg w-48">Status</td>
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
                <td className="px-4 py-3 text-sm text-activitybar-fg w-48">Uptime</td>
                <td className="px-4 py-3 text-sm text-sidebar-fg">
                  {isLoading ? '...' : health ? formatUptime(health.uptime_seconds) : 'N/A'}
                </td>
              </tr>
              <tr className="hover:bg-sidebar-hover/50">
                <td className="px-4 py-3 text-sm text-activitybar-fg w-48">Agents Loaded</td>
                <td className="px-4 py-3 text-sm text-sidebar-fg">
                  {isLoading ? '...' : health?.agents_count ?? 0}
                </td>
              </tr>
              <tr className="hover:bg-sidebar-hover/50">
                <td className="px-4 py-3 text-sm text-activitybar-fg w-48">Tools Available</td>
                <td className="px-4 py-3 text-sm text-sidebar-fg">
                  {isLoading ? '...' : health?.tools_count ?? 0}
                </td>
              </tr>
              <tr className="hover:bg-sidebar-hover/50">
                <td className="px-4 py-3 text-sm text-activitybar-fg w-48">Reasoning Engines</td>
                <td className="px-4 py-3 text-sm text-sidebar-fg">
                  {isLoading ? '...' : health?.engines_count ?? 0}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
