import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

interface StatusItem {
  id: string
  text: string
  icon?: React.ReactNode
  onClick?: () => void
  position: 'left' | 'right'
  tooltip?: string
}

export default function StatusBar() {
  const [connected, setConnected] = useState(true)

  // Fetch real stats from API
  const { data: stats, isError } = useQuery({
    queryKey: ['health-statusbar'],
    queryFn: () => api.get('/api/health').then((r) => r.data),
    refetchInterval: 10000, // Refresh every 10 seconds
  })

  // Update connection status based on API availability
  useEffect(() => {
    setConnected(!isError)
  }, [isError])

  const leftItems: StatusItem[] = [
    {
      id: 'workspace',
      text: 'OpenAgentFlow',
      position: 'left',
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
        </svg>
      ),
      tooltip: 'Current workspace',
    },
    {
      id: 'sync',
      text: connected ? 'Connected' : 'Disconnected',
      position: 'left',
      icon: connected ? (
        <svg className="w-3 h-3 text-accent-green" viewBox="0 0 24 24" fill="currentColor">
          <circle cx="12" cy="12" r="8" />
        </svg>
      ) : (
        <svg className="w-3 h-3 text-accent-red" viewBox="0 0 24 24" fill="currentColor">
          <circle cx="12" cy="12" r="8" />
        </svg>
      ),
      tooltip: 'Server connection status',
    },
  ]

  const agentsCount = stats?.agents_count ?? 0
  const toolsCount = stats?.tools_count ?? 0
  const enginesCount = stats?.engines_count ?? 0

  const rightItems: StatusItem[] = [
    {
      id: 'agents',
      text: `${agentsCount} Agent${agentsCount !== 1 ? 's' : ''}`,
      position: 'right',
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" />
        </svg>
      ),
      tooltip: 'Registered agents',
      onClick: () => window.location.href = '/agents',
    },
    {
      id: 'tools',
      text: `${toolsCount} Tool${toolsCount !== 1 ? 's' : ''}`,
      position: 'right',
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M11.42 15.17l-5.384 5.384a2.025 2.025 0 01-2.862-2.862l5.384-5.384m2.862 2.862a2.121 2.121 0 003 0l3.879-3.879a2.121 2.121 0 000-3l-.707-.707a2.121 2.121 0 00-3 0L11.42 11.17m0 4a2.121 2.121 0 01-3 0L4.54 11.29a2.121 2.121 0 010-3l.707-.707a2.121 2.121 0 013 0l3.879 3.879a2.121 2.121 0 010 3z" />
        </svg>
      ),
      tooltip: 'Available tools',
      onClick: () => window.location.href = '/tools',
    },
    {
      id: 'engines',
      text: `${enginesCount} Engine${enginesCount !== 1 ? 's' : ''}`,
      position: 'right',
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
        </svg>
      ),
      tooltip: 'Reasoning engines',
      onClick: () => window.location.href = '/reasoning',
    },
    {
      id: 'feedback',
      text: 'Feedback',
      position: 'right',
      icon: (
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
        </svg>
      ),
      tooltip: 'Send feedback',
      onClick: () => window.open('https://github.com/v10z/openagentflow/issues', '_blank'),
    },
  ]

  return (
    <div className="h-6 bg-statusbar-bg flex items-center justify-between px-2 text-statusbar-fg text-xs select-none shrink-0">
      {/* Left Side */}
      <div className="flex items-center">
        {leftItems.map((item) => (
          <StatusBarItem key={item.id} item={item} />
        ))}
      </div>

      {/* Right Side */}
      <div className="flex items-center">
        {rightItems.map((item) => (
          <StatusBarItem key={item.id} item={item} />
        ))}
      </div>
    </div>
  )
}

function StatusBarItem({ item }: { item: StatusItem }) {
  return (
    <button
      onClick={item.onClick}
      className={`flex items-center gap-1.5 px-2 py-0.5 hover:bg-statusbar-hover transition-colors ${
        item.onClick ? 'cursor-pointer' : 'cursor-default'
      }`}
      title={item.tooltip}
    >
      {item.icon}
      <span>{item.text}</span>
    </button>
  )
}
