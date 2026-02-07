import { ReactNode, useState, useEffect, useCallback } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import CommandPalette from './CommandPalette'
import StatusBar from './StatusBar'

// Icons - using simple SVG for VSCode-like feel
const Icons = {
  home: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
    </svg>
  ),
  agent: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" />
    </svg>
  ),
  tool: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M11.42 15.17l-5.384 5.384a2.025 2.025 0 01-2.862-2.862l5.384-5.384m2.862 2.862a2.121 2.121 0 003 0l3.879-3.879a2.121 2.121 0 000-3l-.707-.707a2.121 2.121 0 00-3 0L11.42 11.17m0 4a2.121 2.121 0 01-3 0L4.54 11.29a2.121 2.121 0 010-3l.707-.707a2.121 2.121 0 013 0l3.879 3.879a2.121 2.121 0 010 3z" />
    </svg>
  ),
  brain: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
    </svg>
  ),
  graph: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
    </svg>
  ),
  memory: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125v-3.75m16.5 3.75v3.75c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125v-3.75" />
    </svg>
  ),
  settings: (
    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  chevronRight: (
    <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
    </svg>
  ),
  search: (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
    </svg>
  ),
}

interface LayoutProps {
  children: ReactNode
}

const navigation = [
  { id: 'home', name: 'Dashboard', href: '/', icon: Icons.home, shortName: 'Dashboard' },
  { id: 'agents', name: 'Agents', href: '/agents', icon: Icons.agent },
  { id: 'tools', name: 'Tools', href: '/tools', icon: Icons.tool },
  { id: 'reasoning', name: 'Reasoning', href: '/reasoning', icon: Icons.brain },
  { id: 'traces', name: 'Traces', href: '/traces', icon: Icons.graph },
  { id: 'memory', name: 'Memory', href: '/memory', icon: Icons.memory },
]

const bottomNavigation = [
  { id: 'settings', name: 'Settings', href: '/settings', icon: Icons.settings },
]

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const [activeSection, setActiveSection] = useState('home')

  // Find current page
  const currentPage = [...navigation, ...bottomNavigation].find(
    (n) => location.pathname === n.href || location.pathname.startsWith(n.href + '/')
  )

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K for command palette
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setCommandPaletteOpen(true)
      }
      // Cmd/Ctrl + B to toggle sidebar
      if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault()
        setSidebarOpen((prev) => !prev)
      }
      // Escape to close command palette
      if (e.key === 'Escape') {
        setCommandPaletteOpen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Update active section based on route
  useEffect(() => {
    const item = [...navigation, ...bottomNavigation].find(
      (n) => location.pathname === n.href || location.pathname.startsWith(n.href + '/')
    )
    if (item) {
      setActiveSection(item.id)
    }
  }, [location.pathname])

  return (
    <div className="flex flex-col h-screen bg-editor-bg overflow-hidden">
      {/* Title Bar */}
      <div className="h-8 bg-titlebar-bg flex items-center px-4 text-xs text-titlebar-fg select-none shrink-0">
        <span className="font-medium">OpenAgentFlow</span>
        <span className="mx-2 text-sidebar-border">-</span>
        <span>{currentPage?.shortName || currentPage?.name || 'Dashboard'}</span>
      </div>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Activity Bar */}
        <div className="w-12 bg-activitybar-bg flex flex-col items-center py-1 shrink-0">
          {/* Main Nav */}
          <div className="flex-1 flex flex-col items-center gap-0.5">
            {navigation.map((item) => (
              <Link
                key={item.id}
                to={item.href}
                className={`activity-btn ${activeSection === item.id ? 'active' : ''}`}
                title={item.name}
              >
                {item.icon}
              </Link>
            ))}
          </div>

          {/* Bottom Nav */}
          <div className="flex flex-col items-center gap-0.5 pb-1">
            {bottomNavigation.map((item) => (
              <Link
                key={item.id}
                to={item.href}
                className={`activity-btn ${activeSection === item.id ? 'active' : ''}`}
                title={item.name}
              >
                {item.icon}
              </Link>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        {sidebarOpen && (
          <div className="w-60 bg-sidebar-bg border-r border-sidebar-border flex flex-col shrink-0">
            {/* Sidebar Header */}
            <div className="h-9 flex items-center justify-between px-4 text-[11px] font-medium uppercase tracking-wide text-sidebar-fg shrink-0">
              <span>{currentPage?.name || 'Explorer'}</span>
              <button
                onClick={() => setSidebarOpen(false)}
                className="text-activitybar-fg hover:text-sidebar-fg"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Sidebar Content */}
            <div className="flex-1 overflow-y-auto text-sm">
              <SidebarContent activeSection={activeSection} />
            </div>
          </div>
        )}

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Breadcrumb Bar */}
          <div className="h-6 bg-editor-bg border-b border-sidebar-border flex items-center px-3 text-xs shrink-0">
            <div className="breadcrumb">
              <span className="breadcrumb-item">openagentflow</span>
              <span className="breadcrumb-separator">{Icons.chevronRight}</span>
              <span className="breadcrumb-item">{currentPage?.name?.toLowerCase() || 'dashboard'}</span>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <button
                onClick={() => setCommandPaletteOpen(true)}
                className="flex items-center gap-2 text-activitybar-fg hover:text-sidebar-fg px-2 py-0.5 rounded"
              >
                {Icons.search}
                <span className="text-2xs">Ctrl+K</span>
              </button>
            </div>
          </div>

          {/* Page Content */}
          <main className="flex-1 overflow-auto p-4">
            {children}
          </main>
        </div>
      </div>

      {/* Status Bar */}
      <StatusBar />

      {/* Command Palette */}
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        onNavigate={(href) => {
          navigate(href)
          setCommandPaletteOpen(false)
        }}
      />
    </div>
  )
}

function SidebarContent({ activeSection }: { activeSection: string }) {
  // Fetch real stats from API
  const { data: stats } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.get('/api/health').then((r) => r.data),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Fetch agents for the sidebar
  const { data: agentsData } = useQuery({
    queryKey: ['agents-sidebar'],
    queryFn: () => api.get('/api/agents').then((r) => r.data),
    refetchInterval: 30000,
  })

  // Fetch tools for the sidebar
  const { data: toolsData } = useQuery({
    queryKey: ['tools-sidebar'],
    queryFn: () => api.get('/api/tools').then((r) => r.data),
    refetchInterval: 30000,
  })

  const agents = agentsData?.agents || agentsData || []
  const tools = toolsData?.tools || toolsData || []

  // Contextual sidebar content based on active section
  switch (activeSection) {
    case 'home':
      return (
        <div className="py-1">
          <TreeSection title="Quick Access">
            <TreeItem label="Recent Traces" href="/traces" />
            <TreeItem label="All Agents" href="/agents" />
            <TreeItem label="Reasoning Engines" href="/reasoning" />
          </TreeSection>
          <TreeSection title="Statistics">
            <TreeItem label="Agents" badge={String(stats?.agents_count ?? 0)} />
            <TreeItem label="Tools" badge={String(stats?.tools_count ?? 0)} />
            <TreeItem label="Engines" badge={String(stats?.engines_count ?? 0)} />
          </TreeSection>
        </div>
      )

    case 'agents':
      return (
        <div className="py-1">
          <TreeSection title="Registered Agents" collapsible defaultOpen>
            {(Array.isArray(agents) ? agents : []).length === 0 ? (
              <div className="px-4 py-2 text-xs text-activitybar-fg">No agents registered</div>
            ) : (
              (Array.isArray(agents) ? agents : []).map((a: any) => (
                <TreeItem
                  key={a.name || a.id}
                  label={a.name || a.id}
                  status={a.status === 'active' ? 'success' : a.status === 'idle' ? 'pending' : 'running'}
                  href={`/agents/${a.name || a.id}`}
                />
              ))
            )}
          </TreeSection>
          <TreeSection title="Actions">
            <TreeItem label="+ Register Agent" href="/agents?action=register" />
          </TreeSection>
        </div>
      )

    case 'tools':
      return (
        <div className="py-1">
          <TreeSection title="Tool Categories" collapsible defaultOpen>
            {(Array.isArray(tools) ? tools : []).length === 0 ? (
              <div className="px-4 py-2 text-xs text-activitybar-fg">No tools available</div>
            ) : (
              (Array.isArray(tools) ? tools : []).map((t: any) => (
                <TreeItem
                  key={t.name || t.id}
                  label={t.name || t.id}
                  href={`/tools/${t.name || t.id}`}
                />
              ))
            )}
          </TreeSection>
          <TreeSection title="Actions">
            <TreeItem label="+ Register Tool" href="/tools?action=register" />
          </TreeSection>
        </div>
      )

    case 'reasoning':
      return (
        <div className="py-1">
          <TreeSection title="Engine Categories" collapsible defaultOpen>
            <TreeItem label="Core Engines" href="/reasoning?category=core" />
            <TreeItem label="Neuroscience" href="/reasoning?category=neuroscience" />
            <TreeItem label="Physics" href="/reasoning?category=physics" />
          </TreeSection>
          <TreeSection title="Actions">
            <TreeItem label="Run Engine..." href="/reasoning?action=run" />
          </TreeSection>
        </div>
      )

    case 'traces':
      return (
        <div className="py-1">
          <TreeSection title="Filters">
            <TreeItem label="All Traces" href="/traces" />
            <TreeItem label="Recent (1h)" href="/traces?range=1h" />
            <TreeItem label="Today" href="/traces?range=24h" />
            <TreeItem label="This Week" href="/traces?range=7d" />
          </TreeSection>
          <TreeSection title="By Status">
            <TreeItem label="Completed" href="/traces?status=completed" />
            <TreeItem label="Running" href="/traces?status=running" />
            <TreeItem label="Failed" href="/traces?status=failed" />
          </TreeSection>
        </div>
      )

    case 'memory':
      return (
        <div className="py-1">
          <TreeSection title="Memory Tiers" collapsible defaultOpen>
            <TreeItem label="Fleeting" href="/memory?tier=fleeting" />
            <TreeItem label="Short-Term" href="/memory?tier=short_term" />
            <TreeItem label="Long-Term" href="/memory?tier=long_term" />
          </TreeSection>
          <TreeSection title="Actions">
            <TreeItem label="Clear Fleeting Memory" href="/memory?action=clear&tier=fleeting" />
          </TreeSection>
        </div>
      )

    default:
      return (
        <div className="py-1">
          <TreeSection title="Navigation">
            {navigation.map((item) => (
              <TreeItem key={item.id} label={item.name} href={item.href} />
            ))}
          </TreeSection>
        </div>
      )
  }
}

function TreeSection({
  title,
  children,
  collapsible = false,
  defaultOpen = true,
}: {
  title: string
  children: ReactNode
  collapsible?: boolean
  defaultOpen?: boolean
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="mb-2">
      <button
        onClick={() => collapsible && setIsOpen(!isOpen)}
        className="w-full flex items-center gap-1 px-2 py-1 text-[11px] font-medium uppercase tracking-wide text-sidebar-fg hover:bg-sidebar-hover"
      >
        {collapsible && (
          <svg
            className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-90' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
          </svg>
        )}
        <span>{title}</span>
      </button>
      {isOpen && <div>{children}</div>}
    </div>
  )
}

function TreeItem({
  label,
  badge,
  status,
  href,
}: {
  label: string
  badge?: string
  status?: 'success' | 'error' | 'running' | 'pending'
  href?: string
}) {
  const content = (
    <>
      {status && <span className={`status-dot status-${status}`} />}
      <span className="truncate flex-1">{label}</span>
      {badge && (
        <span className="text-2xs text-activitybar-fg bg-sidebar-active px-1.5 py-0.5 rounded">
          {badge}
        </span>
      )}
    </>
  )

  if (href) {
    return (
      <Link to={href} className="tree-item">
        {content}
      </Link>
    )
  }

  return <div className="tree-item">{content}</div>
}
