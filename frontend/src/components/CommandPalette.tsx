import { useState, useEffect, useRef, useMemo } from 'react'

interface Command {
  id: string
  label: string
  shortcut?: string
  category: string
  action: () => void
}

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
  onNavigate: (href: string) => void
}

export default function CommandPalette({ isOpen, onClose, onNavigate }: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // Define commands
  const commands: Command[] = useMemo(
    () => [
      // Navigation
      { id: 'nav-dashboard', label: 'Go to Dashboard', category: 'Navigation', action: () => onNavigate('/') },
      { id: 'nav-agents', label: 'Go to Agents', category: 'Navigation', action: () => onNavigate('/agents') },
      { id: 'nav-tools', label: 'Go to Tools', category: 'Navigation', action: () => onNavigate('/tools') },
      { id: 'nav-reasoning', label: 'Go to Reasoning Engines', category: 'Navigation', action: () => onNavigate('/reasoning') },
      { id: 'nav-traces', label: 'Go to Traces', category: 'Navigation', action: () => onNavigate('/traces') },
      { id: 'nav-memory', label: 'Go to Memory', category: 'Navigation', action: () => onNavigate('/memory') },
      { id: 'nav-settings', label: 'Go to Settings', category: 'Navigation', action: () => onNavigate('/settings') },

      // Actions
      { id: 'action-run-engine', label: 'Run Reasoning Engine...', shortcut: 'Ctrl+R', category: 'Actions', action: () => onNavigate('/reasoning?action=run') },

      // View
      { id: 'view-toggle-sidebar', label: 'Toggle Sidebar', shortcut: 'Ctrl+B', category: 'View', action: () => {} },

      // Help
      { id: 'help-docs', label: 'Open Documentation', category: 'Help', action: () => window.open('https://github.com/v10z/openagentflow', '_blank') },
      { id: 'help-shortcuts', label: 'Keyboard Shortcuts', shortcut: 'Ctrl+/', category: 'Help', action: () => {} },
    ],
    [onNavigate]
  )

  // Filter commands based on query
  const filteredCommands = useMemo(() => {
    if (!query) return commands
    const lowerQuery = query.toLowerCase()
    return commands.filter(
      (cmd) =>
        cmd.label.toLowerCase().includes(lowerQuery) ||
        cmd.category.toLowerCase().includes(lowerQuery)
    )
  }, [commands, query])

  // Group commands by category
  const groupedCommands = useMemo(() => {
    const groups: Record<string, Command[]> = {}
    filteredCommands.forEach((cmd) => {
      if (!groups[cmd.category]) {
        groups[cmd.category] = []
      }
      groups[cmd.category].push(cmd)
    })
    return groups
  }, [filteredCommands])

  // Reset on open/close
  useEffect(() => {
    if (isOpen) {
      setQuery('')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex((prev) => Math.min(prev + 1, filteredCommands.length - 1))
          break
        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex((prev) => Math.max(prev - 1, 0))
          break
        case 'Enter':
          e.preventDefault()
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action()
            onClose()
          }
          break
        case 'Escape':
          e.preventDefault()
          onClose()
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, selectedIndex, filteredCommands, onClose])

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current) {
      const selectedElement = listRef.current.querySelector(`[data-index="${selectedIndex}"]`)
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest' })
      }
    }
  }, [selectedIndex])

  if (!isOpen) return null

  let flatIndex = 0

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="command-palette animate-fade-in">
        {/* Search Input */}
        <div className="flex items-center border-b border-sidebar-border">
          <svg className="w-4 h-4 ml-4 text-activitybar-fg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value)
              setSelectedIndex(0)
            }}
            placeholder="Type a command or search..."
            className="command-input"
          />
        </div>

        {/* Command List */}
        <div ref={listRef} className="command-list">
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-activitybar-fg text-sm">
              No commands found
            </div>
          ) : (
            Object.entries(groupedCommands).map(([category, cmds]) => (
              <div key={category}>
                <div className="px-4 py-1 text-[11px] font-medium uppercase tracking-wide text-activitybar-fg">
                  {category}
                </div>
                {cmds.map((cmd) => {
                  const index = flatIndex++
                  return (
                    <div
                      key={cmd.id}
                      data-index={index}
                      onClick={() => {
                        cmd.action()
                        onClose()
                      }}
                      className={`command-item ${index === selectedIndex ? 'selected' : ''}`}
                    >
                      <span className="command-item-label">{cmd.label}</span>
                      {cmd.shortcut && (
                        <span className="command-item-shortcut">{cmd.shortcut}</span>
                      )}
                    </div>
                  )
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-sidebar-border text-2xs text-activitybar-fg">
          <div className="flex items-center gap-4">
            <span><kbd className="px-1 py-0.5 bg-sidebar-active rounded text-[10px]">&#8593;&#8595;</kbd> navigate</span>
            <span><kbd className="px-1 py-0.5 bg-sidebar-active rounded text-[10px]">&#8629;</kbd> select</span>
            <span><kbd className="px-1 py-0.5 bg-sidebar-active rounded text-[10px]">esc</kbd> close</span>
          </div>
        </div>
      </div>
    </>
  )
}
