import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { oafApi } from '../api/client'

interface MemoryEntry {
  key: string
  content: string
  importance: number
  access_count: number
  created_at: string
  last_accessed: string
}

type Tier = 'fleeting' | 'short_term' | 'long_term'

const tiers: { id: Tier; label: string; description: string }[] = [
  {
    id: 'fleeting',
    label: 'Fleeting',
    description: 'Per-turn scratchpad. Cleared after each agent turn.',
  },
  {
    id: 'short_term',
    label: 'Short-Term',
    description: 'Session context. Persists across turns within a session.',
  },
  {
    id: 'long_term',
    label: 'Long-Term',
    description: 'Persistent storage. Survives across sessions.',
  },
]

function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text
  return text.slice(0, maxLength) + '...'
}

function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso)
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return iso
  }
}

function ImportanceBar({ value }: { value: number }) {
  const clamped = Math.max(0, Math.min(1, value))
  const width = `${clamped * 100}%`

  let barColor = 'bg-activitybar-fg'
  if (clamped >= 0.8) barColor = 'bg-accent-red'
  else if (clamped >= 0.5) barColor = 'bg-accent-yellow'
  else if (clamped >= 0.2) barColor = 'bg-accent-blue'

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-sidebar-active rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${barColor}`} style={{ width }} />
      </div>
      <span className="text-2xs text-activitybar-fg">{value.toFixed(2)}</span>
    </div>
  )
}

export default function Memory() {
  const [selectedTier, setSelectedTier] = useState<Tier>('fleeting')

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['memory', selectedTier],
    queryFn: () => oafApi.memory.get(selectedTier),
  })

  const tierInfo = tiers.find((t) => t.id === selectedTier)!

  // Normalize the response - the API might return { entries: [...] } or just [...]
  const entries: MemoryEntry[] = Array.isArray(data)
    ? data
    : Array.isArray(data?.entries)
    ? data.entries
    : []

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-lg font-medium text-editor-fg mb-1">Memory System</h1>
        <p className="text-sm text-activitybar-fg">
          3-tier memory hierarchy for agent cognition: fleeting scratchpad, session-scoped
          short-term memory, and persistent long-term storage.
        </p>
      </div>

      {/* Tier Tabs */}
      <div className="flex border-b border-sidebar-border mb-0">
        {tiers.map((tier) => (
          <button
            key={tier.id}
            onClick={() => setSelectedTier(tier.id)}
            className={`px-4 py-2 text-sm transition-colors duration-100 border-b-2 -mb-px ${
              selectedTier === tier.id
                ? 'text-white border-accent-blue'
                : 'text-activitybar-fg border-transparent hover:text-sidebar-fg hover:border-sidebar-border'
            }`}
          >
            {tier.label}
          </button>
        ))}
      </div>

      {/* Tier Description */}
      <div className="panel px-4 py-3 rounded-none border-t-0 mb-4">
        <div className="flex items-center gap-2">
          <svg
            className="w-4 h-4 text-accent-blue flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={1.5}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"
            />
          </svg>
          <span className="text-sm text-sidebar-fg">{tierInfo.description}</span>
        </div>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="panel rounded p-8">
          <div className="flex items-center justify-center gap-3 text-activitybar-fg">
            <svg
              className="w-5 h-5 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
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
            <span className="text-sm">Loading {tierInfo.label.toLowerCase()} memory...</span>
          </div>
        </div>
      ) : isError ? (
        <div className="panel rounded p-8">
          <div className="flex flex-col items-center text-center">
            <svg
              className="w-10 h-10 text-accent-red mb-3"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
              />
            </svg>
            <p className="text-sm text-sidebar-fg mb-1">Failed to load memory entries</p>
            <p className="text-2xs text-activitybar-fg">
              {(error as Error)?.message || 'Unknown error'}
            </p>
          </div>
        </div>
      ) : entries.length === 0 ? (
        <div className="panel rounded p-8">
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
                d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125v-3.75m16.5 3.75v3.75c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125v-3.75"
              />
            </svg>
            <h3 className="empty-state-title">No {tierInfo.label} Memory Entries</h3>
            <p className="empty-state-description">
              {selectedTier === 'fleeting' && (
                <>
                  Fleeting memory is populated automatically during agent execution. Each turn
                  creates temporary scratchpad entries that are cleared when the turn completes.
                  Run an agent to see entries appear here.
                </>
              )}
              {selectedTier === 'short_term' && (
                <>
                  Short-term memory stores session context that persists across agent turns.
                  Entries are created when agents store intermediate results, conversation state,
                  or working context during a session.
                </>
              )}
              {selectedTier === 'long_term' && (
                <>
                  Long-term memory holds persistent knowledge that survives across sessions.
                  Agents promote important findings, learned patterns, and user preferences
                  from short-term memory into long-term storage.
                </>
              )}
            </p>
          </div>
        </div>
      ) : (
        /* Memory Entries Table */
        <div className="panel rounded overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-sidebar-border bg-sidebar-active">
                <th className="text-left px-4 py-2 text-2xs font-medium uppercase tracking-wide text-activitybar-fg">
                  Key
                </th>
                <th className="text-left px-4 py-2 text-2xs font-medium uppercase tracking-wide text-activitybar-fg">
                  Content
                </th>
                <th className="text-left px-4 py-2 text-2xs font-medium uppercase tracking-wide text-activitybar-fg">
                  Importance
                </th>
                <th className="text-right px-4 py-2 text-2xs font-medium uppercase tracking-wide text-activitybar-fg">
                  Accesses
                </th>
                <th className="text-left px-4 py-2 text-2xs font-medium uppercase tracking-wide text-activitybar-fg">
                  Created
                </th>
                <th className="text-left px-4 py-2 text-2xs font-medium uppercase tracking-wide text-activitybar-fg">
                  Last Accessed
                </th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry, index) => (
                <tr
                  key={entry.key}
                  className={`border-b border-sidebar-border last:border-b-0 hover:bg-list-hover transition-colors ${
                    index % 2 === 0 ? 'bg-transparent' : 'bg-sidebar-bg/30'
                  }`}
                >
                  <td className="px-4 py-2.5">
                    <span className="font-mono text-token-variable">{entry.key}</span>
                  </td>
                  <td className="px-4 py-2.5">
                    <span
                      className="text-editor-fg"
                      title={entry.content}
                    >
                      {truncate(entry.content, 80)}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <ImportanceBar value={entry.importance} />
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    <span className="font-mono text-token-number">{entry.access_count}</span>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className="text-activitybar-fg text-2xs">
                      {formatTimestamp(entry.created_at)}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <span className="text-activitybar-fg text-2xs">
                      {formatTimestamp(entry.last_accessed)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Footer with entry count */}
          <div className="px-4 py-2 border-t border-sidebar-border bg-sidebar-active">
            <span className="text-2xs text-activitybar-fg">
              {entries.length} {entries.length === 1 ? 'entry' : 'entries'} in{' '}
              {tierInfo.label.toLowerCase()} memory
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
