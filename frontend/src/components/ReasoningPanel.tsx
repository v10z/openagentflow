interface ReasoningPanelProps {
  step: {
    step_id: string
    step_type: string
    content: string
    score: number
    metadata: Record<string, any>
    parent_step_id: string | null
  } | null
  onClose: () => void
}

// --- Step type styling (matches TraceDAG categories) ---

type StepCategory = 'thesis' | 'antithesis' | 'synthesis' | 'evaluation' | 'default'

function categorizeStepType(stepType: string): StepCategory {
  const lower = stepType.toLowerCase()
  if (['thesis', 'initial', 'dream'].includes(lower)) return 'thesis'
  if (['antithesis', 'critique', 'challenge'].includes(lower)) return 'antithesis'
  if (['synthesis', 'final', 'refinement'].includes(lower)) return 'synthesis'
  if (['evaluation', 'judge'].includes(lower)) return 'evaluation'
  return 'default'
}

const badgeStyles: Record<StepCategory, string> = {
  thesis: 'bg-accent-blue text-white',
  antithesis: 'bg-accent-red text-white',
  synthesis: 'bg-accent-green text-editor-bg',
  evaluation: 'bg-accent-yellow text-editor-bg',
  default: 'bg-sidebar-active text-sidebar-fg',
}

// --- Score visualization ---

function ScoreDisplay({ score }: { score: number }) {
  const clamped = Math.max(0, Math.min(1, score))
  const percentage = `${clamped * 100}%`

  let barColor = 'bg-activitybar-fg'
  let label = 'Low'
  if (clamped >= 0.8) {
    barColor = 'bg-accent-green'
    label = 'High'
  } else if (clamped >= 0.5) {
    barColor = 'bg-accent-blue'
    label = 'Medium'
  } else if (clamped >= 0.3) {
    barColor = 'bg-accent-yellow'
    label = 'Fair'
  } else {
    barColor = 'bg-accent-red'
    label = 'Low'
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-2xs text-activitybar-fg uppercase tracking-wide font-medium">
          Score
        </span>
        <span className="text-sm font-mono text-editor-fg">{clamped.toFixed(3)}</span>
      </div>
      <div className="w-full h-2 bg-sidebar-active rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${barColor}`}
          style={{ width: percentage }}
        />
      </div>
      <div className="flex items-center justify-between mt-1">
        <span className="text-2xs text-activitybar-fg">0.0</span>
        <span className="text-2xs text-sidebar-fg font-medium">{label}</span>
        <span className="text-2xs text-activitybar-fg">1.0</span>
      </div>
    </div>
  )
}

// --- Metadata value renderer ---

function MetadataValue({ value }: { value: any }) {
  if (value === null || value === undefined) {
    return <span className="text-activitybar-fg italic">null</span>
  }
  if (typeof value === 'boolean') {
    return (
      <span className={value ? 'text-accent-green' : 'text-accent-red'}>
        {value ? 'true' : 'false'}
      </span>
    )
  }
  if (typeof value === 'number') {
    return <span className="text-token-number font-mono">{value}</span>
  }
  if (typeof value === 'string') {
    return <span className="text-token-string">{value}</span>
  }
  // Objects and arrays
  return (
    <pre className="text-2xs text-token-variable font-mono bg-editor-bg rounded px-2 py-1 overflow-x-auto max-w-full whitespace-pre-wrap break-all">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

// --- Main component ---

export default function ReasoningPanel({ step, onClose }: ReasoningPanelProps) {
  if (!step) return null

  const category = categorizeStepType(step.step_type)
  const badge = badgeStyles[category]
  const metadataEntries = Object.entries(step.metadata || {})

  return (
    <>
      {/* Backdrop overlay */}
      <div
        className="fixed inset-0 bg-black/30 z-40"
        onClick={onClose}
      />

      {/* Slide-in panel */}
      <div className="fixed top-0 right-0 h-full w-[420px] max-w-[90vw] z-50 panel border-l border-sidebar-border bg-sidebar-bg shadow-2xl flex flex-col animate-fade-in">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-sidebar-border shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <span
              className={`inline-block text-2xs px-1.5 py-0.5 rounded font-medium uppercase tracking-wide shrink-0 ${badge}`}
            >
              {step.step_type}
            </span>
            <span
              className="text-xs text-activitybar-fg font-mono truncate"
              title={step.step_id}
            >
              {step.step_id}
            </span>
          </div>
          <button
            onClick={onClose}
            className="w-6 h-6 flex items-center justify-center rounded hover:bg-sidebar-hover text-activitybar-fg hover:text-sidebar-fg transition-colors shrink-0 ml-2"
            title="Close panel"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto">
          {/* Content section */}
          <div className="px-4 py-3 border-b border-sidebar-border">
            <h3 className="text-2xs text-activitybar-fg uppercase tracking-wide font-medium mb-2">
              Content
            </h3>
            <div className="bg-editor-bg rounded border border-sidebar-border p-3 max-h-[300px] overflow-y-auto">
              <p className="text-sm text-editor-fg leading-relaxed whitespace-pre-wrap break-words">
                {step.content}
              </p>
            </div>
          </div>

          {/* Score section */}
          <div className="px-4 py-3 border-b border-sidebar-border">
            <ScoreDisplay score={step.score} />
          </div>

          {/* Parent step reference */}
          {step.parent_step_id && (
            <div className="px-4 py-3 border-b border-sidebar-border">
              <h3 className="text-2xs text-activitybar-fg uppercase tracking-wide font-medium mb-1.5">
                Parent Step
              </h3>
              <div className="flex items-center gap-2">
                <svg
                  className="w-3.5 h-3.5 text-activitybar-fg shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M4.5 19.5l15-15m0 0H8.25m11.25 0v11.25"
                  />
                </svg>
                <span className="text-xs font-mono text-token-variable">
                  {step.parent_step_id}
                </span>
              </div>
            </div>
          )}

          {/* Metadata section */}
          <div className="px-4 py-3">
            <h3 className="text-2xs text-activitybar-fg uppercase tracking-wide font-medium mb-2">
              Metadata
              {metadataEntries.length > 0 && (
                <span className="ml-1.5 text-2xs bg-sidebar-active px-1.5 py-0.5 rounded normal-case tracking-normal">
                  {metadataEntries.length} {metadataEntries.length === 1 ? 'entry' : 'entries'}
                </span>
              )}
            </h3>

            {metadataEntries.length === 0 ? (
              <div className="text-sm text-activitybar-fg italic py-2">
                No metadata entries for this step.
              </div>
            ) : (
              <div className="border border-sidebar-border rounded overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-sidebar-active border-b border-sidebar-border">
                      <th className="text-left px-3 py-1.5 text-2xs font-medium uppercase tracking-wide text-activitybar-fg w-1/3">
                        Key
                      </th>
                      <th className="text-left px-3 py-1.5 text-2xs font-medium uppercase tracking-wide text-activitybar-fg">
                        Value
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {metadataEntries.map(([key, value], index) => (
                      <tr
                        key={key}
                        className={`border-b border-sidebar-border last:border-b-0 ${
                          index % 2 === 0 ? 'bg-transparent' : 'bg-sidebar-bg/50'
                        }`}
                      >
                        <td className="px-3 py-2 align-top">
                          <span className="font-mono text-token-keyword text-xs">{key}</span>
                        </td>
                        <td className="px-3 py-2 align-top text-xs break-all">
                          <MetadataValue value={value} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-sidebar-border bg-sidebar-active shrink-0">
          <span className="text-2xs text-activitybar-fg">
            Step type: {step.step_type} | Category: {category}
          </span>
        </div>
      </div>
    </>
  )
}
