'use client'

import { useEffect, useState, useCallback } from 'react'
import { api, TrainingStatusResponse } from '@/lib/api'

const STAGES = [
  { key: 'initializing',  label: 'Initializing Engine' },
  { key: 'preprocessing', label: 'Preprocessing Data' },
  { key: 'training',      label: 'Training Models' },
  { key: 'evaluating',    label: 'Evaluating Performance' },
  { key: 'selecting',     label: 'Selecting Best Model' },
  { key: 'saving',        label: 'Saving Artifacts' },
] as const

type StageKey = typeof STAGES[number]['key']

interface Props {
  /** How often to poll in ms — defaults to 2500 */
  pollInterval?: number
  /** Called when status transitions to 'ready' */
  onReady?: () => void
}

export default function TrainingStatus({ pollInterval = 2500, onReady }: Props) {
  const [status, setStatus] = useState<TrainingStatusResponse | null>(null)
  const [visible, setVisible] = useState(false)

  const fetchStatus = useCallback(async () => {
    try {
      const s = await api.getTrainingStatus()
      setStatus(s)
      if (s.status === 'training') {
        setVisible(true)
      } else if (s.status === 'ready' && visible) {
        // Give a moment to show "complete" then fade out
        setTimeout(() => {
          setVisible(false)
          onReady?.()
        }, 2000)
      }
    } catch {
      // Backend may not be up yet — silent fail
    }
  }, [visible, onReady])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, pollInterval)
    return () => clearInterval(id)
  }, [fetchStatus, pollInterval])

  if (!visible || !status) return null
  if (status.status !== 'training' && status.status !== 'ready') return null

  const currentStageIdx = STAGES.findIndex(s => s.key === status.stage)
  const isComplete = status.status === 'ready'

  return (
    <div
      id="training-status-overlay"
      className="fixed bottom-6 right-6 z-50 animate-fade-in-up"
      style={{ maxWidth: '340px' }}
    >
      <div className="glass-card p-5 animate-glow-pulse" style={{ animationDuration: '4s' }}>
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="relative flex-shrink-0">
            <div
              className={`w-2.5 h-2.5 rounded-full ${
                isComplete ? 'bg-[var(--ns-success)]' : 'bg-[var(--ns-electric)] animate-status-pulse'
              }`}
            />
            {!isComplete && (
              <div
                className="absolute inset-0 rounded-full bg-[var(--ns-electric)] opacity-40"
                style={{ animation: 'statusPulse 1.5s ease-in-out infinite' }}
              />
            )}
          </div>
          <div>
            <p
              className="text-xs font-bold uppercase tracking-widest"
              style={{ color: 'var(--ns-electric)' }}
            >
              {isComplete ? 'Engine Ready' : 'AI Engine Training'}
            </p>
            <p className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>
              {isComplete
                ? 'Model loaded successfully'
                : 'Nexsure autonomous pipeline active'}
            </p>
          </div>
        </div>

        {/* Stage Steps */}
        <div className="space-y-2">
          {STAGES.map((stage, idx) => {
            const isPast    = idx < currentStageIdx || isComplete
            const isCurrent = idx === currentStageIdx && !isComplete

            return (
              <div key={stage.key} className="flex items-center gap-3">
                {/* Indicator */}
                <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
                  {isPast ? (
                    <svg
                      className="w-4 h-4"
                      style={{ color: 'var(--ns-success)' }}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2.5}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  ) : isCurrent ? (
                    <div
                      className="w-3 h-3 rounded-full animate-status-pulse"
                      style={{ background: 'var(--ns-electric)' }}
                    />
                  ) : (
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ background: 'var(--ns-text-muted)', opacity: 0.4 }}
                    />
                  )}
                </div>

                {/* Label */}
                <span
                  className="text-xs font-medium"
                  style={{
                    color: isPast
                      ? 'var(--ns-success)'
                      : isCurrent
                      ? 'var(--ns-text-primary)'
                      : 'var(--ns-text-muted)',
                    fontFamily: isCurrent ? 'var(--ns-text-mono)' : undefined,
                  }}
                >
                  {stage.label}
                  {isCurrent && (
                    <span
                      className="ml-1 animate-status-pulse"
                      style={{ color: 'var(--ns-electric)' }}
                    >
                      …
                    </span>
                  )}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
