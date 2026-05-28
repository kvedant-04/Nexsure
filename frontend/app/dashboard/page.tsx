'use client'

import { useState, useEffect, useCallback } from 'react'
import Link from 'next/link'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList
} from 'recharts'
import {
  api,
  SystemInfoResponse,
  ModelInsightsResponse,
  PredictionLog,
  VersionHistoryEntry,
  formatPercent,
  formatLatency,
  formatTimestamp,
  formatErrorMessage,
} from '@/lib/api'
import AIObservatory from '@/components/AIObservatory'
import TrainingStatus from '@/components/TrainingStatus'

// ─── Subcomponents ────────────────────────────────────────────────────────────

function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`rounded-xl animate-shimmer ${className}`} style={{ minHeight: '1rem' }} />
}

function KPICard({
  label,
  value,
  subLabel,
  color,
  loading,
}: {
  label: string
  value: string
  subLabel?: string
  color: string
  loading: boolean
}) {
  return (
    <div
      className="glass-card p-5 flex flex-col"
      style={{ borderTop: `2px solid ${color}40` }}
    >
      <p className="text-xs uppercase tracking-widest font-bold mb-1" style={{ color: 'var(--ns-text-muted)' }}>
        {label}
      </p>
      {loading ? (
        <Skeleton className="h-8 w-24 mb-1" />
      ) : (
        <p className="text-3xl font-bold mono animate-count-up" style={{ color }}>
          {value}
        </p>
      )}
      {subLabel && (
        <p className="text-xs mt-1" style={{ color: 'var(--ns-text-muted)' }}>
          {subLabel}
        </p>
      )}
    </div>
  )
}

function ModelHealthBadge({ health }: { health: string }) {
  const cls =
    health === 'healthy'     ? 'badge-healthy' :
    health === 'training'    ? 'badge-training' :
    health === 'degraded'    ? 'badge-degraded' :
    'badge-unavailable'

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${cls}`}>
      <span className="w-1.5 h-1.5 rounded-full animate-status-pulse" style={{ background: 'currentColor' }} />
      {health}
    </span>
  )
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const [[tn, fp], [fn, tp]] = matrix
  const total = tn + fp + fn + tp
  
  const getIntensity = (val: number) => {
    if (total === 0) return 0
    return Math.max(0.1, val / total) // Min opacity 10%
  }

  const cells = [
    { label: 'True Neg.', abbr: 'TN', value: tn, intent: 'correct' },
    { label: 'False Pos.', abbr: 'FP', value: fp, intent: 'incorrect' },
    { label: 'False Neg.', abbr: 'FN', value: fn, intent: 'incorrect' },
    { label: 'True Pos.', abbr: 'TP', value: tp, intent: 'correct' },
  ]
  
  return (
    <div className="grid grid-cols-2 gap-3">
      {cells.map(c => {
        const intensity = getIntensity(c.value)
        const baseColor = c.intent === 'correct' ? 'var(--ns-electric-rgb, 0, 132, 255)' : 'var(--ns-slate-rgb, 100, 116, 139)'
        const bgColor = `rgba(${baseColor}, ${intensity * 0.25})`
        const borderColor = `rgba(${baseColor}, ${intensity * 0.5 + 0.2})`
        const textColor = c.intent === 'correct' ? 'var(--ns-electric)' : 'var(--ns-text-secondary)'

        return (
          <div
            key={c.abbr}
            className="flex flex-col items-center justify-center rounded-xl p-4 transition-all"
            style={{ 
              background: bgColor, 
              border: `1px solid ${borderColor}`,
              boxShadow: `inset 0 1px 0 rgba(255,255,255,0.05)`
            }}
          >
            <span className="text-3xl font-bold mono" style={{ color: textColor }}>{c.value}</span>
            <span className="text-xs font-bold tracking-widest mt-1" style={{ color: textColor }}>{c.abbr}</span>
            <span className="text-xs mt-0.5" style={{ color: 'var(--ns-text-muted)' }}>{c.label}</span>
          </div>
        )
      })}
    </div>
  )
}

// ─── Main Dashboard ────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [systemInfo, setSystemInfo]       = useState<SystemInfoResponse | null>(null)
  const [modelInsights, setModelInsights] = useState<ModelInsightsResponse | null>(null)
  const [logs, setLogs]                   = useState<PredictionLog[]>([])
  const [versionHistory, setVersionHistory] = useState<VersionHistoryEntry[]>([])
  const [loading, setLoading]             = useState(true)
  const [error, setError]                 = useState<string | null>(null)
  const [lastRefresh, setLastRefresh]     = useState<string>('')

  const fetchAll = useCallback(async () => {
    try {
      const [info, insights, logData, histData] = await Promise.allSettled([
        api.getSystemInfo(),
        api.getModelInsights(),
        api.getLogs(),
        api.getVersionHistory(),
      ])

      if (info.status === 'fulfilled')     setSystemInfo(info.value)
      if (insights.status === 'fulfilled') setModelInsights(insights.value)
      if (logData.status === 'fulfilled')  setLogs(logData.value)
      if (histData.status === 'fulfilled') setVersionHistory(histData.value)

      setLastRefresh(new Date().toLocaleTimeString())
      setError(null)
    } catch (err) {
      setError(formatErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchAll()
  }, [fetchAll])

  // Feature importance chart data
  const featureChartData = modelInsights?.feature_importance
    ? Object.entries(modelInsights.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 7)
        .map(([name, value]) => ({ name: name.replace('_', ' ').title(), value: +(value * 100).toFixed(1) }))
    : []

  const BAR_COLORS = ['#3b82f6', '#6366f1', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444', '#0ea5e9']

  // Performance metrics chart data
  const performanceChartData = systemInfo ? [
    { name: 'Accuracy', value: +(systemInfo.accuracy * 100).toFixed(2), fill: '#005ac2' },
    { name: 'Precision', value: +(systemInfo.precision * 100).toFixed(2), fill: '#059669' },
    { name: 'Recall', value: +(systemInfo.recall * 100).toFixed(2), fill: '#6366f1' },
    { name: 'F1 Score', value: +(systemInfo.f1_score * 100).toFixed(2), fill: '#d97706' },
  ] : []

  return (
    <>
      <TrainingStatus onReady={fetchAll} />

      {/* Top Nav */}
      <nav
        className="fixed top-0 left-0 right-0 z-40"
        style={{
          background: 'var(--ns-bg-overlay)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid var(--ns-border)',
        }}
      >
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-2 text-sm transition-colors"
              style={{ color: 'var(--ns-text-muted)' }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Home
            </Link>
            <span style={{ color: 'var(--ns-text-muted)', opacity: 0.5 }}>·</span>
            <h1 className="text-sm font-bold" style={{ color: 'var(--ns-text-primary)' }}>
              Observatory
            </h1>
          </div>

          <div className="flex items-center gap-4">
            {systemInfo && (
              <ModelHealthBadge health={systemInfo.model_health} />
            )}
            {lastRefresh && (
              <span className="text-xs mono" style={{ color: 'var(--ns-text-muted)' }}>
                Last sync: {lastRefresh}
              </span>
            )}
            <button
              onClick={fetchAll}
              className="btn-ghost text-xs px-3 py-1.5"
              disabled={loading}
            >
              Refresh
            </button>
          </div>
        </div>
      </nav>

      <main className="pt-20 pb-16 max-w-7xl mx-auto px-6">

        {error && (
          <div
            className="mb-6 px-5 py-3 rounded-xl text-sm animate-fade-in-up"
            style={{
              background: 'var(--ns-danger-dim)',
              border: '1px solid rgba(220,38,38,0.25)',
              color: 'var(--ns-danger)',
            }}
          >
            ⚠ {error}
          </div>
        )}

        {/* ── Page Header ── */}
        <div className="mb-8 animate-fade-in-up">
          <h2 className="text-3xl font-bold mb-1" style={{ color: 'var(--ns-text-primary)' }}>
            Observatory
          </h2>
          <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
            Real-time performance metrics, feature analysis, and prediction audit trail
          </p>
        </div>

        {/* ── KPI Strip ── */}
        <section
          className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6 animate-fade-in-up"
          style={{ animationDelay: '0.05s' }}
        >
          <KPICard
            label="Accuracy"
            value={formatPercent(systemInfo?.accuracy)}
            subLabel="Overall correctness"
            color="#00d2ff"
            loading={loading}
          />
          <KPICard
            label="Precision"
            value={formatPercent(systemInfo?.precision)}
            subLabel="True positive rate"
            color="#10b981"
            loading={loading}
          />
          <KPICard
            label="Recall"
            value={formatPercent(systemInfo?.recall)}
            subLabel="Coverage of positives"
            color="#a78bfa"
            loading={loading}
          />
          <KPICard
            label="F1 Score"
            value={formatPercent(systemInfo?.f1_score)}
            subLabel="Harmonic mean"
            color="#f59e0b"
            loading={loading}
          />
        </section>

        {/* ── Analytics Row ── */}
        <section
          className="grid lg:grid-cols-2 gap-5 mb-6 animate-fade-in-up"
          style={{ animationDelay: '0.1s' }}
        >
          {/* Confusion Matrix */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: 'var(--ns-text-secondary)' }}>
              Confusion Matrix
            </h3>
            {loading ? (
              <div className="grid grid-cols-2 gap-2">
                {[1,2,3,4].map(i => <Skeleton key={i} className="h-24" />)}
              </div>
            ) : systemInfo?.confusion_matrix ? (
              <ConfusionMatrix matrix={systemInfo.confusion_matrix} />
            ) : (
              <div className="flex items-center justify-center h-40 text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                Not Available
              </div>
            )}
          </div>

          {/* Feature Importance */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: 'var(--ns-text-secondary)' }}>
              AI Decision Intelligence (Global Mode)
            </h3>
            {loading ? (
              <div className="space-y-2">
                {[1,2,3,4,5].map(i => <Skeleton key={i} className="h-6" />)}
              </div>
            ) : featureChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={featureChartData} layout="vertical" margin={{ left: 8, right: 24 }}>
                  <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} tickFormatter={v => `${v}%`} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    width={100}
                  />
                  <Tooltip
                    formatter={(v: number) => [`${v}%`, 'Decision Impact']}
                    contentStyle={{
                      background: 'var(--ns-bg-overlay)',
                      border: '1px solid var(--ns-border)',
                      borderRadius: '8px',
                      color: 'var(--ns-text-primary)',
                      boxShadow: '0 4px 20px rgba(15,23,42,0.08)'
                    }}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {featureChartData.map((_, i) => (
                      <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-40 text-sm text-center px-4" style={{ color: 'var(--ns-text-muted)' }}>
                Live model reasoning becomes available after an active underwriting assessment.
              </div>
            )}
          </div>
        </section>

        {/* ── Model Performance Metrics ── */}
        {performanceChartData.length > 0 && (
          <section
            className="glass-card p-6 mb-6 animate-fade-in-up"
            style={{ animationDelay: '0.15s' }}
          >
            <h3 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: 'var(--ns-text-secondary)' }}>
              Model Performance Metrics
            </h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={performanceChartData} margin={{ top: 24, right: 16, left: -16, bottom: 0 }} barCategoryGap="20%">
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} domain={[0, 100]} tickFormatter={v => `${v}%`} axisLine={false} tickLine={false} />
                <Tooltip
                  cursor={{ fill: 'rgba(15,23,42,0.04)' }}
                  formatter={(v: number, name: string, props: any) => [`${v}%`, props.payload.name]}
                  contentStyle={{
                    background: 'var(--ns-bg-overlay)',
                    border: '1px solid var(--ns-border)',
                    borderRadius: '8px',
                    color: 'var(--ns-text-primary)',
                    boxShadow: '0 4px 20px rgba(15,23,42,0.08)'
                  }}
                />
                <Bar dataKey="value" radius={[4,4,0,0]} animationDuration={1500} animationEasing="ease-out">
                  {performanceChartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.fill} 
                      fillOpacity={0.85} 
                      style={{ filter: `drop-shadow(0 0 8px ${entry.fill}40)` }}
                    />
                  ))}
                  <LabelList 
                    dataKey="value" 
                    position="top" 
                    formatter={(val: number) => `${val.toFixed(2)}%`} 
                    fill="var(--ns-text-primary)" 
                    fontSize={11} 
                    fontWeight="bold" 
                    offset={8}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs mt-3 text-center" style={{ color: 'var(--ns-text-muted)' }}>
              Real-time evaluation metrics from the currently deployed model
            </p>
          </section>
        )}

        {/* ── Model Snapshot + Prediction Logs ── */}
        <section
          className="grid lg:grid-cols-2 gap-5 mb-6 animate-fade-in-up"
          style={{ animationDelay: '0.2s' }}
        >
          {/* Model Info Card */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider mb-4" style={{ color: 'var(--ns-text-secondary)' }}>
              Model Snapshot
            </h3>
            {loading ? (
              <div className="space-y-3">
                {[1,2,3,4,5].map(i => <Skeleton key={i} className="h-5" />)}
              </div>
            ) : systemInfo ? (
              <div className="space-y-2">
                {[
                  { label: 'Model', value: systemInfo.model_name?.replace('_', ' ') ?? 'Unavailable' },
                  { label: 'Version', value: systemInfo.model_version ?? 'Unavailable' },
                  { label: 'Dataset Size', value: systemInfo.dataset_size ? `${systemInfo.dataset_size.toLocaleString()} records` : 'Unavailable' },
                  { label: 'Feature Count', value: systemInfo.feature_count?.toString() ?? 'Unavailable' },
                  { label: 'Last Trained', value: formatTimestamp(systemInfo.training_timestamp) },
                  { label: 'Inference Latency', value: formatLatency(systemInfo.inference_latency_ms) },
                  { label: 'Training Duration', value: systemInfo.training_duration_s ? `${systemInfo.training_duration_s.toFixed(1)}s` : 'Unavailable' },
                  { label: 'ROC AUC', value: systemInfo.roc_auc ? formatPercent(systemInfo.roc_auc) : 'Unavailable' },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between items-center py-1.5 border-b last:border-0" style={{ borderColor: 'var(--ns-border)' }}>
                    <span className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>{label}</span>
                    <span className="text-xs font-medium mono" style={{ color: 'var(--ns-text-secondary)' }}>{value}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>No model data available.</p>
            )}
          </div>

          {/* Prediction Logs */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold uppercase tracking-wider" style={{ color: 'var(--ns-text-secondary)' }}>
                Prediction Audit Log
              </h3>
              <span className="text-xs mono" style={{ color: 'var(--ns-text-muted)' }}>
                {logs.length} entries
              </span>
            </div>

            {logs.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-48 text-center">
                <svg
                  className="w-8 h-8 mb-3"
                  style={{ color: 'var(--ns-text-muted)', opacity: 0.4 }}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                  No predictions yet
                </p>
                <p className="text-xs mt-1" style={{ color: 'var(--ns-text-muted)', opacity: 0.6 }}>
                  Assess a claim to generate entries
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--ns-border)' }}>
                      {['#', 'Age', 'Confidence', 'Verdict', 'Latency'].map(h => (
                        <th
                          key={h}
                          className="py-2 px-2 text-left font-bold uppercase tracking-wider"
                          style={{ color: 'var(--ns-text-muted)' }}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...logs].reverse().slice(0, 10).map(log => (
                      <tr
                        key={log.id}
                        style={{ borderBottom: '1px solid var(--ns-border)' }}
                        className="hover:bg-[rgba(15,23,42,0.02)] transition-colors"
                      >
                        <td className="py-2 px-2 mono" style={{ color: 'var(--ns-text-muted)' }}>
                          {log.id}
                        </td>
                        <td className="py-2 px-2 mono" style={{ color: 'var(--ns-text-secondary)' }}>
                          {log.age}
                        </td>
                        <td className="py-2 px-2">
                          <div className="flex items-center gap-2">
                            <div
                              className="h-1 rounded-full overflow-hidden"
                              style={{ width: '40px', background: 'rgba(15,23,42,0.06)' }}
                            >
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${log.confidence}%`,
                                  background: log.result === 'APPROVED'
                                    ? 'linear-gradient(90deg, #10b981, #00d2ff)'
                                    : 'linear-gradient(90deg, #ef4444, #f59e0b)',
                                }}
                              />
                            </div>
                            <span className="mono" style={{ color: 'var(--ns-text-secondary)' }}>
                              {log.confidence.toFixed(1)}%
                            </span>
                          </div>
                        </td>
                        <td className="py-2 px-2">
                          <span
                            className={`px-2 py-0.5 rounded text-xs font-bold ${
                              log.result === 'APPROVED' ? 'badge-approved' : 'badge-rejected'
                            }`}
                          >
                            {log.result}
                          </span>
                        </td>
                        <td className="py-2 px-2 mono" style={{ color: 'var(--ns-text-muted)' }}>
                          {log.inference_latency_ms?.toFixed(1)}ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        {/* ── AI Observatory Panel ── */}
        <section className="animate-fade-in-up" style={{ animationDelay: '0.25s' }}>
          <AIObservatory
            systemInfo={systemInfo}
            modelInsights={modelInsights}
            loading={loading}
          />
        </section>

      </main>
    </>
  )
}