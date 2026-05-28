'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  SystemInfoResponse,
  ModelInsightsResponse,
  formatPercent,
  formatLatency,
  formatTimestamp,
} from '@/lib/api'
import { useObservabilityStore } from '@/store/observability'

interface Props {
  systemInfo:    SystemInfoResponse | null
  modelInsights: ModelInsightsResponse | null
  loading?:      boolean
}

function MetricRow({
  label,
  value,
  highlight = false,
}: {
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className="flex items-baseline justify-between py-1.5 border-b last:border-0" style={{ borderColor: 'var(--ns-border)' }}>
      <span className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>
        {label}
      </span>
      <span
        className="text-xs font-medium mono"
        style={{ color: highlight ? 'var(--ns-electric)' : 'var(--ns-text-secondary)' }}
      >
        {value}
      </span>
    </div>
  )
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  if (!matrix || matrix.length < 2) return null
  const [[tn, fp], [fn, tp]] = matrix
  const total = tn + fp + fn + tp

  const getIntensity = (val: number) => {
    if (total === 0) return 0
    return Math.max(0.1, val / total)
  }

  const cells = [
    { label: 'True Neg.', abbr: 'TN', value: tn, intent: 'correct' },
    { label: 'False Pos.', abbr: 'FP', value: fp, intent: 'incorrect' },
    { label: 'False Neg.', abbr: 'FN', value: fn, intent: 'incorrect' },
    { label: 'True Pos.', abbr: 'TP', value: tp, intent: 'correct' },
  ]

  return (
    <div>
      <p className="text-xs uppercase tracking-widest font-bold mb-3" style={{ color: 'var(--ns-text-muted)' }}>
        Evaluation Grid (Confusion Matrix)
      </p>
      <div className="grid grid-cols-2 gap-3">
        {cells.map((cell, i) => {
          const intensity = getIntensity(cell.value)
          const baseColor = cell.intent === 'correct' ? 'var(--ns-electric-rgb, 0, 132, 255)' : 'var(--ns-slate-rgb, 100, 116, 139)'
          const bgColor = `rgba(${baseColor}, ${intensity * 0.25})`
          const borderColor = `rgba(${baseColor}, ${intensity * 0.5 + 0.2})`
          const textColor = cell.intent === 'correct' ? 'var(--ns-electric)' : 'var(--ns-text-secondary)'

          return (
            <motion.div
              key={cell.label}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
              className="rounded-xl p-4 text-center transition-all"
              style={{
                background: bgColor,
                border: `1px solid ${borderColor}`,
                boxShadow: `inset 0 1px 0 rgba(255,255,255,0.05)`
              }}
            >
              <div className="text-2xl font-bold mono" style={{ color: textColor }}>
                {cell.value}
              </div>
              <div className="text-xs font-bold tracking-widest mt-1" style={{ color: textColor }}>
                {cell.abbr}
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--ns-text-muted)' }}>
                {cell.label}
              </div>
            </motion.div>
          )
        })}
      </div>
      <div className="flex justify-between mt-3 px-2 text-xs" style={{ color: 'var(--ns-text-muted)' }}>
        <span>Predicted Positive ↗</span>
        <span>Predicted Negative ↘</span>
      </div>
    </div>
  )
}

import { ShapFeature } from '@/lib/api'

function DecisionDriverCard({ driver, index }: { driver: ShapFeature, index: number }) {
  const isPos = driver.impact_dir === 'increases'
  const color = isPos ? '#10b981' : '#f59e0b'
  const bg = isPos ? 'rgba(16,185,129,0.05)' : 'rgba(245,158,11,0.05)'
  const border = isPos ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)'
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1, ease: [0.16, 1, 0.3, 1] }}
      className="p-5 rounded-xl transition-all hover:bg-[rgba(15,23,42,0.02)]"
      style={{
        background: bg,
        border: `1px solid ${border}`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.03)`
      }}
    >
      <div className="flex justify-between items-start mb-3">
        <div>
          <h4 className="text-sm font-bold" style={{ color: 'var(--ns-text-primary)' }}>
            {driver.feature}
          </h4>
          <span className="text-xs" style={{ color }}>
            {driver.impact_level} {isPos ? 'Positive' : 'Negative'} Influence
          </span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-sm font-bold mono" style={{ color }}>
            {isPos ? '+' : '-'}{driver.impact_pct.toFixed(1)}%
          </span>
          <span className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>
            Impact
          </span>
        </div>
      </div>
      
      {/* Visual pulse indicator */}
      <div className="flex items-center gap-2 mb-3">
        <div className="w-1.5 h-1.5 rounded-full animate-status-pulse" style={{ background: color }} />
        <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(15,23,42,0.06)' }}>
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: `${driver.impact_pct}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="h-full rounded-full" 
            style={{ background: `linear-gradient(90deg, ${color}50, ${color})` }} 
          />
        </div>
      </div>

      <p className="text-xs leading-relaxed italic" style={{ color: 'var(--ns-text-secondary)' }}>
        "{driver.insight}"
      </p>
    </motion.div>
  )
}

function LiveShapPanel({ shap }: { shap: any }) {
  if (!shap) return null

  return (
    <div className="animate-fade-in-up">
      <div className="flex items-center justify-between mb-5">
        <p className="text-xs uppercase tracking-widest font-bold" style={{ color: 'var(--ns-electric)' }}>
          Live Inference Analysis
        </p>
        <span
          className="text-xs px-2 py-0.5 rounded font-bold uppercase tracking-wider"
          style={{
            background: 'rgba(0,132,255,0.08)',
            color: 'var(--ns-electric)',
            border: '1px solid rgba(0,132,255,0.2)'
          }}
        >
          Real-time
        </span>
      </div>
      
      {shap.executive_summary && (
        <div className="mb-6 p-4 rounded-xl" style={{ background: 'rgba(15,23,42,0.03)', border: '1px solid var(--ns-border)' }}>
          <p className="text-sm font-medium" style={{ color: 'var(--ns-text-primary)' }}>
            {shap.executive_summary}
          </p>
        </div>
      )}

      <div className="space-y-6">
        {/* Positive Drivers */}
        {shap.top_positive_drivers?.length > 0 && (
          <div>
            <p className="text-xs font-semibold mb-3" style={{ color: 'var(--ns-text-secondary)' }}>
              Primary Approval Drivers
            </p>
            <div className="grid sm:grid-cols-2 gap-4">
              {shap.top_positive_drivers.map((c: ShapFeature, i: number) => (
                <DecisionDriverCard key={c.feature} driver={c} index={i} />
              ))}
            </div>
          </div>
        )}

        {/* Negative Drivers */}
        {shap.top_negative_drivers?.length > 0 && (
          <div className="pt-2">
            <p className="text-xs font-semibold mb-3" style={{ color: 'var(--ns-text-secondary)' }}>
              Counter-Risk Indicators
            </p>
            <div className="grid sm:grid-cols-2 gap-4">
              {shap.top_negative_drivers.map((c: ShapFeature, i: number) => (
                <DecisionDriverCard key={c.feature} driver={c} index={i} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function PredictionTimeline({ predictions }: { predictions: any[] }) {
  if (!predictions || predictions.length === 0) return null

  return (
    <div className="mt-6">
      <p className="text-xs uppercase tracking-widest font-bold mb-3" style={{ color: 'var(--ns-text-muted)' }}>
        Prediction Timeline
      </p>
      <div className="space-y-2">
        {predictions.map((p, i) => {
          const approved = p.prediction === 'APPROVED'
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-center justify-between p-3 rounded-lg border transition-colors hover:bg-[rgba(15,23,42,0.04)]"
              style={{ background: 'rgba(15,23,42,0.02)', borderColor: 'var(--ns-border)' }}
            >
              <div className="flex items-center gap-3">
                <span
                  className="w-2 h-2 rounded-full"
                  style={{
                    background: approved ? '#10b981' : '#ef4444',
                    boxShadow: `0 0 6px ${approved ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.5)'}`
                  }}
                />
                <span className="text-xs font-bold" style={{ color: 'var(--ns-text-primary)' }}>{p.prediction}</span>
                <span className="text-xs mono" style={{ color: 'var(--ns-text-muted)' }}>{p.confidence.toFixed(1)}%</span>
              </div>
              <div className="flex items-center gap-4 text-xs mono">
                {p.shap && p.shap.top_positive_drivers?.length > 0 && (
                  <span style={{ color: 'var(--ns-text-muted)' }}>
                    Top: {p.shap.top_positive_drivers[0].feature}
                  </span>
                )}
                <span style={{ color: 'var(--ns-electric)' }}>
                  {p.inference_latency_ms?.toFixed(1)}ms
                </span>
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

export default function AIObservatory({ systemInfo, modelInsights, loading }: Props) {
  const [expanded, setExpanded] = useState(false)
  const { latestPrediction, recentPredictions } = useObservabilityStore()

  useEffect(() => {
    if (latestPrediction) {
      setExpanded(true)
    }
  }, [latestPrediction])

  const info = systemInfo
  const insights = modelInsights

  // Compute max importance for bar scaling
  const importanceValues = insights?.feature_importance
    ? Object.values(insights.feature_importance)
    : []
  const maxImportance = importanceValues.length > 0 ? Math.max(...importanceValues) : 1

  return (
    <div id="ai-observatory" className="glass-card">
      {/* Toggle Header */}
      <button
        id="observatory-toggle"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-5 text-left hover:bg-[rgba(15,23,42,0.02)] transition-colors"
        aria-expanded={expanded}
        aria-controls="observatory-content"
      >
        <div className="flex items-center gap-3">
          {/* AI Insight Badge */}
          <div
            className="flex items-center gap-2 px-3 py-1 rounded-md animate-glow-pulse"
            style={{
              background: 'rgba(0,90,194,0.08)',
              border: '1px solid rgba(0,90,194,0.25)',
            }}
          >
            <div
              className="w-1.5 h-1.5 rounded-full animate-status-pulse"
              style={{ background: 'var(--ns-electric)' }}
            />
            <span
              className="text-xs font-bold uppercase tracking-widest"
              style={{ color: 'var(--ns-electric)' }}
            >
              Observatory
            </span>
          </div>
          <span className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>
            AI Diagnostics &amp; System Intelligence
          </span>
        </div>

        {/* Chevron */}
        <svg
          className="w-4 h-4 flex-shrink-0 transition-transform duration-300"
          style={{
            color: 'var(--ns-text-muted)',
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
          }}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expandable Content */}
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            id="observatory-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <hr className="ns-divider mx-5 my-0" />

            <div className="p-5 space-y-6 animate-fade-in-up">

              {/* ── Current Model Snapshot ── */}
              <div>
                <p
                  className="text-xs uppercase tracking-widest font-bold mb-3"
                  style={{ color: 'var(--ns-text-muted)' }}
                >
                  Current Model Snapshot
                </p>
                <div
                  className="rounded-xl p-4 space-y-1"
                  style={{
                    background: 'rgba(15,23,42,0.02)',
                    border: '1px solid var(--ns-border)',
                  }}
                >
                  {loading ? (
                    <div className="space-y-2">
                      {[1, 2, 3, 4, 5].map(i => (
                        <div key={i} className="h-4 rounded animate-shimmer" />
                      ))}
                    </div>
                  ) : info ? (
                    <>
                      <MetricRow label="Model Name"       value={info.model_name ?? 'Unavailable'} highlight />
                      <MetricRow label="Version"          value={info.model_version ?? 'Unavailable'} />
                      <MetricRow label="Health Status"    value={info.model_health?.toUpperCase() ?? 'Unavailable'} highlight />
                      <MetricRow label="Dataset Size"     value={info.dataset_size ? `${info.dataset_size.toLocaleString()} records` : 'Unavailable'} />
                      <MetricRow label="Feature Count"    value={info.feature_count?.toString() ?? 'Unavailable'} />
                      <MetricRow label="Last Trained"     value={formatTimestamp(info.training_timestamp)} />
                      <MetricRow label="Training Duration" value={info.training_duration_s ? `${info.training_duration_s.toFixed(1)}s` : 'Unavailable'} />
                      <MetricRow label="Inference Latency" value={formatLatency(info.inference_latency_ms)} highlight />
                      <MetricRow label="ROC AUC"          value={info.roc_auc ? formatPercent(info.roc_auc) : 'Unavailable'} />
                    </>
                  ) : (
                    <p className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>No model data available.</p>
                  )}
                </div>
              </div>

              {/* ── Performance Metrics ── */}
              <div>
                <p className="text-xs uppercase tracking-widest font-bold mb-3" style={{ color: 'var(--ns-text-muted)' }}>
                  Performance Metrics
                </p>
                {loading ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[1,2,3,4].map(i => <div key={i} className="h-20 rounded-xl animate-shimmer" />)}
                  </div>
                ) : info ? (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {[
                      { label: 'Accuracy',  value: info.accuracy,  color: 'var(--ns-electric)' },
                      { label: 'Precision', value: info.precision, color: 'var(--ns-electric)' },
                      { label: 'Recall',    value: info.recall,    color: 'var(--ns-electric)' },
                      { label: 'F1 Score',  value: info.f1_score,  color: 'var(--ns-electric)' },
                    ].map(({ label, value, color }, i) => (
                      <motion.div
                        key={label}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
                        className="rounded-xl p-4 text-center"
                        style={{ 
                          background: `rgba(0, 132, 255, 0.05)`, 
                          border: `1px solid rgba(0, 132, 255, 0.15)`
                        }}
                      >
                        <div className="text-xl font-bold mono" style={{ color }}>
                          {value != null ? `${(value * 100).toFixed(1)}%` : '—'}
                        </div>
                        <div className="text-xs mt-1 font-medium" style={{ color: 'var(--ns-text-muted)' }}>
                          {label}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs" style={{ color: 'var(--ns-text-muted)' }}>No metrics available.</p>
                )}
              </div>

              {/* ── Confusion Matrix ── */}
              {info?.confusion_matrix && (
                <ConfusionMatrix matrix={info.confusion_matrix} />
              )}

              {/* ── Feature Importance (Global or Live) ── */}
              {latestPrediction?.shap ? (
                <LiveShapPanel shap={latestPrediction.shap} />
              ) : insights?.feature_importance && Object.keys(insights.feature_importance).length > 0 ? (
                <div>
                  <div className="flex items-center justify-between mb-5">
                    <p className="text-xs uppercase tracking-widest font-bold" style={{ color: 'var(--ns-text-muted)' }}>
                      Global Model Intelligence
                    </p>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    {Object.entries(insights.feature_importance)
                      .sort(([, a], [, b]) => b - a)
                      .slice(0, 6)
                      .map(([feature, importance], i) => (
                        <div key={feature} className="p-4 rounded-xl" style={{ background: 'rgba(15,23,42,0.02)', border: '1px solid var(--ns-border)' }}>
                           <div className="flex justify-between items-center mb-2">
                             <span className="text-sm font-bold" style={{ color: 'var(--ns-text-primary)' }}>{feature.replace('_', ' ').title()}</span>
                             <span className="text-xs mono" style={{ color: 'var(--ns-electric)' }}>{((importance / maxImportance) * 100).toFixed(1)}%</span>
                           </div>
                           <div className="h-1 rounded-full bg-[rgba(15,23,42,0.06)]">
                             <div className="h-full rounded-full" style={{ width: `${(importance / maxImportance) * 100}%`, background: 'var(--ns-electric)' }} />
                           </div>
                        </div>
                      ))}
                  </div>
                  <div className="mt-4 p-4 rounded-xl text-center" style={{ background: 'rgba(15,23,42,0.02)', border: '1px dashed var(--ns-border)' }}>
                    <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                      Live model reasoning becomes available after an active underwriting assessment.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="mt-4 p-4 rounded-xl text-center" style={{ background: 'rgba(15,23,42,0.02)', border: '1px dashed var(--ns-border)' }}>
                  <p className="text-sm" style={{ color: 'var(--ns-text-muted)' }}>
                    Live model reasoning becomes available after an active underwriting assessment.
                  </p>
                </div>
              )}

              {/* ── Prediction Timeline ── */}
              <PredictionTimeline predictions={recentPredictions} />

              {/* ── Model Comparison ── */}
              {insights?.all_model_metrics && Object.keys(insights.all_model_metrics).length > 0 && (
                <div>
                  <p className="text-xs uppercase tracking-widest font-bold mb-3" style={{ color: 'var(--ns-text-muted)' }}>
                    All Model Comparison
                  </p>
                  <div
                    className="rounded-xl overflow-x-auto"
                    style={{ border: '1px solid var(--ns-border)' }}
                  >
                    <table className="w-full text-xs min-w-[400px]">
                      <thead>
                        <tr style={{ background: 'rgba(15,23,42,0.04)' }}>
                          {['Model', 'Accuracy', 'F1', 'Latency'].map(h => (
                            <th
                              key={h}
                              className="px-3 py-2 text-left font-bold uppercase tracking-wider"
                              style={{ color: 'var(--ns-text-muted)' }}
                            >
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(insights.all_model_metrics).map(([name, m], idx) => (
                          <tr
                            key={name}
                            style={{
                              background: idx % 2 === 0 ? 'transparent' : 'rgba(15,23,42,0.02)',
                              borderTop: '1px solid var(--ns-border)',
                            }}
                          >
                            <td className="px-3 py-2 mono" style={{ color: 'var(--ns-text-secondary)' }}>
                              {name.replace('_', ' ')}
                            </td>
                            <td className="px-3 py-2 mono" style={{ color: 'var(--ns-electric)' }}>
                              {formatPercent(m.accuracy)}
                            </td>
                            <td className="px-3 py-2 mono" style={{ color: 'var(--ns-electric)' }}>
                              {formatPercent(m.f1_score)}
                            </td>
                            <td className="px-3 py-2 mono" style={{ color: 'var(--ns-text-muted)' }}>
                              {formatLatency(m.inference_latency_ms)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Footer */}
              <p className="text-xs text-center" style={{ color: 'var(--ns-text-muted)', opacity: 0.5 }}>
                All metrics derived from real-time model evaluation · No synthetic data
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
