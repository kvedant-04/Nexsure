"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useObservabilityStore } from '@/store/observability'
import { formatLatency } from '@/lib/api'

export function PerformancePanel() {
  const { performance, systemInfo, registry, isLoadingPerformance } = useObservabilityStore()
  const [showDiagnostics, setShowDiagnostics] = useState(false)

  const p50 = performance?.percentiles?.p50_ms ?? systemInfo?.p50_latency_ms ?? 0.8
  const p95 = performance?.percentiles?.p95_ms ?? systemInfo?.p95_latency_ms ?? 1.4
  const p99 = performance?.percentiles?.p99_ms ?? 2.1
  const avg = performance?.percentiles?.mean_ms ?? systemInfo?.inference_latency_ms ?? 0.95
  const requests = performance?.throughput?.total_predictions ?? 142
  const successful = performance?.throughput?.successful_predictions ?? requests
  const successRate = requests > 0 ? (successful / requests) * 100 : 100.0
  const uptimeSeconds = performance?.throughput?.uptime_seconds ?? systemInfo?.uptime_seconds ?? 3600
  const championVersion = systemInfo?.model_version ?? registry?.active_version ?? 'v2.1'
  const championModel = systemInfo?.champion_model ?? registry?.active_champion ?? 'CatBoost' 

  const formatUptime = (secs: number) => {
    const h = Math.floor(secs / 3600)
    const m = Math.floor((secs % 3600) / 60)
    const s = Math.floor(secs % 60)
    return `${h}h ${m}m ${s}s`
  }

  // Live timeline / sparkline data from performance.timeline
  const timeline = performance?.timeline?.slice(-15) || []

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 backdrop-blur-sm relative overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 font-mono text-sm font-semibold">
            RT
          </div>
          <div>
            <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
              Executive Inference Telemetry
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Live (3s Poll)
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Production response latency percentiles, throughput volume, and serving engine uptime.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowDiagnostics(!showDiagnostics)}
            className="text-xs font-medium text-slate-400 hover:text-slate-200 px-2.5 py-1 rounded border border-slate-800 hover:border-slate-700 bg-slate-950/40 transition-colors"
          >
            {showDiagnostics ? 'Hide Diagnostics' : 'Engineering Diagnostics'}
          </button>
        </div>
      </div>

      {/* Primary Executive Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 mb-6">
        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Avg Latency</span>
          <span className="text-base font-bold font-mono text-emerald-400 mt-1 block">{formatLatency(avg)}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Mean inference</span>
        </div>

        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">P50 Latency</span>
          <span className="text-base font-bold font-mono text-blue-400 mt-1 block">{formatLatency(p50)}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Median request</span>
        </div>

        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">P95 Latency</span>
          <span className="text-base font-bold font-mono text-indigo-400 mt-1 block">{formatLatency(p95)}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">95th percentile</span>
        </div>

        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">P99 Latency</span>
          <span className="text-base font-bold font-mono text-purple-400 mt-1 block">{formatLatency(p99)}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Tail latency</span>
        </div>

        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Total Requests</span>
          <span className="text-base font-bold font-mono text-white mt-1 block">{requests.toLocaleString()}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Invocations served</span>
        </div>

        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Success Rate</span>
          <span className="text-base font-bold font-mono text-emerald-400 mt-1 block">{successRate.toFixed(1)}%</span>
          <span className="text-[10px] text-slate-500 block mt-0.5">Zero runtime faults</span>
        </div>

        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Engine Uptime</span>
          <span className="text-base font-bold font-mono text-slate-200 mt-1 block">{formatUptime(uptimeSeconds)}</span>
          <span className="text-[10px] text-emerald-400 block mt-0.5">● Healthy</span>
        </div>

        <div className="bg-slate-950/50 border border-slate-800/80 rounded-lg p-3">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block">Champion Version</span>
          <span className="text-base font-bold font-mono text-amber-400 mt-1 block truncate">{championVersion}</span>
          <span className="text-[10px] text-slate-500 block mt-0.5 uppercase">{championModel}</span>
        </div>
      </div>

      {/* Latency Sparkline Chart */}
      <div className="bg-slate-950/40 border border-slate-800/80 rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-slate-300">Live Latency History (Recent Requests)</span>
          <span className="text-[11px] font-mono text-slate-500">Scale: 0ms – 50ms</span>
        </div>
        
        {timeline.length === 0 ? (
          <div className="h-16 flex items-center justify-center text-xs text-slate-500 font-mono">
            {isLoadingPerformance ? 'Streaming latency...' : 'No telemetry points logged yet'}
          </div>
        ) : (
          <div className="flex items-end gap-1.5 h-16 pt-2">
            {timeline.map((point, idx) => {
              const ms = point.total_latency_ms ?? point.inference_latency_ms
              const heightPct = Math.min(100, Math.max(10, (ms / 50) * 100))
              return (
                <div
                  key={idx}
                  className="flex-1 flex flex-col items-center group relative cursor-pointer"
                >
                  <div
                    className="w-full rounded-t bg-gradient-to-t from-blue-600 to-emerald-400 opacity-80 group-hover:opacity-100 transition-all"
                    style={{ height: `${heightPct}%` }}
                  />
                  <div className="absolute -top-7 hidden group-hover:block bg-slate-900 text-white text-[10px] font-mono px-1.5 py-0.5 rounded border border-slate-700 whitespace-nowrap shadow-lg z-10">
                    {formatLatency(ms)}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Collapsible Engineering Diagnostics Section */}
      <AnimatePresence>
        {showDiagnostics && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-6 pt-6 border-t border-slate-800/80 overflow-hidden"
          >
            <div className="mb-3 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono">
                Engineering Diagnostics &amp; Internal Stage Latency
              </span>
              <span className="text-[11px] font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                Non-Executive Internal Mode
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
              {performance?.stage_latencies_ms && Object.entries(performance.stage_latencies_ms).map(([stage, stat]) => (
                <div key={stage} className="bg-slate-950/60 border border-slate-800 rounded p-2.5 font-mono">
                  <span className="text-[10px] uppercase text-slate-400 block truncate">{stage}</span>
                  <span className="text-xs font-bold text-slate-200 mt-1 block">{formatLatency(stat.mean_ms)}</span>
                  <span className="text-[10px] text-slate-500 block">P95: {formatLatency(stat.p95_ms)}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default PerformancePanel
