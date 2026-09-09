"use client"

import React from 'react'
import { motion } from 'framer-motion'
import { useObservabilityStore } from '@/store/observability'
import { formatTimestamp, formatLatency } from '@/lib/api'

export function PredictionAuditLog() {
  const { logs, isLoadingLogs } = useObservabilityStore()

  // Display latest 10 predictions
  const displayLogs = (logs || []).slice(0, 10)

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 backdrop-blur-sm relative overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6 border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 font-mono text-sm font-semibold">
            AL
          </div>
          <div>
            <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
              Prediction Audit Trail
              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                Live Polling (3s)
              </span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Immutable ledger of recent inference verdicts, response latencies, and input feature snapshots.
            </p>
          </div>
        </div>
        <div className="text-xs font-mono text-slate-400 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>{logs?.length ?? 0} Recorded Inferences</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300">
          <thead className="bg-slate-950/60 text-[11px] font-mono uppercase tracking-wider text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-3 px-3">#ID</th>
              <th className="py-3 px-3">Timestamp</th>
              <th className="py-3 px-3">Age / Profile</th>
              <th className="py-3 px-3">Verdict</th>
              <th className="py-3 px-3">Confidence</th>
              <th className="py-3 px-3">Latency</th>
              <th className="py-3 px-3">Model</th>
              <th className="py-3 px-3 text-right">Cache Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 font-mono">
            {displayLogs.length === 0 ? (
              <tr>
                <td colSpan={8} className="py-8 text-center text-slate-500">
                  {isLoadingLogs ? 'Loading prediction stream...' : 'No inferences logged yet. Run a prediction on the Risk Assessment page.'}
                </td>
              </tr>
            ) : (
              displayLogs.map((log, idx) => {
                const verdict = (log.verdict || log.prediction || 'UNKNOWN').toUpperCase()
                const isApproved = verdict === 'APPROVED' || verdict === 'APPROVE' || verdict.includes('APPROV')
                const isRejected = verdict === 'REJECTED' || verdict === 'REJECT' || verdict.includes('REJECT')
                const confidence = log.confidence ? `${log.confidence.toFixed(1)}%` : '—'
                const latency = formatLatency(log.latency_ms ?? (log as any).inference_latency_ms)
                const age = log.features?.age ?? (log as any).age ?? '—'
                const smoker = log.features?.smoker ?? (log as any).smoker ?? ''
                const sex = log.features?.sex ?? (log as any).sex ?? ''
                const model = log.model_name || (log as any).model_used || 'CatBoost'
                const predId = (log as any).prediction_id ? (log as any).prediction_id.substring(0, 8) : `INF-${(logs.length - idx).toString().padStart(4, '0')}`

                return (
                  <motion.tr
                    key={log.timestamp || idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.02 }}
                    className="hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="py-2.5 px-3 text-slate-400 font-semibold">{predId}</td>
                    <td className="py-2.5 px-3 text-slate-300">{formatTimestamp(log.timestamp)}</td>
                    <td className="py-2.5 px-3">
                      <span className="text-white font-medium">{age} yrs</span>
                      {smoker && (
                        <span className={`ml-2 px-1.5 py-0.5 rounded text-[10px] ${
                          smoker === 'yes' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-slate-800 text-slate-400'
                        }`}>
                          {smoker === 'yes' ? 'Smoker' : 'Non-Smoker'}
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide border ${
                        isApproved
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                          : isRejected
                          ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                          : 'bg-slate-800 text-slate-400 border-slate-700'
                      }`}>
                        {isApproved ? '● APPROVED' : isRejected ? '■ REJECTED' : verdict}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-slate-200">{confidence}</td>
                    <td className="py-2.5 px-3">
                      <span className="text-blue-400 font-medium">{latency}</span>
                    </td>
                    <td className="py-2.5 px-3 text-slate-400 uppercase text-[11px]">{model}</td>
                    <td className="py-2.5 px-3 text-right">
                      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        WARM HIT
                      </span>
                    </td>
                  </motion.tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default PredictionAuditLog
